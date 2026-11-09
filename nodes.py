# SPDX-License-Identifier: Apache-2.0
"""Long-window Viggle Remake swap node for ComfyUI."""

import importlib.util
import os
import sys

import torch
import comfy.nested_tensor
import comfy.utils
import folder_paths
from comfy_extras.nodes_custom_sampler import Guider_Basic


def _load_remake_helpers():
    """Load implementation helpers from the separately installed official Viggle package."""
    for custom_nodes_dir in folder_paths.get_folder_paths("custom_nodes"):
        helper_path = os.path.join(custom_nodes_dir, "ComfyUI-Viggle-Remake", "nodes.py")
        if not os.path.isfile(helper_path):
            continue
        module_name = "_viggle_remake_official_nodes"
        spec = importlib.util.spec_from_file_location(module_name, helper_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError(
        "Viggle Remake Long Video requires the official ComfyUI-Viggle-Remake "
        "custom node installed in ComfyUI/custom_nodes/"
    )


_remake = _load_remake_helpers()
RemakeConditioning = _remake.RemakeConditioning
Noise = _remake.Noise
FPS = _remake.FPS

class RemakeLongVideo:
    """Render a continuous take in bounded windows, carrying a 17-frame latent prefix."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "source": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "ref_1": ("IMAGE",),
                "pins": ("STRING", {"default": "", "tooltip": "Optional pinned frames in the first window only, e.g. '12'."}),
                "window_frames": ("INT", {"default": 294, "min": 90, "max": 345, "step": 17,
                                           "tooltip": "Maximum render length per hop; lower this if your GPU runs out of memory."}),
                "short_edge": ("INT", {"default": 480, "min": 480, "max": 736, "step": 32}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                  "control_after_generate": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "ref_2": ("IMAGE",),
                "ref_3": ("IMAGE",),
                "ref_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"
    CATEGORY = "Viggle Remake"

    def run(self, model, clip, vae, audio_vae, source, prompt, ref_1, pins, window_frames, short_edge, sampler, sigmas, seed,
            audio=None, ref_2=None, ref_3=None, ref_4=None):
        source = source[..., :3]
        total = source.shape[0]
        if total < 1:
            raise ValueError("source has no frames")

        # A render is <=345 frames; each later hop spends its first 17 on the previous take.
        legal = [17 * k + 5 for k in range(5, 21)]  # 90..345
        max_render = max((n for n in legal if n <= window_frames), default=None)
        if max_render is None:
            raise ValueError("window_frames must allow at least 90 frames")
        prefix_frames = 17
        refs = [r for r in (ref_1, ref_2, ref_3, ref_4) if r is not None]
        if pins.strip() and len(pins.replace(",", " ").split()) != len(refs):
            raise ValueError("pins must be empty or give one frame per reference image")
        initial_pins = pins

        out_parts = []
        out_count = 0
        segment = 0
        while out_count < total:
            render_start = 0 if segment == 0 else out_count - prefix_frames
            remaining = total - render_start
            wanted = min(max_render, remaining)
            render_frames = next((n for n in legal if n >= wanted), None)
            if render_frames is None:
                render_frames = max(n for n in legal if n <= max_render)
            # The final window may be shorter than the model floor; repeat its last source frame and crop on output.
            render_frames = max(legal[0], render_frames)
            end = min(total, render_start + render_frames)
            chunk = source[render_start:end]
            if chunk.shape[0] < render_frames:
                chunk = torch.cat([chunk, chunk[-1:].expand(render_frames - chunk.shape[0], -1, -1, -1)], dim=0)

            chunk_audio = None
            if audio is not None:
                sr = audio["sample_rate"]
                wave = audio["waveform"]
                a0 = round(render_start / FPS * sr)
                a1 = round(end / FPS * sr)
                part = wave[..., a0:a1]
                wanted_samples = round(render_frames / FPS * sr)
                if part.shape[-1] < wanted_samples:
                    part = torch.nn.functional.pad(part, (0, wanted_samples - part.shape[-1]))
                chunk_audio = {"waveform": part, "sample_rate": sr}

            local_pins = initial_pins if segment == 0 else ""
            positive, latent = RemakeConditioning().run(
                clip, vae, audio_vae, chunk, prompt, ref_1, local_pins, render_frames, short_edge,
                audio=chunk_audio, ref_2=ref_2, ref_3=ref_3, ref_4=ref_4)
            video_latent, audio_latent = latent["samples"].unbind()
            video_mask = torch.ones_like(video_latent)
            audio_mask = torch.zeros_like(audio_latent)

            if segment:
                prefix = out_parts[-1][-prefix_frames:]
                if prefix.shape[0] != prefix_frames:
                    raise ValueError(f"hop {segment}: previous take has fewer than {prefix_frames} frames")
                # The H3 VAE accepts 17-frame guides by padding to 22; its first five temporal latents
                # correspond to those 17 frames. Lock them through sampling, then discard them at the join.
                guide = torch.cat([prefix, prefix[-1:].expand(5, -1, -1, -1)], dim=0)
                # VAE output is [batch, channels, time, height, width]; retain the first five temporal latents.
                clean = vae.encode(guide)[:, :, :5]
                if clean.shape[2] != 5 or clean.shape[3:] != video_latent.shape[3:]:
                    raise ValueError(f"hop {segment}: prefix VAE latent shape {tuple(clean.shape)} does not fit "
                                     f"the target {tuple(video_latent.shape)}")
                video_latent = video_latent.clone()
                video_latent[:, :, :5] = clean.to(video_latent)
                video_mask[:, :, :5] = 0

            latent["samples"] = comfy.nested_tensor.NestedTensor((video_latent, audio_latent))
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

            c = positive[0][1]
            skip = [b["latent"].shape for b in c.get("minimax_refs", []) + c.get("minimax_keyframes", [])
                    if "latent" in b]
            hop_seed = (int(seed) + segment * 0x9E3779B1) & 0xffffffffffffffff
            noise = Noise(hop_seed, skip)
            guider = Guider_Basic(model)
            guider.set_conds(positive)
            result = guider.sample(noise.generate_noise(latent), latent["samples"], sampler, sigmas,
                                   denoise_mask=latent["noise_mask"], disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                                   seed=hop_seed)
            frames = vae.decode(result.unbind()[0])
            if frames.ndim == 5:
                frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
            if segment:
                frames = frames[prefix_frames:]
            take = min(total - out_count, frames.shape[0])
            if take <= 0:
                raise ValueError(f"hop {segment} produced no new frames")
            out_parts.append(frames[:take])
            out_count += take
            segment += 1

        images = torch.cat(out_parts, dim=0)[:total]
        if audio is None:
            audio = {"waveform": torch.zeros(1, 2, round(total / FPS * 32000)), "sample_rate": 32000}
        else:
            sr = audio["sample_rate"]
            waveform = audio["waveform"]
            audio_samples = round(total / FPS * sr)
            waveform = waveform[..., :audio_samples]
            if waveform.shape[-1] < audio_samples:
                waveform = torch.nn.functional.pad(waveform, (0, audio_samples - waveform.shape[-1]))
            audio = {"waveform": waveform, "sample_rate": sr}
        return (images, audio)



NODE_CLASS_MAPPINGS = {"RemakeLongVideo": RemakeLongVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"RemakeLongVideo": "Viggle Remake Long Video"}
