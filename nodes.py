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


def _model_with_capstone(model, capstone, start_step=4):
    """Append a locked, temporary capstone only from the configured denoise step onward."""
    patched = model.clone()
    previous = patched.model_options.get("model_function_wrapper")
    warned = {"shape": False, "mask": False}

    def model_wrapper(apply_model, args):
        def invoke(call_args):
            if previous is not None:
                return previous(apply_model, call_args)
            return apply_model(call_args["input"], call_args["timestep"], **call_args["c"])

        cond = args["c"]
        shapes = cond.get("latent_shapes")
        if not shapes:
            return invoke(args)
        transformer_options = cond.get("transformer_options") or {}
        sample_sigmas = transformer_options.get("sample_sigmas")
        step = 0
        if sample_sigmas is not None:
            timestep = float(args["timestep"].flatten()[0])
            step = int(torch.argmin((sample_sigmas.float().cpu() - timestep).abs()))
        if step < start_step:
            return invoke(args)
        window = (cond.get("transformer_options") or {}).get("context_window")
        if window is not None and window.index_list[-1] != window.total_frames - 1:
            return invoke(args)

        parts = comfy.utils.unpack_latents(args["input"], shapes)
        video_index = max(range(len(parts)), key=lambda i: parts[i].ndim)
        video = parts[video_index]
        if (video.ndim != 5 or capstone.ndim != 5 or capstone.shape[1] != video.shape[1]
                or capstone.shape[3:] != video.shape[3:]):
            if not warned["shape"]:
                print(f"[Viggle Long Video] capstone shape {tuple(capstone.shape)} does not fit "
                      f"video latent {tuple(video.shape)}; capstone context skipped")
                warned["shape"] = True
            return invoke(args)

        # MiniMax H3 exposes denoise_mask as a video-only 5D condition. Audio has
        # its own audio_denoise_mask condition, so this mask must not be unpacked
        # with the multimodal latent_shapes (video + audio).
        mask = cond.get("denoise_mask")
        if mask is None:
            if not warned["mask"]:
                print("[Viggle Long Video] no packed denoise mask in model call; capstone context skipped")
                warned["mask"] = True
            return invoke(args)
        if (mask.ndim != 5 or mask.shape[0] != video.shape[0]
                or mask.shape[2:] != video.shape[2:]):
            if not warned["mask"]:
                print("[Viggle Long Video] denoise mask shape does not match video latent; capstone context skipped")
                warned["mask"] = True
            return invoke(args)

        original_t = video.shape[2]
        extra = capstone.to(device=video.device, dtype=video.dtype).expand(video.shape[0], -1, -1, -1, -1)
        parts[video_index] = torch.cat((video, extra), dim=2)
        extended_mask = torch.cat((mask, torch.zeros_like(mask[:, :, :1])), dim=2)
        packed, extended_shapes = comfy.utils.pack_latents(parts)
        extended_cond = dict(cond)
        extended_cond["latent_shapes"] = extended_shapes
        extended_cond["denoise_mask"] = extended_mask
        extended_args = dict(args, input=packed, c=extended_cond)

        output = invoke(extended_args)
        output_parts = comfy.utils.unpack_latents(output, extended_shapes)
        output_parts[video_index] = output_parts[video_index][:, :, :original_t]
        return comfy.utils.pack_latents(output_parts)[0]

    patched.set_model_unet_function_wrapper(model_wrapper)
    return patched

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
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "source": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "window_frames": ("INT", {"default": 294, "min": 90, "max": 345, "step": 17,
                                           "tooltip": "Maximum render length per hop; lower this if your GPU runs out of memory."}),
                "short_edge": ("INT", {"default": 480, "min": 480, "max": 736, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                  "control_after_generate": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "ref_1": ("IMAGE",),
                "ref_2": ("IMAGE",),
                "ref_3": ("IMAGE",),
                "ref_4": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"
    CATEGORY = "Viggle Remake"

    def run(self, model, clip, vae, audio_vae, sampler, sigmas, source, prompt, window_frames, short_edge, seed,
            audio=None, ref_1=None, ref_2=None, ref_3=None, ref_4=None, unique_id=None,
            use_keystone=False, use_capstone=False, capstone_start_step=4):
        source = source[..., :3]
        total = source.shape[0]
        if total < 1:
            raise ValueError("source has no frames")
        if ref_1 is None:
            raise ValueError("ref_1 is required: connect the character reference image")

        # A render is <=345 frames; each later hop spends its first 17 on the previous take.
        legal = [17 * k + 5 for k in range(5, 21)]  # 90..345
        max_render = max((n for n in legal if n <= window_frames), default=None)
        if max_render is None:
            raise ValueError("window_frames must allow at least 90 frames")
        prefix_frames = 17

        # Predict the number of render hops so ComfyUI can show one progress bar
        # spanning all sampler steps instead of restarting it at every hop.
        estimated_out = 0
        total_hops = 0
        while estimated_out < total:
            render_start = 0 if total_hops == 0 else estimated_out - prefix_frames
            remaining = total - render_start
            wanted = min(max_render, remaining)
            predicted_frames = next((n for n in legal if n >= wanted), None)
            if predicted_frames is None:
                predicted_frames = max(n for n in legal if n <= max_render)
            predicted_frames = max(legal[0], predicted_frames)
            produced = predicted_frames if total_hops == 0 else predicted_frames - prefix_frames
            estimated_out += min(total - estimated_out, produced)
            total_hops += 1

        steps_per_hop = max(0, int(sigmas.shape[-1]) - 1)
        progress = comfy.utils.ProgressBar(total_hops * steps_per_hop, node_id=unique_id)

        out_parts = []
        out_count = 0
        segment = 0
        clean_keystone = None
        clean_capstone = None
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

            positive, latent = RemakeConditioning().run(
                clip, vae, audio_vae, chunk, prompt, ref_1, "", render_frames, short_edge,
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
                if use_keystone and clean_keystone is not None:
                    if clean_keystone.shape[1] != video_latent.shape[1] or clean_keystone.shape[3:] != video_latent.shape[3:]:
                        raise ValueError(f"hop {segment}: keystone latent shape {tuple(clean_keystone.shape)} does not fit "
                                         f"the target {tuple(video_latent.shape)}")
                    # The overlap is discarded at the join, so this replaces its first latent only as clean context.
                    video_latent[:, :, :1] = clean_keystone.to(video_latent)

            latent["samples"] = comfy.nested_tensor.NestedTensor((video_latent, audio_latent))
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

            c = positive[0][1]
            skip = [b["latent"].shape for b in c.get("minimax_refs", []) + c.get("minimax_keyframes", [])
                    if "latent" in b]
            hop_seed = (int(seed) + segment * 0x9E3779B1) & 0xffffffffffffffff
            noise = Noise(hop_seed, skip)
            hop_model = model
            if segment and use_capstone and clean_capstone is not None:
                hop_model = _model_with_capstone(model, clean_capstone, capstone_start_step)
                print(f"[Viggle Long Video] using fixed capstone from window 1 in window {segment + 1}, "
                      f"starting at sampler step {capstone_start_step + 1}")
            guider = Guider_Basic(hop_model)
            guider.set_conds(positive)

            def on_step(step, x0, x, total_steps):
                absolute = segment * steps_per_hop + step + 1
                progress.update_absolute(min(absolute, total_hops * steps_per_hop))

            result = guider.sample(noise.generate_noise(latent), latent["samples"], sampler, sigmas,
                                   denoise_mask=latent["noise_mask"], callback=on_step,
                                   disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=hop_seed)
            progress.update_absolute(min((segment + 1) * steps_per_hop, total_hops * steps_per_hop))
            sampled_video = result.unbind()[0]
            if segment == 0 and use_keystone:
                clean_keystone = sampled_video[:, :, :1].detach()
                print("[Viggle Long Video] captured clean first-window keystone context")
            if segment == 0 and use_capstone:
                # Keep the first window's clean capstone as a fixed anchor for all later windows.
                capstone_index = 5 * ((sampled_video.shape[2] - 1) // 5)
                clean_capstone = sampled_video[:, :, capstone_index:capstone_index + 1].detach()
                print("[Viggle Long Video] captured fixed first-window capstone")
            frames = vae.decode(sampled_video)
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


class RemakeLongVideoProtected(RemakeLongVideo):
    """Experimental long-take variant that reuses clean context from its first rendered window."""

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"] = dict(types["required"])
        types["required"]["use_keystone"] = ("BOOLEAN", {"default": True,
            "tooltip": "Reuse the first rendered frame as a locked context latent at the start of every later window."})
        types["required"]["use_capstone"] = ("BOOLEAN", {"default": True,
            "tooltip": "Use the first window's near-final latent as locked temporary context in later windows."})
        types["required"]["capstone_start_step"] = ("INT", {"default": 4, "min": 0, "max": 200,
            "tooltip": "Initial sampler steps without the capstone. With 6 steps, 4 means it is used for the final 2."})
        return types

    FUNCTION = "run_protected"

    def run_protected(self, model, clip, vae, audio_vae, sampler, sigmas, source, prompt, window_frames, short_edge,
                      seed, use_keystone=True, use_capstone=True, capstone_start_step=4, audio=None, ref_1=None,
                      ref_2=None, ref_3=None, ref_4=None, unique_id=None):
        return super().run(model, clip, vae, audio_vae, sampler, sigmas, source, prompt, window_frames, short_edge,
                           seed, audio=audio, ref_1=ref_1, ref_2=ref_2, ref_3=ref_3, ref_4=ref_4,
                           unique_id=unique_id, use_keystone=use_keystone, use_capstone=use_capstone,
                           capstone_start_step=capstone_start_step)

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    CATEGORY = "Viggle Remake"



NODE_CLASS_MAPPINGS = {"RemakeLongVideo": RemakeLongVideo,
                       "RemakeLongVideoProtected": RemakeLongVideoProtected}
NODE_DISPLAY_NAME_MAPPINGS = {"RemakeLongVideo": "Viggle Remake Long Video",
                              "RemakeLongVideoProtected": "Viggle Remake Long Video (Context Protection)"}
