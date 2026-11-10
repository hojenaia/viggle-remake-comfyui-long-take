# SPDX-License-Identifier: Apache-2.0
"""Long-window Viggle Remake swap node for ComfyUI."""

import importlib.util
import os
import sys

import torch
import comfy.audio
import comfy.nested_tensor
import comfy.utils
import node_helpers
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
_AUDIO_SR = _remake.AUDIO_SR
_AUDIO_LATENT_FPS = _remake.AUDIO_LATENT_FPS


def _resize_reference(image, width, height):
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)


def _cover_reference(image, width, height):
    """Resize and center-crop an image onto the output canvas."""
    h, w = image.shape[1:3]
    scale = max(width / w, height / h)
    scaled_w, scaled_h = max(width, round(w * scale)), max(height, round(h * scale))
    left, top = (scaled_w - width) // 2, (scaled_h - height) // 2
    image = _resize_reference(image, scaled_w, scaled_h)
    return image[:, top:top + height, left:left + width]


def _timestamp_labels(frame_count):
    """Match the official conditioner’s 2 fps video timestamp labels."""
    indexes, cursor = [], 0.0
    while round(cursor) < frame_count:
        if not indexes or round(cursor) > indexes[-1]:
            indexes.append(round(cursor))
        cursor += FPS / 2
    times = [i / 2 for i in range(len(indexes))]
    times += times[-1:] * (len(times) % 2)
    return [(times[i] + times[i + 1]) / 2 for i in range(0, len(times), 2)]


def _long_pin_conditioning(clip, vae, audio_vae, source, prompt, pin, local_pin_frame,
                           frames, short_edge, audio, character_sheet=None):
    """Build official Remake conditioning with a global sheet and a per-hop pin."""
    nf = frames
    source = source[..., :3]
    h, w = source.shape[1:3]
    if min(h, w) != short_edge or h % 32 or w % 32:
        long_edge = short_edge * 16 // 9 // 32 * 32
        w, h = (long_edge, short_edge) if w >= h else (short_edge, long_edge)
        source = comfy.utils.common_upscale(source.movedim(-1, 1), w, h, "bicubic", "center").movedim(1, -1)
    source = source[:nf]

    pin = pin[:1, ..., :3]
    sheet = character_sheet[:1, ..., :3] if character_sheet is not None else None
    refs = [image for image in (sheet, pin) if image is not None]
    labels = ["<Video 1>: ", *["<%.1f seconds>" % t for t in _timestamp_labels(nf)]]
    labels.extend("<Picture %d>: " % (index + 1) for index in range(len(refs)))
    tokens = {"qwen3vl_32b": [[token for section in labels + [prompt]
                              for token in clip.tokenize(section)["qwen3vl_32b"][0]]]}
    positive = clip.encode_from_tokens_scheduled(tokens)

    video_latent = vae.encode(source)
    blocks = [{"kind": "video", "latent_t": video_latent.shape[2], "latent_h": h // 16,
               "latent_w": w // 16, "ref_audio_t": 0, "latent": video_latent, "audio_latent": None}]
    for ref in refs:
        rh, rw = ref.shape[1:3]
        target_h = max(32, round(rh * short_edge / min(rh, rw) / 32) * 32)
        target_w = max(32, round(rw * short_edge / min(rh, rw) / 32) * 32)
        resized = ref if (rh, rw) == (target_h, target_w) else _resize_reference(ref, target_w, target_h)
        blocks.append({"kind": "image", "latent_h": target_h // 16, "latent_w": target_w // 16,
                       "latent": vae.encode(resized)})

    # The pin is a reference and a keyframe; the optional sheet is a reference only.
    pin_on_canvas = _cover_reference(pin, w, h)
    keyframes = [{"resolved_frame_index": local_pin_frame, "latent": vae.encode(pin_on_canvas)}]
    conditioning = {"minimax_refs": blocks, "minimax_keyframes": keyframes}
    positive = node_helpers.conditioning_set_values(positive, conditioning)

    samples = round((nf + 1) / FPS * _AUDIO_SR)
    waveform = torch.zeros(2, samples)
    if audio is not None:
        waveform, sample_rate = audio["waveform"][0].float(), audio["sample_rate"]
        waveform = waveform.repeat(2, 1) if waveform.shape[0] == 1 else waveform[:2]
        waveform = waveform[:, :round(nf / FPS * sample_rate)]
        if sample_rate != _AUDIO_SR:
            waveform = comfy.audio.resample(waveform, sample_rate, _AUDIO_SR)
    waveform = torch.nn.functional.pad(waveform, (0, max(0, samples - waveform.shape[1])))[:, :samples]
    waveform = torch.nn.functional.pad(waveform, (0, -samples % 800))
    audio_latent = audio_vae.encode(waveform[None].movedim(1, -1))[..., :round(nf / FPS * _AUDIO_LATENT_FPS)]

    target = torch.zeros(1, 24, (nf - 5) // 17 * 5 + 2, h // 16, w // 16)
    latent = {"samples": comfy.nested_tensor.NestedTensor((target, audio_latent)),
              "noise_mask": comfy.nested_tensor.NestedTensor((torch.ones_like(target), torch.zeros_like(audio_latent)))}
    return positive, latent

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


class RemakeLongVideoPins:
    """Long take node with one transformed pin for each render window."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "source": ("IMAGE", {"tooltip": "24 fps source frames. Resize/crop using the same framing as the pin workflow."}),
                "prompt": ("STRING", {"multiline": True}),
                "pin_images": ("IMAGE", {"tooltip": "Qwen-generated pin images in source order: frame 0, then the first new frame of each later window."}),
                "use_character_sheet": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "window_frames": ("INT", {"default": 294, "min": 90, "max": 345, "step": 17,
                                             "tooltip": "Maximum render length per hop; each later window reuses 17 frames."}),
                "short_edge": ("INT", {"default": 480, "min": 480, "max": 736, "step": 32}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                  "control_after_generate": True}),
            },
            "optional": {
                "character_sheet": ("IMAGE", {"tooltip": "Global appearance reference, included as a reference image but never pinned as a keyframe."}),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"
    CATEGORY = "Viggle Remake"

    def run(self, model, clip, vae, audio_vae, source, prompt, pin_images, use_character_sheet,
            window_frames, short_edge, sampler, sigmas, seed, character_sheet=None, audio=None):
        source = source[..., :3]
        total = source.shape[0]
        if total < 1:
            raise ValueError("source has no frames")
        if use_character_sheet and character_sheet is None:
            raise ValueError("connect character_sheet or turn off use_character_sheet")

        legal = [17 * k + 5 for k in range(5, 21)]
        max_render = max((n for n in legal if n <= window_frames), default=None)
        if max_render is None:
            raise ValueError("window_frames must allow at least 90 frames")
        prefix_frames = 17
        hop_advance = max_render - prefix_frames
        expected_pins = 1 if total <= max_render else 1 + (total - max_render + hop_advance - 1) // hop_advance
        if pin_images.shape[0] != expected_pins:
            raise ValueError(f"expected {expected_pins} pin images for {total} source frames at window_frames={max_render}; got {pin_images.shape[0]}")

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
            render_frames = max(legal[0], render_frames)
            end = min(total, render_start + render_frames)
            chunk = source[render_start:end]
            if chunk.shape[0] < render_frames:
                chunk = torch.cat([chunk, chunk[-1:].expand(render_frames - chunk.shape[0], -1, -1, -1)], dim=0)

            chunk_audio = None
            if audio is not None:
                sample_rate = audio["sample_rate"]
                waveform = audio["waveform"]
                start_sample = round(render_start / FPS * sample_rate)
                end_sample = round(end / FPS * sample_rate)
                part = waveform[..., start_sample:end_sample]
                wanted_samples = round(render_frames / FPS * sample_rate)
                if part.shape[-1] < wanted_samples:
                    part = torch.nn.functional.pad(part, (0, wanted_samples - part.shape[-1]))
                chunk_audio = {"waveform": part, "sample_rate": sample_rate}

            pin = pin_images[segment:segment + 1]
            local_pin_frame = 0 if segment == 0 else prefix_frames
            positive, latent = _long_pin_conditioning(
                clip, vae, audio_vae, chunk, prompt, pin, local_pin_frame,
                render_frames, short_edge, chunk_audio,
                character_sheet=character_sheet if use_character_sheet else None)
            video_latent, audio_latent = latent["samples"].unbind()
            video_mask = torch.ones_like(video_latent)
            audio_mask = torch.zeros_like(audio_latent)

            if segment:
                prefix = out_parts[-1][-prefix_frames:]
                if prefix.shape[0] != prefix_frames:
                    raise ValueError(f"hop {segment}: previous take has fewer than {prefix_frames} frames")
                guide = torch.cat([prefix, prefix[-1:].expand(5, -1, -1, -1)], dim=0)
                clean = vae.encode(guide)[:, :, :5]
                if clean.shape[2] != 5 or clean.shape[3:] != video_latent.shape[3:]:
                    raise ValueError(f"hop {segment}: prefix VAE latent shape {tuple(clean.shape)} does not fit "
                                     f"the target {tuple(video_latent.shape)}")
                video_latent = video_latent.clone()
                video_latent[:, :, :5] = clean.to(video_latent)
                video_mask[:, :, :5] = 0

            latent["samples"] = comfy.nested_tensor.NestedTensor((video_latent, audio_latent))
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
            conditioning = positive[0][1]
            skip = [block["latent"].shape for block in conditioning.get("minimax_refs", []) + conditioning.get("minimax_keyframes", [])
                    if "latent" in block]
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
            sample_rate = audio["sample_rate"]
            waveform = audio["waveform"]
            audio_samples = round(total / FPS * sample_rate)
            waveform = waveform[..., :audio_samples]
            if waveform.shape[-1] < audio_samples:
                waveform = torch.nn.functional.pad(waveform, (0, audio_samples - waveform.shape[-1]))
            audio = {"waveform": waveform, "sample_rate": sample_rate}
        return (images, audio)


class ViggleLongTakePinFrames:
    """Select and resize the source frame needed at the start of each long-take window."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": ("IMAGE", {"tooltip": "All source frames at 24 fps."}),
            "window_frames": ("INT", {"default": 294, "min": 90, "max": 345, "step": 17,
                                        "tooltip": "Use the same value as Viggle Remake Long Video (Pins)."}),
            "width": ("INT", {"forceInput": True}),
            "height": ("INT", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("frames_to_pin", "source_frame_indexes", "resized_source")
    FUNCTION = "select"
    CATEGORY = "Viggle Remake"

    def select(self, source, window_frames, width, height):
        total = int(source.shape[0])
        if total < 1:
            raise ValueError("source has no frames")
        legal = [17 * k + 5 for k in range(5, 21)]
        max_render = max((n for n in legal if n <= window_frames), default=None)
        if max_render is None:
            raise ValueError("window_frames must allow at least 90 frames")
        advance = max_render - 17
        indexes = [0]
        next_index = max_render
        while next_index < total:
            indexes.append(next_index)
            next_index += advance

        resized_source = comfy.utils.common_upscale(source[..., :3].movedim(-1, 1), width, height,
                                                    "lanczos", "disabled").movedim(1, -1)
        frames = resized_source[indexes]
        return (frames, ", ".join(str(index) for index in indexes), resized_source)



NODE_CLASS_MAPPINGS = {"RemakeLongVideo": RemakeLongVideo, "RemakeLongVideoPins": RemakeLongVideoPins,
                       "ViggleLongTakePinFrames": ViggleLongTakePinFrames}
NODE_DISPLAY_NAME_MAPPINGS = {"RemakeLongVideo": "Viggle Remake Long Video",
                              "RemakeLongVideoPins": "Viggle Remake Long Video (Pins)",
                              "ViggleLongTakePinFrames": "Viggle Long Take Pin Frames"}
