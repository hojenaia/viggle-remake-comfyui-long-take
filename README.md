# Viggle Remake: long takes for ComfyUI

A ComfyUI custom node for rendering a continuous Viggle Remake swap across multiple bounded MiniMax H3 renders. Each later render receives the previous take's final 17 frames as locked VAE latents. The overlap is removed when the rendered windows are joined.

## Contents

- `nodes.py` and `__init__.py`: the long video custom node. It uses the helper classes from the separately installed official Viggle Remake package.
- `street-dance-long-api.json`: API-format example for a continuous solo dance. The source video and character reference image are not included.

## Install

Install the official `ComfyUI-Viggle-Remake` custom node first. Then clone this repository directly inside `ComfyUI/custom_nodes/` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hojenaia/viggle-remake-comfyui-long-take.git
```

The official Viggle Remake package provides the original Conditioning, Keyframe Until, and Noise nodes; this repository registers only **Viggle Remake Long Video**. Confirm that it appears in the node list. The model files and LoRA are not included.

## Node inputs

Connect the MiniMax H3 model with the Viggle Remake LoRA, CLIP, video and audio VAEs, source frames, character reference, sampler, and sigmas. The output provides generated images and the source audio. If no audio is connected, the output is silent.

`window_frames` defaults to 294 and can be raised to 345 if the GPU has enough memory. Source footage must be 24 fps. The node is intended for one continuous shot; it does not detect cuts or rewrite timestamped prompts for each window. Use prompts that describe actions without absolute timestamps.

## Status

Experimental. The node has been used in local renders, but results depend on the ComfyUI version, available memory, and source material. Review identity and motion continuity at every window seam.

## License

Apache-2.0. See `LICENSE`.
