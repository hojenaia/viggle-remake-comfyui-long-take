# Viggle Remake: long takes for ComfyUI

A ComfyUI custom node for rendering a continuous Viggle Remake swap across multiple bounded MiniMax H3 renders. Each later render receives the previous take's final 17 frames as locked VAE latents. The overlap is removed when the rendered windows are joined.

## Contents

- `nodes.py` and `__init__.py`: the original long video node plus a pins-based variant. Both use helper classes from the separately installed official Viggle Remake package.
- `street-dance-long-api.json`: API-format example for a continuous solo dance. The source video and character reference image are not included.
- `workflows/viggle-long-take-pin-generator.json`: Qwen Image 2.1 workflow that selects the source frames needed for a long take and generates one pin for each in sequence.
- `workflows/viggle-long-take-unified.json`: combined MiniMax H3 Image pin-generation and Viggle Remake long-video workflow.

## Install

Install the official `ComfyUI-Viggle-Remake` custom node first. Then clone this repository directly inside `ComfyUI/custom_nodes/` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hojenaia/viggle-remake-comfyui-long-take.git
```

The official Viggle Remake package provides the original Conditioning, Keyframe Until, and Noise nodes; this repository registers **Viggle Remake Long Video** and **Viggle Remake Long Video (Pins)**. Confirm that both appear in the node list. The model files and LoRA are not included.

## Node inputs

Connect the MiniMax H3 model with the Viggle Remake LoRA, CLIP, video and audio VAEs, source frames, character reference, sampler, and sigmas. The output provides generated images and the source audio. If no audio is connected, the output is silent.

`window_frames` defaults to 294 and can be raised to 345 if the GPU has enough memory. Source footage must be 24 fps. The node is intended for one continuous shot; it does not detect cuts or rewrite timestamped prompts for each window. Use prompts that describe actions without absolute timestamps.

## Pins node

Use **Viggle Remake Long Video (Pins)** when you have generated a Qwen-transformed pin for each long-video window. Connect the ordered pin images as an `IMAGE` batch. The expected pins come from source frames `0`, then the first new frame of each later window. With `window_frames=294`, those source frame indexes are `0, 294, 571, 848, ...`; with `345`, they are `0, 345, 673, 1001, ...`. Later-window pins are applied at local frame 17, after the 17-frame overlap.

For pin generation, set `Resolution Selector` to 9:16, 1 MP, multiple 32, resize the source frames to its output size (768 × 1376), and extract the pin frames from that resized sequence. This keeps the source and pins aligned. The long-video node then applies the same center-cover transform to each pin and its source window for the selected `short_edge` output canvas.

The optional `character_sheet` is a global identity reference. The `use_character_sheet` switch controls whether it is included; the per-window pin remains a local scene and pose reference in either mode. If the switch is enabled, connect a character sheet. If disabled, the pins alone condition the character identity and scene.

Both workflows use `Viggle Long Take Pin Frames` to resize the full source sequence to 768 × 1376 and select the frames needed for the long take. The pin-only example edits them with Qwen Image 2.1. The unified workflow instead uses MiniMax H3 Image with the supplied body-swap LoRA. It runs one frame at a time through `BFS Frame Loop Start/End` and collects the pins into a batch. The body reference keeps its original aspect ratio. These workflows require the BFSNodes package that provides the frame-loop nodes.

In the unified workflow, the pins and the long-video node both receive the same resized source sequence, so source frames and pins share the 768 × 1376 canvas. The pins feed directly into **Viggle Remake Long Video (Pins)**, which renders the full source and sends the generated video and source audio to the video output nodes. The same source video, body reference, and `window_frames` setting are shared by both stages. Resizing the full sequence creates an additional in-memory image batch.

## Status

Experimental. The node has been used in local renders, but results depend on the ComfyUI version, available memory, and source material. Review identity and motion continuity at every window seam.

## License

Apache-2.0. See `LICENSE`.
