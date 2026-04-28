# 🎬 AI Text-to-Video Generator

### Powered by Zeroscope V2 XL Diffusion Model

## Overview
End-to-end Text-to-Video AI pipeline that converts text prompts into 
short videos using state-of-the-art diffusion models.

## Features
- Text-to-Video generation using Zeroscope V2 XL (576x320)
- Two-stage pipeline: base generation + upscaling
- Prompt enhancement with negative prompting
- Interactive Gradio web UI
- Runs on GPU (T4/A100)

## Tech Stack
- Python, PyTorch
- Diffusers (HuggingFace)
- Zeroscope V2 XL Diffusion Model
- Gradio (Web UI)
- CUDA (GPU Acceleration)
- Kaggle Notebooks

## Sample Prompts
- "A golden retriever playing in autumn leaves, slow motion"
- "Ocean waves crashing on beach at sunset, cinematic"
- "A rocket launching into space with fire and smoke"
- "Snow falling in quiet forest at night, atmospheric"
- ## Demo Output

A cat sitting on a beach at sunset
<img width="721" height="403" alt="image" src="https://github.com/user-attachments/assets/726853d2-b9b0-40e4-9f3d-4a396d6b6a43" />


*Prompt: "A cat sitting on a beach at sunset"*

## Architecture
