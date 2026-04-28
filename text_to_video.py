# ============================================
# AI Text-to-Video Generator
# Author: Gunika Arora
# Model: Zeroscope V2 XL Diffusion Model
# ============================================

import torch
import tempfile
import gradio as gr
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# ============================================
# Step 1: Load Model
# ============================================
print("Loading Zeroscope V2 XL model...")
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()
print("✅ Model loaded!")

# ============================================
# Step 2: Video Generation Function
# ============================================
def generate_video(prompt, num_frames, num_steps, guidance_scale):
    print(f"Generating: {prompt}")
    
    enhanced_prompt = f"{prompt}, high quality, cinematic, sharp focus, detailed"
    negative_prompt = "low quality, blurry, pixelated, distorted, watermark, ugly"
    
    torch.cuda.empty_cache()
    
    with torch.inference_mode():
        frames = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_frames=int(num_frames),
            num_inference_steps=int(num_steps),
            guidance_scale=guidance_scale,
            height=320,
            width=576,
        ).frames[0]
    
    output_path = "/kaggle/working/output.mp4"
    export_to_video(frames, output_path, fps=8)
    print("✅ Video saved!")
    return output_path

# ============================================
# Step 3: Gradio UI
# ============================================
with gr.Blocks(title="AI Text-to-Video Generator") as demo:
    
    gr.Markdown("""
    # 🎬 AI Text-to-Video Generator
    ### Zeroscope V2 XL Diffusion Model | Built by Gunika Arora
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="A astronaut riding a horse on the moon...",
                lines=3
            )
            with gr.Accordion("Advanced Settings", open=False):
                num_frames = gr.Slider(
                    minimum=8, maximum=24, value=8, step=4,
                    label="Frames"
                )
                num_steps = gr.Slider(
                    minimum=10, maximum=50, value=20, step=5,
                    label="Inference Steps"
                )
                guidance_scale = gr.Slider(
                    minimum=5.0, maximum=20.0, value=9.0, step=0.5,
                    label="Guidance Scale"
                )
            
            generate_btn = gr.Button(
                "🎬 Generate Video", 
                variant="primary", 
                size="lg"
            )
            
            gr.Examples(
                examples=[
                    ["A golden retriever playing in autumn leaves"],
                    ["A rocket launching into space with fire and smoke"],
                    ["Ocean waves crashing on beach at sunset"],
                    ["A butterfly landing on a red flower"],
                    ["Snow falling in quiet forest at night"],
                ],
                inputs=prompt_input
            )
        
        with gr.Column(scale=1):
            video_output = gr.Video(label="Generated Video")
            gr.Markdown("""
            **Tips for better videos:**
            - Add "cinematic", "slow motion", "4k" in prompt
            - 20 steps = fast, 40 steps = better quality
            - 8 frames = ~1 sec video
            """)
    
    generate_btn.click(
        fn=generate_video,
        inputs=[prompt_input, num_frames, num_steps, guidance_scale],
        outputs=video_output
    )

demo.launch(share=True, debug
