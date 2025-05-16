# text_to_image.py
import gc

from diffusers import StableDiffusionPipeline
import torch
import os

# Load the model
def load_pipeline(model_id="runwayml/stable-diffusion-v1-5", device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Loading model: {model_id} on {device}")
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    pipeline = pipeline.to(device)
    return pipeline

# Generate image
def generate_image(prompt, pipeline, num_inference_steps=25, guidance_scale=7, output_dir="outputs", filename="generated_image.png"):
    print(f"Generating image for prompt: '{prompt}'")
    image = pipeline(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,height=384,width=384).images[0]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    print(f"Image saved to {output_path}")
    gc.collect()
    return output_path

# Main logic
if __name__ == "__main__":
    prompt = input("Enter your text prompt: ")
    pipeline = load_pipeline()
    generate_image(prompt, pipeline)





