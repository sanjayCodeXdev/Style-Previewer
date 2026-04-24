import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

# 1. Load the pre-trained Image-to-Image model
model_id = "timbrooks/instruct-pix2pix"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

inference_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=dtype, safety_checker=None
).to(device)

# 2. Load your cropped 512x512 stock image
input_image_path = "image.png" 
image = load_image(input_image_path)

# 3. Test different beard styles using text conditions
prompts = [
    "add a thick lumberjack beard",
    "add a neat goatee",
    "add a handlebar mustache"
]

# 4. Generate the outputs
for i, prompt in enumerate(prompts):
    output_image = inference_pipe(
        prompt, 
        image=image, 
        num_inference_steps=20, 
        image_guidance_scale=1.5 
    ).images[0]
    
    output_image.save(f"test_style_{i}.png")
    print(f"Saved: '{prompt}' as test_style_{i}.png")