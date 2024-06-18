from diffusers import ControlNetModel,StableDiffusionControlNetImg2ImgPipeline,UniPCMultistepScheduler
from diffusers.utils import load_image, make_image_grid
import torch
from PIL import Image




controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", 
    torch_dtype=torch.float16,
    use_safetensors=True
    )

pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "lykon/dreamshaper-8", 
    controlnet=controlnet,
    torch_dtype=torch.float16, 
    use_safetensors=True
    )

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
pipeline.to(device)

# speed up diffusion process with faster scheduler and memory optimization
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
#pipeline.enable_model_cpu_offload()

# Insert IP-Adapter into the pipeline
pipeline.load_ip_adapter(
    "h94/IP-Adapter", 
    subfolder="models", 
    weight_name="ip-adapter-plus-face_sd15.bin")


# Define the prompt
user_prompt = "masterpiece, portrait of a person, anime style, high quality, RAW photo, 8k uhd"
# Load user image
user_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png")
generator = torch.Generator(device=device).manual_seed(33)


result_image = pipeline(
prompt=user_prompt,
num_inference_steps=50,
generator=generator,
image=user_image,
).images[0]

output = make_image_grid([user_image, result_image], rows=1, cols=2)
output.show()
