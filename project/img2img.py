from diffusers import ControlNetModel,StableDiffusionControlNetPipeline,UniPCMultistepScheduler
from diffusers.utils import load_image
import torch



controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", 
    torch_dtype=torch.float16)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "lykon/dreamshaper-8", 
    controlnet=controlnet,
    torch_dtype=torch.float16, 
    safety_checker=None)

#pipeline.to("cuda")

pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

pipeline.load_ip_adapter(
    "h94/IP-Adapter", 
    subfolder="models", 
    weight_name="ip-adapter-plus-face_sd15.bin")


# Define the prompt
user_prompt = "masterpiece, portrait of a person, anime style, high quality, RAW photo, 8k uhd"

# Load user image
user_image = load_image("gallery/upwork_copy.png")

generator = torch.Generator(device="cpu").manual_seed(33)

result = pipeline(
    prompt=user_prompt,
    image=user_image,
    num_inference_steps=50,
    generator=generator,
    ).images[0]

result.show()
result.save("gallery/1.png")