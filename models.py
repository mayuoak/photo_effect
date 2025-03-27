import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

class GhibliModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load control model
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        ).to(self.device)

        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, torch_dtype=torch.float16
        ).to(self.device)

        if self.device == "cpu":
            self.pipe.enable_model_cpu_offload()

    def generate_image(self, input_image: Image):
        prompt = "A beautiful Studio Ghibli-style painting, soft lighting, vivid colors, high detail"
        output_image = self.pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]

        return output_image
