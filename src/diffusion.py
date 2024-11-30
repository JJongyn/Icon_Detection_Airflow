from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import torch
import os
import random

class AppIconGenerator:
    def __init__(self, controlnet_model: str, diffusion_model: str, device: str = "cpu"):
        # ControlNet 모델 로드
        self.controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float32)

        # Stable Diffusion 파이프라인 설정
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            diffusion_model,
            controlnet=self.controlnet,
            torch_dtype=torch.float32
        ).to(device)

    def _generate_app_icon_prompt(self, icon_name: str = "gmail"):
        """
        Generate prompts for app icon generation.
        Args:
            icon_name (str): Name of the icon.
        Returns:
            Tuple[str, str]: Positive and negative prompts.
        """
        styles = [
            "material design", "flat minimal", "modern flat", "android style",
            "google design language", "simple vector", "clean minimal"
        ]
        colors = [
            "google blue (#4285F4)", "google red (#DB4437)",
            "google yellow (#F4B400)", "google green (#0F9D58)",
            "white", "light gray", "material blue", "material red"
        ]
        layouts = [
            "centered icon", "material design grid", "geometric layout",
            "adaptive icon shape", "circular container", "rounded square"
        ]
        details = [
            "subtle gradient", "flat shadow", "minimal depth",
            "clean outline", "sharp edges", "smooth corners"
        ]
        backgrounds = [
            "pure white", "slightly off-white", "material light gray",
            "subtle paper texture", "flat color"
        ]
        quality_terms = [
            "high quality", "4k resolution", "vector graphics",
            "clean rendering", "sharp details", "professional design"
        ]

        prompt = f"A {random.choice(styles)} {icon_name} app icon for Chrome OS, "
        prompt += f"using {random.choice(colors)} as primary color. "
        prompt += f"Icon features {random.choice(layouts)} with {random.choice(details)}. "
        prompt += f"Set against {random.choice(backgrounds)} background. "
        prompt += f"{random.choice(quality_terms)}. "
        prompt += "Maintain Google Material Design guidelines. "
        prompt += "Professional UI/UX design, digital art."

        negative_prompt = (
            "realistic, 3d effect, heavy texture, noise, grain, "
            "photography, blur, camera photo, dramatic shadows, "
            "skeuomorphic design, beveled edges, complex patterns"
        )

        return prompt, negative_prompt

    def _generate_icon(self, condition_image_path: str, output_path: str, icon_name: str, idx: int):
        """
        Generate app icon images using ControlNet and Stable Diffusion.
        Args:
            condition_image_path (str): Path to the condition image.
            output_path (str): Directory to save the generated image.
            icon_name (str): Name of the app icon.
            idx (int): Index for file naming.
        """
        # Load condition image
        condition_image = Image.open(condition_image_path).convert("RGB")
        
        # Generate prompts
        prompt, negative_prompt = self._generate_app_icon_prompt(icon_name)
        
        # Generate augmented image
        result = self.pipe(prompt, num_inference_steps=10, image=condition_image, negative_prompt=negative_prompt).images[0]
        
        # Save the result
        os.makedirs(output_path, exist_ok=True)
        file_name = f"augmented_{icon_name}_{idx}.png"
        file_path = os.path.join(output_path, file_name)
        result.save(file_path)
        print(f"Generated icon saved at: {file_path}")

    def generate_icons_for_all_images(self, input_path: str, output_path: str):
        """
        Generate augmented icons for all images in the input directory.
        Args:
            input_path (str): Directory containing the condition images.
            output_path (str): Directory to save the generated images.
        """
        # Get all image files in the input directory
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # For each image file in the input directory, generate augmented icons
        for image_file in image_files:
            icon_name, _ = os.path.splitext(image_file)  # Use the file name without extension as the icon name
            condition_image_path = os.path.join(input_path, image_file)

            for idx in range(1):  # Generate 5 augmented images for each icon
                self._generate_icon(condition_image_path, output_path, icon_name, idx)


