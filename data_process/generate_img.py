import sys
### Set specific local diffuser codebase
sys.path.insert(0, '/oscar/data/dritchi1/ljunyu/code/concept/break-a-scene/third_party/diffusers-0.12.1-patch/src')

import argparse

from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch

class ImgGenerator:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument(
            "--pretrained_model_name_or_path",
            type=str,
            default="stabilityai/stable-diffusion-2-1-base",
            help="Path to pretrained model or model identifier from huggingface.co/models.",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            required=False,
            help=(
                "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
                " float32 precision."
            ),
        )
        parser.add_argument(
            "--prompt", type=str, default="a photo of a chair with red armrest, blue backrest, green seat, and yellow legs."
        )
        parser.add_argument("--output_path", type=str, default="./result.jpg")
        parser.add_argument("--device", type=str, default="cuda")
        self.args = parser.parse_args()

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path, 
            revision=self.args.revision
        )
        self.pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline.to(self.args.device)

    @torch.no_grad()
    def infer_and_save(self, prompts):
        for i in range(5):
            images = self.pipeline(prompts).images
            save_path = self.args.output_path.replace(".jpg", f"_{i}.jpg")
            images[0].save(save_path)

if __name__ == "__main__":
    img_generator = ImgGenerator()
    img_generator.infer_and_save(
        prompts=[img_generator.args.prompt]
    )