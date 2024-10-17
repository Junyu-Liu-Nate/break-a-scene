"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
### Set specific local diffuser codebase
sys.path.insert(0, '/oscar/data/dritchi1/ljunyu/code/concept/break-a-scene/third_party/diffusers-0.12.1-patch/src')

import argparse

from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch


class BreakASceneInference:
    def __init__(self):
        self._parse_args()
        # self._load_pipeline()
        self._load_pipeline_lora()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True)
        
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
            "--prompt", type=str, default="a photo of <asset0> at the beach"
        )
        parser.add_argument("--output_path", type=str, default="outputs/result.jpg")
        parser.add_argument("--device", type=str, default="cuda")
        self.args = parser.parse_args()

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline.to(self.args.device)

    # def _load_pipeline_lora(self):
    #     # Load previous pipeline
    #     self.pipeline = DiffusionPipeline.from_pretrained(
    #         self.args.pretrained_model_name_or_path, revision=self.args.revision
    #     )
    #     self.pipeline.scheduler = DDIMScheduler(
    #         beta_start=0.00085,
    #         beta_end=0.012,
    #         beta_schedule="scaled_linear",
    #         clip_sample=False,
    #         set_alpha_to_one=False,
    #     )
    #     self.pipeline.to(self.args.device)

    #     # load attention processors
    #     self.pipeline.unet.load_attn_procs(self.args.model_path)

    def _load_pipeline_lora(self):
        # Load base model components
        unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
            torch_dtype=torch.float16,
        )

        vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.args.revision,
            torch_dtype=torch.float16,
        )

        # Load the text_encoder from the model path (LoRA weights directory)
        text_encoder = CLIPTextModel.from_pretrained(
            self.args.model_path,
            revision=self.args.revision,
            torch_dtype=torch.float16,
        )

        # Load the tokenizer from the model path (LoRA weights directory)
        tokenizer = CLIPTokenizer.from_pretrained(
            self.args.model_path,
            revision=self.args.revision,
        )

        # Resize token embeddings in the text_encoder
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Load the LoRA weights into the unet
        unet.load_attn_procs(self.args.model_path)

        # Create the pipeline using StableDiffusionPipeline
        self.pipeline = StableDiffusionPipeline(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            ),
            safety_checker=None,  # Include if you don't have a safety checker
            feature_extractor=None,  # Include if you don't have a feature extractor
        )

        self.pipeline.to(self.args.device)

    @torch.no_grad()
    def infer_and_save(self, prompts):
        for i in range(5):
            images = self.pipeline(prompts).images
            save_path = self.args.output_path.replace(".jpg", f"_{i}.jpg")
            images[0].save(save_path)


if __name__ == "__main__":
    break_a_scene_inference = BreakASceneInference()
    break_a_scene_inference.infer_and_save(
        prompts=[break_a_scene_inference.args.prompt]
    )
