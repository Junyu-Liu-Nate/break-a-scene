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
# sys.path.append('third_party/diffusers/src') ### Set specific local diffuser codebase
sys.path.insert(0, '/oscar/data/dritchi1/ljunyu/code/concept/break-a-scene/third_party/diffusers-0.12.1-patch/src')

import argparse
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint
from torch.utils.data import Dataset
import numpy as np

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import ptp_utils
from ptp_utils import AttentionStore
from diffusers.models.cross_attention import CrossAttention

from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor

from utils.dataset import DreamBoothMultiDataset
from utils.arg_parser import parse_args

check_min_version("0.12.0")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    masks = [example["instance_masks"] for example in examples]
    token_ids = [example["token_ids"] for example in examples]

    if with_prior_preservation:
        input_ids = [example["class_prompt_ids"] for example in examples] + input_ids
        pixel_values = [example["class_images"] for example in examples] + pixel_values

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    masks = torch.stack(masks)
    token_ids = torch.stack(token_ids)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "instance_masks": masks,
        "token_ids": token_ids,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


class SpatialDreambooth:
    def __init__(self):
        self.args = parse_args()
        self.main()

    def main(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            logging_dir=logging_dir,
        )

        if (
            self.args.train_text_encoder
            and self.args.gradient_accumulation_steps > 1
            and self.accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # Generate class images if prior preservation is enabled.
        if self.args.with_prior_preservation:
            class_images_dir = Path(self.args.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.args.num_class_images:
                torch_dtype = (
                    torch.float16
                    if self.accelerator.device.type == "cuda"
                    else torch.float32
                )
                if self.args.prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif self.args.prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif self.args.prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                pipeline = DiffusionPipeline.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=self.args.revision,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(self.args.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(
                    sample_dataset, batch_size=self.args.sample_batch_size
                )

                sample_dataloader = self.accelerator.prepare(sample_dataloader)
                pipeline.to(self.accelerator.device)

                for example in tqdm(
                    sample_dataloader,
                    desc="Generating class images",
                    disable=not self.accelerator.is_local_main_process,
                ):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = (
                            class_images_dir
                            / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Handle the repository creation
        if self.accelerator.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            self.args.pretrained_model_name_or_path, self.args.revision
        )

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.args.revision,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
        )

        ########## Add LoRA config ##########
        # We only train the additional adapter LoRA layers
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=4
            )

        self.unet.set_attn_processor(lora_attn_procs)
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)

        self.accelerator.register_for_checkpointing(self.lora_layers)
        #####################################

        # Load the tokenizer
        if self.args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.tokenizer_name, revision=self.args.revision, use_fast=False
            )
        elif self.args.pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.args.revision,
                use_fast=False,
            )

        # Add assets tokens to tokenizer
        total_assets_indices = []
        for assets_indices in self.args.assets_indices_lists:
            for assets_idx in assets_indices:
                total_assets_indices.append(assets_idx)
        self.total_num_of_assets = len(total_assets_indices)
        print(f"Total number of assets: {self.total_num_of_assets}")

        self.all_placeholder_tokens = [
            self.args.placeholder_token.replace(">", f"{idx}>")
            for idx in total_assets_indices
        ]
        self.placeholder_tokens = []
        for idx, assets_indices in enumerate(self.args.assets_indices_lists):
            self.placeholder_tokens.append([])
            for assets_idx in assets_indices:
                self.placeholder_tokens[idx].append(self.args.placeholder_token.replace(">", f"{assets_idx}>"))
        print(f"All placeholder tokens: {self.all_placeholder_tokens}")
        print(f"Placeholder tokens: {self.placeholder_tokens}")
        num_added_tokens = self.tokenizer.add_tokens(self.all_placeholder_tokens)
        # assert num_added_tokens == self.args.num_of_assets
        # assert num_added_tokens == total_num_of_assets
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.all_placeholder_tokens
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.args.instance_prompt = "a photo of " + " and ".join(
            self.all_placeholder_tokens
        )
        print(f"Instance prompt: {self.args.instance_prompt}")
        # exit()

        all_initializer_tokens = []
        for initializer_tokens in self.args.initializer_tokens_list:
            for initializer_token in initializer_tokens:
                all_initializer_tokens.append(initializer_token)
        if len(all_initializer_tokens) > 0:
            # Use initializer tokens
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            for tkn_idx, initializer_token in enumerate(all_initializer_tokens):
                curr_token_ids = self.tokenizer.encode(
                    initializer_token, add_special_tokens=False
                )
                # assert (len(curr_token_ids)) == 1
                token_embeds[self.placeholder_token_ids[tkn_idx]] = token_embeds[
                    curr_token_ids[0]
                ]
        else:
            # Initialize new tokens randomly
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            token_embeds[-self.total_num_of_assets :] = token_embeds[
                -3 * self.total_num_of_assets : -2 * self.total_num_of_assets
            ]

        # Set validation scheduler for logging
        self.validation_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.validation_scheduler.set_timesteps(50)

        # We start by only optimizing the embeddings
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # We start by only optimizing the embeddings
        params_to_optimize = self.text_encoder.get_input_embeddings().parameters()
        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.initial_learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothMultiDataset(
            instance_data_root=self.args.instance_data_dir,
            placeholder_tokens=self.placeholder_tokens,
            class_data_root=self.args.class_data_dir
            if self.args.with_prior_preservation
            else None,
            class_prompt=self.args.class_prompt,
            tokenizer=self.tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop
        )
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(
                examples, self.args.with_prior_preservation
            ),
            num_workers=self.args.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps
            * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps
            * self.args.gradient_accumulation_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        (
            self.unet,
            self.text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet, self.text_encoder, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        if (
            self.args.train_text_encoder
            and self.accelerator.unwrap_model(self.text_encoder).dtype != torch.float32
        ):
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        ### TODO: Check what does this do?
        # if len(self.args.initializer_tokens) > 0:
        #     # Only for logging
        #     self.args.initializer_tokens = ", ".join(self.args.initializer_tokens)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth", config=vars(self.args))

        # Train
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self.args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    num_update_steps_per_epoch * self.args.gradient_accumulation_steps
                )

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        # keep original embeddings as reference
        orig_embeds_params = (
            self.accelerator.unwrap_model(self.text_encoder)
            .get_input_embeddings()
            .weight.data.clone()
        )

        # Create attention controller
        self.controller = AttentionStore()
        self.register_attention_control(self.controller)

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                print(f"step {step}, batch content sizes:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key} size: {value.size()}")
                    else:
                        print(f"  {key} type: {type(value)}")

                if self.args.phase1_train_steps == global_step:
                    # self.unet.requires_grad_(True)
                    self.unet.requires_grad_(False)
                    if self.args.train_text_encoder:
                        self.text_encoder.requires_grad_(True)

                    # unet_params = self.unet.parameters()

                    # params_to_optimize = (
                    #     itertools.chain(unet_params, self.text_encoder.parameters())
                    #     if self.args.train_text_encoder
                    #     else itertools.chain(
                    #         unet_params,
                    #         self.text_encoder.get_input_embeddings().parameters(),
                    #     )
                    # )
                    # del optimizer

                    ########## Set optimizer to update only LoRA from UNet ##########
                    params_to_optimize = (
                        itertools.chain(self.lora_layers.parameters(), self.text_encoder.parameters())
                        if self.args.train_text_encoder
                        else itertools.chain(
                            self.lora_layers.parameters(),
                            self.text_encoder.get_input_embeddings().parameters(),
                        )
                    )
                    del optimizer
                    #################################################

                    optimizer = optimizer_class(
                        params_to_optimize,
                        lr=self.args.learning_rate,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        weight_decay=self.args.adam_weight_decay,
                        eps=self.args.adam_epsilon,
                    )
                    del lr_scheduler
                    lr_scheduler = get_scheduler(
                        self.args.lr_scheduler,
                        optimizer=optimizer,
                        num_warmup_steps=self.args.lr_warmup_steps
                        * self.args.gradient_accumulation_steps,
                        num_training_steps=self.args.max_train_steps
                        * self.args.gradient_accumulation_steps,
                        num_cycles=self.args.lr_num_cycles,
                        power=self.args.lr_power,
                    )
                    # optimizer, lr_scheduler = self.accelerator.prepare(
                    #     optimizer, lr_scheduler
                    # )
                    self.lora_layers, optimizer, lr_scheduler = self.accelerator.prepare(
                        self.lora_layers, optimizer, lr_scheduler
                    )

                logs = {}

                # Skip steps until we reach the resumed step
                if (
                    self.args.resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    latents = self.vae.encode(
                        batch["pixel_values"].to(dtype=self.weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    model_pred = self.unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(
                            latents, noise, timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                        )

                    # print(f'with_prior_preservation: {self.args.with_prior_preservation}')
                    if self.args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred_prior, model_pred = torch.chunk(model_pred, 2, dim=0)
                        target_prior, target = torch.chunk(target, 2, dim=0)

                        if self.args.apply_masked_loss:
                            max_masks = torch.max(
                                batch["instance_masks"], axis=1
                            ).values
                            downsampled_mask = F.interpolate(
                                input=max_masks, size=(64, 64)
                            )
                            model_pred = model_pred * downsampled_mask
                            target = target * downsampled_mask

                        # Compute instance loss
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )

                        # Compute prior loss
                        prior_loss = F.mse_loss(
                            model_pred_prior.float(),
                            target_prior.float(),
                            reduction="mean",
                        )

                        # Add the prior loss to the instance loss.
                        loss = loss + self.args.prior_loss_weight * prior_loss
                    else:
                        if self.args.apply_masked_loss:
                            max_masks = torch.max(
                                batch["instance_masks"], axis=1
                            ).values
                            downsampled_mask = F.interpolate(
                                input=max_masks, size=(64, 64)
                            )
                            model_pred = model_pred * downsampled_mask
                            target = target * downsampled_mask
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )

                    # Attention loss
                    if self.args.lambda_attention != 0:
                        attn_loss = 0
                        for batch_idx in range(self.args.train_batch_size):
                            GT_masks = F.interpolate(
                                input=batch["instance_masks"][batch_idx], size=(16, 16)
                            )
                            agg_attn = self.aggregate_attention(
                                res=16,
                                from_where=("up", "down"),
                                is_cross=True,
                                select=batch_idx,
                            )
                            curr_cond_batch_idx = self.args.train_batch_size + batch_idx
                            
                            # print(f'curr_cond_batch_idx: {curr_cond_batch_idx}')
                            # print(f'len(GT_masks): {len(GT_masks)}')
                            # print(f'batch["input_ids"]: {batch["input_ids"]}')

                            for mask_id in range(len(GT_masks)):
                                curr_placeholder_token_id = self.placeholder_token_ids[
                                    batch["token_ids"][batch_idx][mask_id]
                                ]

                                # print(f'    curr_placeholder_token_id: {curr_placeholder_token_id}')
                                asset_idx = (
                                    (
                                        batch["input_ids"][curr_cond_batch_idx]
                                        == curr_placeholder_token_id
                                    )
                                    .nonzero()
                                    .item()
                                )
                                # print(f'    asset_idx: {asset_idx}')

                                asset_attn_mask = agg_attn[..., asset_idx]
                                asset_attn_mask = (
                                    asset_attn_mask / asset_attn_mask.max()
                                )
                                attn_loss += F.mse_loss(
                                    GT_masks[mask_id, 0].float(),
                                    asset_attn_mask.float(),
                                    reduction="mean",
                                )

                        attn_loss = self.args.lambda_attention * (
                            attn_loss / self.args.train_batch_size
                        )
                        logs["attn_loss"] = attn_loss.detach().item()
                        loss += attn_loss

                    self.accelerator.backward(loss)

                    # No need to keep the attention store
                    self.controller.attention_store = {}
                    self.controller.cur_step = 0

                    if self.accelerator.sync_gradients:
                        # params_to_clip = (
                        #     itertools.chain(
                        #         self.unet.parameters(), self.text_encoder.parameters()
                        #     )
                        #     if self.args.train_text_encoder
                        #     else self.unet.parameters()
                        # )
                        # self.accelerator.clip_grad_norm_(
                        #     params_to_clip, self.args.max_grad_norm
                        # )

                        params_to_clip = (
                            itertools.chain(
                                self.lora_layers.parameters(), self.text_encoder.parameters()
                            )
                            if self.args.train_text_encoder
                            else self.lora_layers.parameters()
                        )
                        self.accelerator.clip_grad_norm_(
                            params_to_clip, self.args.max_grad_norm
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)

                    if global_step < self.args.phase1_train_steps:
                        #TODO: May need to enforce only update selected assets
                        # Let's make sure we don't update any embedding weights besides the newly added token
                        with torch.no_grad():
                            self.accelerator.unwrap_model(
                                self.text_encoder
                            ).get_input_embeddings().weight[
                                : -self.total_num_of_assets
                            ] = orig_embeds_params[
                                : -self.total_num_of_assets
                            ]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(
                                self.args.output_dir, f"checkpoint-{global_step}"
                            )
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                    if (
                        self.args.log_checkpoints
                        and global_step % self.args.img_log_steps == 0
                        and global_step > self.args.phase1_train_steps
                    ):
                        ckpts_path = os.path.join(
                            self.args.output_dir, "checkpoints", f"{global_step:05}"
                        )
                        os.makedirs(ckpts_path, exist_ok=True)
                        self.save_pipeline(ckpts_path)

                        img_logs_path = os.path.join(self.args.output_dir, "img_logs")
                        os.makedirs(img_logs_path, exist_ok=True)

                        if self.args.lambda_attention != 0:
                            self.controller.cur_step = 1
                            last_sentence = batch["input_ids"][curr_cond_batch_idx]
                            last_sentence = last_sentence[
                                (last_sentence != 0)
                                & (last_sentence != 49406)
                                & (last_sentence != 49407)
                            ]
                            last_sentence = self.tokenizer.decode(last_sentence)
                            self.save_cross_attention_vis(
                                last_sentence,
                                attention_maps=agg_attn.detach().cpu(),
                                path=os.path.join(
                                    img_logs_path, f"{global_step:05}_step_attn.jpg"
                                ),
                            )
                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                        self.perform_full_inference(
                            path=os.path.join(
                                img_logs_path, f"{global_step:05}_full_pred.jpg"
                            )
                        )
                        full_agg_attn = self.aggregate_attention(
                            res=16, from_where=("up", "down"), is_cross=True, select=0
                        )
                        self.save_cross_attention_vis(
                            self.args.instance_prompt,
                            attention_maps=full_agg_attn.detach().cpu(),
                            path=os.path.join(
                                img_logs_path, f"{global_step:05}_full_attn.jpg"
                            ),
                        )
                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                logs["loss"] = loss.detach().item()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break

        self.save_pipeline(self.args.output_dir)

        self.accelerator.end_training()

    def save_pipeline(self, path):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                revision=self.args.revision,
            )
            pipeline.save_pretrained(path)

    # def save_pipeline(self, path):
    #     self.accelerator.wait_for_everyone()
    #     if self.accelerator.is_main_process:
    #         lora_weights = {name: param for name, param in self.pipeline.unet.named_parameters() if 'lora' in name}
    #         lora_weights_path = os.path.join(path, 'lora_weights.pth')
    #         torch.save(lora_weights, lora_weights_path)
    #         logger.info(f"Saved LoRA parameters to {lora_weights_path}")

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count

    def get_average_attention(self):
        average_attention = {
            key: [
                item / self.controller.cur_step
                for item in self.controller.attention_store[key]
            ]
            for key in self.controller.attention_store
        }
        return average_attention

    def aggregate_attention(
        self, res: int, from_where: List[str], is_cross: bool, select: int
    ):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res**2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        self.args.train_batch_size, -1, res, res, item.shape[-1]
                    )[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    @torch.no_grad()
    def perform_full_inference(self, path, guidance_scale=7.5):
        self.unet.eval()
        self.text_encoder.eval()

        latents = torch.randn((1, 4, 64, 64), device=self.accelerator.device)
        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.accelerator.device)
        input_ids = self.tokenizer(
            [self.args.instance_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)
        cond_embeddings = self.text_encoder(input_ids)[0]
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )
            noise_pred = pred.sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.validation_scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents

        images = self.vae.decode(latents.to(self.weight_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        self.unet.train()
        if self.args.train_text_encoder:
            self.text_encoder.train()

        Image.fromarray(images[0]).save(path)

    @torch.no_grad()
    def save_cross_attention_vis(self, prompt, attention_maps, path):
        tokens = self.tokenizer.encode(prompt)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(
                image, self.tokenizer.decode(int(tokens[i]))
            )
            images.append(image)
        vis = ptp_utils.view_images(np.stack(images, axis=0))
        vis.save(path)

class P2PCrossAttnProcessor(torch.nn.Module):
    def __init__(self, controller, place_in_unet):
        # super().__init__()
        super(P2PCrossAttnProcessor, self).__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


if __name__ == "__main__":
    SpatialDreambooth()
