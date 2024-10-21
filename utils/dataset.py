from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class DreamBoothMultiDataset(Dataset):
    """
    Dataset for multiple images
    """

    def __init__(
        self,
        instance_data_root,
        placeholder_tokens,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        flip_p=0.5,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.flip_p = flip_p

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.placeholder_tokens = placeholder_tokens
        self.all_placeholder_tokens = []
        for tokens_list in self.placeholder_tokens:
            for token in tokens_list:
                self.all_placeholder_tokens.append(token)

        self.instances = []
        for folder in sorted(self.instance_data_root.iterdir()):
            if folder.is_dir():
                instance_img_path = folder / "img.jpg"
                instance_image = self.image_transforms(Image.open(instance_img_path))

                instance_masks = []
                i = 0
                while (folder / f"mask{i}.png").exists():
                    instance_mask_path = folder / f"mask{i}.png"
                    curr_mask = Image.open(instance_mask_path)
                    curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
                    instance_masks.append(curr_mask)
                    i += 1

                if instance_masks:
                    instance_masks = torch.cat(instance_masks)

                self.instances.append((instance_image, instance_masks))

        self._length = len(self.instances)

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self._length)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        instance_image, instance_masks = self.instances[index % len(self.instances)]
        # print(f'index: {index}')
        # print(f'instance_image size: {instance_image.size()}')
        # print(f'instance_masks size: {instance_masks.size()}')
        
        # tokens_ids_to_use = random.sample(range(len(self.placeholder_tokens[index % len(self.placeholder_tokens)])), k=random.randrange(1, len(self.placeholder_tokens[index % len(self.placeholder_tokens)]) + 1))
        # print(f'tokens_ids_to_use: {tokens_ids_to_use}')
        # tokens_to_use = [self.placeholder_tokens[index % len(self.placeholder_tokens)][tkn_i] for tkn_i in tokens_ids_to_use]
        # Get the placeholder tokens for the current index
        current_tokens_list = self.placeholder_tokens[index % len(self.placeholder_tokens)]
        # print(f'current_tokens_list: {current_tokens_list}')

        # Determine the number of tokens to use
        num_tokens_to_use = random.randrange(1, len(current_tokens_list) + 1)
        # print(f'num_tokens_to_use: {num_tokens_to_use}')

        # Randomly select the token IDs to use
        tokens_ids_to_use = random.sample(range(len(current_tokens_list)), k=num_tokens_to_use)
        # print(f'tokens_ids_to_use: {tokens_ids_to_use}')

        # Retrieve the actual tokens using the selected indices
        tokens_to_use = [current_tokens_list[tkn_id] for tkn_id in tokens_ids_to_use]
        # print(f'tokens_to_use: {tokens_to_use}')
        
        prompt = "a photo of " + " and ".join(tokens_to_use)

        tokens_ids_to_use_global = []
        ### Here the ids are global ids in all placeholder tokens
        for token_to_use in tokens_to_use:
            tokens_ids_to_use_global.append(self.all_placeholder_tokens.index(token_to_use))
        example = {
            "instance_images": instance_image,
            "instance_masks": instance_masks[tokens_ids_to_use],
            "token_ids": torch.tensor(tokens_ids_to_use_global),
            "instance_prompt_ids": self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        }

        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["instance_masks"] = TF.hflip(example["instance_masks"])

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        # Debug print for each item's size in the example
        # print(f"Example at index {index}:")
        # for key, value in example.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"  {key} size: {value.size()}")
        #     else:
        #         print(f"  {key} type: {type(value)}")
        return example

class DreamBoothSynthDataset(DreamBoothMultiDataset):
    """
    Dataset for synthesizing images by pasting masked areas onto a blank image
    """

    def __init__(
        self,
        instance_data_root,
        placeholder_tokens,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        flip_p=0.5,
    ):
        super().__init__(
            instance_data_root,
            placeholder_tokens,
            tokenizer,
            class_data_root,
            class_prompt,
            size,
            center_crop,
            flip_p,
        )

    def get_bounding_box(self, mask):
        """Calculate the bounding box of the non-zero region in the mask."""
        # Assuming mask is a 3D tensor of shape [1, H, W]
        if mask.ndim == 3:
            mask = mask.squeeze(0)  # Reduce to [H, W] if it's [1, H, W]

        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        if not rows.any() or not cols.any():
            return None

        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax

    def __getitem__(self, index):
        blank_image = torch.zeros((3, self.size, self.size))
        # TF.to_pil_image(blank_image).save('blank_image.png')
        blank_mask = torch.zeros((len(self.all_placeholder_tokens), 1, self.size, self.size))

        # Randomly select the number of tokens to use (at most 4)
        num_tokens_to_use = random.randint(1, min(4, len(self.all_placeholder_tokens)))
        # print(f'num_tokens_to_use: {num_tokens_to_use}')

        # Randomly select the token IDs to use
        tokens_ids_to_use = random.sample(range(len(self.all_placeholder_tokens)), k=num_tokens_to_use)
        # print(f'tokens_ids_to_use: {tokens_ids_to_use}')

        # Retrieve the actual tokens using the selected indices
        # print(f'self.all_placeholder_tokens: {self.all_placeholder_tokens}')
        tokens_to_use = [self.all_placeholder_tokens[tkn_id] for tkn_id in tokens_ids_to_use]
        # print(f'tokens_to_use: {tokens_to_use}')

        # Synthesize the prompt
        prompt = "a photo of " + " and ".join(tokens_to_use)

        # Initialize occupancy mask
        occupancy_mask = torch.zeros((self.size, self.size), dtype=torch.bool)

        # print(f'lens of self.instances: {len(self.instances)}')
        for token_id in tokens_ids_to_use:
            # Determine which instance and mask to use based on the global token ID
            cumulative_masks = 0
            for instance_image, instance_masks in self.instances:
                num_masks = instance_masks.size(0)
                if token_id < cumulative_masks + num_masks:
                    mask_index = token_id - cumulative_masks
                    mask = instance_masks[mask_index]
                    instance_img = instance_image
                    # print(f'mask_index: {mask_index}, token_id: {token_id}, cumulative_masks: {cumulative_masks}')
                    break
                cumulative_masks += num_masks

            # Remove singleton dimensions from mask
            mask = mask.squeeze()  # Now shape should be [512, 512]

            # Convert mask to boolean
            mask = mask.bool()

            # Verify mask dimensions
            if mask.ndim != 2:
                print(f"Unexpected mask dimensions: {mask.shape}")
                continue

            # Get non-zero indices of the mask
            mask_indices = mask.nonzero(as_tuple=False)  # shape [N, 2]
            if mask_indices.numel() == 0:
                print("No active area found in mask. Mask is likely all zeros.")
                continue

            ymin, xmin = mask_indices.min(dim=0)[0]
            ymax, xmax = mask_indices.max(dim=0)[0]

            # Extract the mask_region and image_region
            mask_region = mask[ymin:ymax+1, xmin:xmax+1]  # shape [h, w]
            image_region = instance_img[:, ymin:ymax+1, xmin:xmax+1]  # shape [3, h, w]
            h, w = mask_region.shape

            found = False
            for _ in range(100):
                x0 = random.randint(0, self.size - w)
                y0 = random.randint(0, self.size - h)

                occupancy_slice = occupancy_mask[y0:y0+h, x0:x0+w]

                # Check for overlap
                overlap = torch.logical_and(occupancy_slice, mask_region).any()

                if not overlap:
                    # Shifted mask and image
                    shifted_mask = torch.zeros((self.size, self.size), dtype=torch.bool)
                    shifted_image = torch.zeros_like(instance_img)

                    shifted_mask[y0:y0+h, x0:x0+w] = mask_region
                    shifted_image[:, y0:y0+h, x0:x0+w] = image_region

                    # Update the blank_image and blank_mask
                    mask_region_expanded = shifted_mask.unsqueeze(0)  # shape [1, 512, 512]
                    blank_image = torch.where(
                        mask_region_expanded.expand_as(blank_image),
                        shifted_image,
                        blank_image
                    )

                    # Update the blank_mask for this token
                    blank_mask[token_id, 0] = shifted_mask

                    # Update occupancy_mask
                    occupancy_mask = torch.logical_or(occupancy_mask, shifted_mask)

                    found = True
                    break

            if not found:
                print(f"Failed to place mask for token {token_id} after 100 attempts.")

        # print(f'blank_image size: {blank_image.size()}')
        # print(f'blank_mask size: {blank_mask.size()}')
        
        # # Save the composed image and masks as PNG to visualize
        # composed_image_path = f"composed_image_{index}.png"        
        # # Inverse normalization
        # inv_normalize = transforms.Normalize(
        #     mean=[-0.5 / 0.5],
        #     std=[1 / 0.5]
        # )
        # # Apply the inverse normalization
        # blank_image = inv_normalize(blank_image)
        # # Save the composed image
        # TF.to_pil_image(blank_image).save(composed_image_path)

        # # Save each mask separately
        # for i, mask in enumerate(blank_mask):
        #     mask_to_save = mask.squeeze().clamp(0, 1) * 255
        #     mask_path = f"composed_mask_{index}_mask_{i}.png"
        #     TF.to_pil_image(mask_to_save.byte()).save(mask_path)
        
        # exit()

        example = {
            "instance_images": blank_image,
            "instance_masks": blank_mask[tokens_ids_to_use],
            "token_ids": torch.tensor(tokens_ids_to_use),
            "instance_prompt_ids": self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        }

        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["instance_masks"] = TF.hflip(example["instance_masks"])

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example