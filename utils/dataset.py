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
        
        tokens_ids_to_use = random.sample(range(len(self.placeholder_tokens[index % len(self.placeholder_tokens)])), k=random.randrange(1, len(self.placeholder_tokens[index % len(self.placeholder_tokens)]) + 1))
        tokens_to_use = [self.placeholder_tokens[index % len(self.placeholder_tokens)][tkn_i] for tkn_i in tokens_ids_to_use]
        prompt = "a photo of with " + " and ".join(tokens_to_use)

        example = {
            "instance_images": instance_image,
            "instance_masks": instance_masks[tokens_ids_to_use],
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

        # Debug print for each item's size in the example
        print(f"Example at index {index}:")
        for key, value in example.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key} size: {value.size()}")
            else:
                print(f"  {key} type: {type(value)}")
        return example