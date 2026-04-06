import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset


class InfiniteImageNetLoader:
    def __init__(self, cache_dir, batch_size=32, image_size=256, num_workers=6):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

        print(f"[*] Loading ImageNet from local SSD: {self.cache_dir}")
        self.dataset = load_dataset(
            "imagenet-1k",
            split="train",
            cache_dir=self.cache_dir
        )

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.dataset.set_transform(self._transform_batch)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._custom_collate,
            drop_last=True,
            shuffle=True
        )

        self.data_iter = iter(self.dataloader)

    def _transform_batch(self, examples):
        examples["pixel_values"] = [self.transform(img.convert("RGB")) for img in examples["image"]]
        return examples

    def _custom_collate(self, batch):
        images = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return images, labels

    def get_next_batch(self):
        try:
            images, labels = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            images, labels = next(self.data_iter)

        return images, labels

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next_batch()