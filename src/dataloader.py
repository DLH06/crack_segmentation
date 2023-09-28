import glob
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CrackDataset(Dataset):
    def __init__(self, root_dir="data/train", image_size=448):
        self.root_dir = root_dir

        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.image_paths = glob.glob(os.path.join(self.image_dir, "*"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask_path = glob.glob(
            os.path.join(
                self.mask_dir,
                os.path.basename(img_path).replace(os.path.splitext(img_path)[1], "*"),
            )
        )[0]
        mask = Image.open(mask_path).convert("L")

        if self.image_transform and self.mask_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return image, mask


if __name__ == "__main__":
    test = CrackDataset("data/train")
    print("Number of train samples:", len(test))
    print(test[0][0].shape)
    print(test[0][1].shape)
    test = CrackDataset("data/test")
    print("Number of test samples:", len(test))
