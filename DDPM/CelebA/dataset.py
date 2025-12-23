import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CelebAHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def get_transforms():
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
