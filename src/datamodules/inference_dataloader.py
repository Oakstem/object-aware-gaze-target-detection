import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class PNGDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.image_files = [f for f in os.listdir(directory) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return image