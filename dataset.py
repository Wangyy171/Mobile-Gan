import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class PairDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, size=256):
        self.lr = sorted([os.path.join(lr_dir,f) for f in os.listdir(lr_dir)])
        self.hr = sorted([os.path.join(hr_dir,f) for f in os.listdir(hr_dir)])
        self.tf = T.Compose([
            T.Resize((size,size)),
            T.ToTensor(),
            T.Normalize([0.5]*3,[0.5]*3)
        ])
    def __len__(self): return len(self.lr)
    def __getitem__(self, i):
        x = self.tf(Image.open(self.lr[i]).convert('RGB'))
        y = self.tf(Image.open(self.hr[i]).convert('RGB'))
        return x, y
