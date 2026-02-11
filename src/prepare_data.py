import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset

def train_hr_transform():
    return transforms.Compose([transforms.ToTensor()])

def train_lr_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

class TrainDataset(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.image_filenames = dataset_dir
        self.hr_transform = train_hr_transform()
        self.lr_transform = train_lr_transform()
    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]).convert('RGB'))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image
    def __len__(self):
        return len(self.image_filenames)

class ValDataset(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.image_filenames = dataset_dir
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')
        lr_image = transforms.Resize((256,256), interpolation=Image.BICUBIC)(hr_image)
        return to_tensor(lr_image), to_tensor(hr_image)
    def __len__(self):
        return len(self.image_filenames)
