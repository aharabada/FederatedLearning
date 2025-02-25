from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import torch
import pandas as pd

class EyeDiameterDataset(Dataset):
    def __init__(self, annotations_file: str):
        self.img_labels = pd.read_csv(annotations_file)
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = read_image(img_path).float()
        
        # remove alpha channel
        if image.shape[0] == 4:
            image = image[:3, :, :]
        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(min(image.shape[1:])),
            transforms.Resize((128, 128)),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        image = transform(image)
        label = torch.tensor(float(self.img_labels.iloc[idx, 1]), dtype=torch.float32)
        return image, label
    
    
class EyeBinaryMaskDataset(Dataset):
    def __init__(self, annotations_file: str, preprocess: bool = True, length: int = 0):
        self.data = []
        self.labels = []
        self.length = length
        if preprocess:
            self.img_labels = pd.read_csv(annotations_file)
            self.length = len(self.img_labels)
            self.__preprocess_data()
            self.__preprocess_labels()
        
    def __preprocess_data(self):
        for idx in range(len(self.img_labels)):
            img_path = self.img_labels.iloc[idx, 0]
            image = read_image(img_path).float()
            
            if image.shape[0] == 4:
                image = image[:3, :, :]
            
            img_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.CenterCrop(min(image.shape[1:])),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            image = img_transform(image)
            self.data.append(image)
            
    def __preprocess_labels(self):
        for idx in range(len(self.img_labels)):
            label_path = self.img_labels.iloc[idx, 1]
            label = read_image(label_path).float()
            
            if label.shape[0] == 4:
                label = label[:3, :, :]
            
            mask_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.CenterCrop(min(label.shape[1:])),
                transforms.ConvertImageDtype(torch.float)
            ])
            
            label = mask_transform(label)
            label = (label > 200).float()
            self.labels.append(label)
       
    def __len__(self):
        if self.data is not None and self.labels is not None and len(self.data) > 0 and len(self.labels) > 0 and len(self.data) == len(self.labels):
            return len(self.data)
        else:
            return self.length
   
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_data_loader(dataset_path: str, dataset_type: object = EyeDiameterDataset, batch_size: int = 64, preprocess: bool = True, length: int = 0) -> torch.utils.data.DataLoader:
    dataset = dataset_type(dataset_path, preprocess, length)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return dataloader