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
            transforms.Resize((128, 128)),  # von 256x256 auf 128x128 reduziert
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        image = transform(image)
        label = torch.tensor(float(self.img_labels.iloc[idx, 1]), dtype=torch.float32)
        return image, label


def create_data_loader(data_set_path: str, batch_size: int = 8) -> torch.utils.data.DataLoader:
    dataset = EyeDiameterDataset(data_set_path)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return dataloader