from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import torch
import pandas as pd
    
    
class EyeBinaryMaskDataset(Dataset):
    """
    A custom dataset class for loading and preprocessing eye binary mask images and their corresponding labels.
    The dataset is provided by Jan Kaminski working on his masters thesis at LUH at the institute CHI.
    
    Attributes:
        data (list): A list to store preprocessed image data.
        labels (list): A list to store preprocessed label data.
        length (int): The length of the dataset.
    Args:
        annotations_file (str): Path to the CSV file containing image and label paths.
        preprocess (bool): Whether to preprocess the data and labels. Default is True.
        length (int): The length of the dataset. Default is 0.
    Methods:
        __preprocess_data(): Preprocesses the image data by reading, transforming, and storing it.
        __preprocess_labels(): Preprocesses the label data by reading, transforming, and storing it.
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the image and label at the specified index.
    """
    
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
        """
        Preprocesses the image data by reading, transforming, and storing it.
        """
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
        """
        Preprocesses the label data by reading, transforming, and storing it.
        """
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


def create_data_loader(dataset_path: str, dataset_type: object = EyeBinaryMaskDataset, batch_size: int = 8, preprocess: bool = True, length: int = 0) -> torch.utils.data.DataLoader:
    """
    Creates a data loader for the specified dataset.
    
    Args:
        dataset_path (str): Path to the dataset.
        dataset_type (object): The dataset class to use. Default is EyeBinaryMaskDataset.
        batch_size (int): The batch size. Default is 8.
        preprocess (bool): Whether to preprocess the data and labels. Default is True.
        length (int): The length of the dataset. Default is 0. This case is handled separately in the nn-clients.
    Returns:
        torch.utils.data.DataLoader: The data loader for the specified dataset.
    """
    dataset = dataset_type(dataset_path, preprocess, length)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return dataloader