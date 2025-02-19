import torch
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

from UNet import PupilSegmentationUNet
from DataSet import create_data_loader, EyeBinaryMaskDataset

class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = torch.nn.BCELoss()
        
    def forward(self, pred, target):
        # Ensure target is float and normalized
        target = target.float()
        if target.max() > 1 or target.min() < 0:
            target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        bce_loss = self.bce(pred, target)
        
        # Dice Loss
        smooth = 1e-5
        intersection = (pred * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 0.5 * bce_loss + 0.5 * dice_loss

def calculate_iou(pred, target):
    target = target.float()
    if target.max() > 1 or target.min() < 0:
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
    
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    return (intersection + 1e-5) / (union + 1e-5)

class Host:
    def __init__(self, model: torch.nn.Module, data_loader: dict[str, torch.utils.data.DataLoader]):
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Host Device: {self.device}")
        self.train_losses = []
        self.valid_losses = []
        self.test_metrics = None
    
    def normalize_target(self, target):
        target = target.float()
        if target.max() > 1 or target.min() < 0:
            target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        return target
        
    def __evaluate_model(self, dataset):
        self.model.eval()
        criterion = DiceBCELoss()
        total_loss = 0
        total_iou = 0
        num_samples = 0
        
        with torch.no_grad():
            for data, target in dataset:
                data = data.float().to(self.device)
                target = self.normalize_target(target).to(self.device)
                output = self.model(data)
                total_loss += criterion(output, target).item()
                total_iou += calculate_iou(output, target).item()
                num_samples += 1
        
        avg_loss = total_loss / num_samples
        avg_iou = total_iou / num_samples
        
        return {
            'loss': avg_loss,
            'iou': avg_iou
        }
            
    def __save_model(self, path: str = "models"):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "host_model_unet.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    def __save_training_data(self, path: str = "training_process_data"):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "train_losses_unet.npy"), np.array(self.train_losses))
        np.save(os.path.join(path, "valid_losses_unet.npy"), np.array(self.valid_losses))
        
        if self.test_metrics is not None:
            np.save(os.path.join(path, "test_metrics_unet.npy"), np.array([
                self.test_metrics['loss'],
                self.test_metrics['iou']
            ]))
        print(f"Training data saved to {path}/")
            
    def inital_training(self, iterations: int):
        self.model.to(self.device)
        
        train_dataset = self.data_loader["train"]
        train2_dataset = self.data_loader["train2"]
        valid_dataset = self.data_loader["valid"]
        test_dataset = self.data_loader["test"]
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        criterion = DiceBCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        for epoch in range(iterations):
            self.model.train()
            train_loss = 0
            num_batches = 0
            
            # Training auf dem ersten Datensatz
            for data, target in tqdm.tqdm(train_dataset, desc=f"Epoch {epoch+1}/{iterations} - Train 1"):
                data = data.float().to(self.device)
                target = self.normalize_target(target).to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
                
            # Training auf dem zweiten Datensatz
            for data, target in tqdm.tqdm(train2_dataset, desc=f"Epoch {epoch+1}/{iterations} - Train 2"):
                data = data.float().to(self.device)
                target = self.normalize_target(target).to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            
            # Validierung
            self.model.eval()
            valid_loss = 0
            valid_batches = 0
            with torch.no_grad():
                for data, target in valid_dataset:
                    data = data.float().to(self.device)
                    target = self.normalize_target(target).to(self.device)
                    output = self.model(data)
                    valid_loss += criterion(output, target).item()
                    valid_batches += 1
            
            epoch_train_loss = train_loss / num_batches
            epoch_valid_loss = valid_loss / valid_batches
            
            self.train_losses.append(epoch_train_loss)
            self.valid_losses.append(epoch_valid_loss)
            
            scheduler.step(epoch_valid_loss)
            
            print(f"Epoch {epoch + 1}/{iterations}")
            print(f"Train Loss: {epoch_train_loss:.4f}")
            print(f"Valid Loss: {epoch_valid_loss:.4f}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
        self.__save_model()
        
        print("\nEvaluating final model performance on test set...")
        self.test_metrics = self.__evaluate_model(test_dataset)
        
        print("\nFinal Test Results:")
        print("=" * 50)
        print(f"Test Loss: {self.test_metrics['loss']:.6f}")
        print(f"Test IoU: {self.test_metrics['iou']:.6f}")
        print("=" * 50)
        
        self.__save_training_data()
        
    def federated_training(self):
        pass

def load_model(model_path: str = "models/host_model.pth"):
    model = PupilSegmentationUNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def visualize_predictions(model, data_loader, num_examples=5):
    device = torch.device("cpu")
    model.to(device)
    
    dataiter = iter(data_loader)
    images, masks = next(dataiter)
    
    model.eval()
    with torch.no_grad():
        images = images.float().to(device)
        predictions = model(images)
        
    images = images.cpu()
    masks = masks.cpu()
    predictions = predictions.cpu()
    
    fig, axes = plt.subplots(3, num_examples, figsize=(20, 12))
    
    for idx in range(num_examples):
        # Originalbild
        img = images[idx].numpy().transpose(1, 2, 0)
        axes[0, idx].imshow(img.squeeze(), cmap='gray')
        axes[0, idx].axis('off')
        axes[0, idx].set_title('Original')
        
        # Ground Truth Maske
        axes[1, idx].imshow(masks[idx].numpy(), cmap='gray')
        axes[1, idx].axis('off')
        axes[1, idx].set_title('Ground Truth')
        
        # Vorhergesagte Maske
        pred_mask = predictions[idx].numpy() > 0.5
        axes[2, idx].imshow(pred_mask.squeeze(), cmap='gray')
        axes[2, idx].axis('off')
        axes[2, idx].set_title('Prediction')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    unet = PupilSegmentationUNet()
    data_loader = {
        "train": create_data_loader("dataset/1k_images/host_data/training_data/binary_mask/annotations_binary_mask.csv", 
                                  EyeBinaryMaskDataset, batch_size=8),
        "train2": create_data_loader("dataset/1k_images/client_data/training_data/binary_mask/annotations_binary_mask.csv", 
                                   EyeBinaryMaskDataset, batch_size=8),
        "valid": create_data_loader("dataset/1k_images/host_data/valid_data/binary_mask/annotations_binary_mask.csv", 
                                  EyeBinaryMaskDataset, batch_size=8),
        "test": create_data_loader("dataset/1k_images/host_data/test_data/binary_mask/annotations_binary_mask.csv", 
                                 EyeBinaryMaskDataset, batch_size=8)
    }
    
    host = Host(unet, data_loader)
    host.inital_training(20)