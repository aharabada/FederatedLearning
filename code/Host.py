import torch
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

from nn_util.UNet import PupilSegmentationUNet, load_model
from nn_util.DataSet import create_data_loader, EyeBinaryMaskDataset
from nn_util.Metrics import DiceBCELoss, calculate_iou


class Host:
    def __init__(self, model: torch.nn.Module, data_loader: dict[str, torch.utils.data.DataLoader]):
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Host Device: {self.device}")
        self.train_losses = []
        self.valid_losses = []
        self.test_metrics = None
        
    def evaluate_model(self, dataset):
        # set seed
        torch.manual_seed(420)
        np.random.seed(420)
        
        self.model.eval()
        criterion = DiceBCELoss()
        total_loss = 0
        total_iou = 0
        num_samples = 0
        
        with torch.no_grad():
            for data, target in dataset:
                data = data.float().to(self.device)
                target = target.float().to(self.device)
                output = self.model(data)
                total_loss += criterion(output, target).item()
                total_iou += calculate_iou(output, target).item()
                num_samples += 1
                
                data.detach()
                target.detach()
                output.detach()
        
        avg_loss = total_loss / num_samples
        avg_iou = total_iou / num_samples
        
        # reset seed
        torch.seed()
        np.random.seed()
        
        return {
            'loss': avg_loss,
            'iou': avg_iou
        }
            
    def save_model(self, path: str = "models", model_name: str = "host_model_unet_320.pth"):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, model_name)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    def __save_training_data(self, path: str = "training_process_data", file_name: str = "unet_320_dropout"):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, f"train_losses_{file_name}.npy"), np.array(self.train_losses))
        np.save(os.path.join(path, f"valid_losses_{file_name}.npy"), np.array(self.valid_losses))
        
        if self.test_metrics is not None:
            np.save(os.path.join(path, f"test_metrics_{file_name}.npy"), np.array([
                self.test_metrics['loss'],
                self.test_metrics['iou']
            ]))
        print(f"Training data saved to {path}/")
            
    def inital_training(self, iterations: int):
        self.model.to(self.device)
        
        train_dataset = self.data_loader["train"]
        valid_dataset = self.data_loader["valid"]
        test_dataset = self.data_loader["test"]
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-5)
        criterion = DiceBCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        for epoch in range(iterations):
            self.model.train()
            train_loss = 0
            num_batches = 0
            
            # Training
            for data, target in tqdm.tqdm(train_dataset, desc=f"Epoch {epoch+1}/{iterations}"):
                data = data.float().to(self.device)
                target = target.float().to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            
            # Validation
            self.model.eval()
            valid_loss = 0
            valid_batches = 0
            with torch.no_grad():
                for data, target in valid_dataset:
                    data = data.float().to(self.device)
                    target = target.float().to(self.device)
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
            
        self.save_model()
        
        print("\nEvaluating final model performance on test set...")
        self.test_metrics = self.evaluate_model(test_dataset)
        
        print("\nFinal Test Results:")
        print("=" * 50)
        print(f"Test Loss: {self.test_metrics['loss']:.6f}")
        print(f"Test IoU: {self.test_metrics['iou']:.6f}")
        print("=" * 50)
        
        # self.__save_training_data()
        
    def update_model(self, new_parameters: dict):
        self.model.load_state_dict(new_parameters, strict=True)
        
    def update_model_by_gradient(self, new_gradient: dict):
        for name, param in self.model.named_parameters():
            if name in new_gradient:
                param.grad = new_gradient[name]
                
        




############################################################################################################
# Methods and code snippets for the visualization of the model predictions
############################################################################################################

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
        img = images[idx].numpy().transpose(1, 2, 0)
        axes[0, idx].imshow(img.squeeze(), cmap='gray')
        axes[0, idx].axis('off')
        axes[0, idx].set_title('Original')
        
        axes[1, idx].imshow(masks[idx].squeeze().numpy(), cmap='gray')
        axes[1, idx].axis('off')
        axes[1, idx].set_title('Ground Truth')
        
        pred_mask = predictions[idx].numpy() > 0.5
        axes[2, idx].imshow(pred_mask.squeeze(), cmap='gray')
        axes[2, idx].axis('off')
        axes[2, idx].set_title('Prediction')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    unet = PupilSegmentationUNet()
    data_loader = {
        "train": create_data_loader("dataset/1k_images/host_data/training_data/binary_mask/annotations_binary_mask.csv", EyeBinaryMaskDataset),
        "valid": create_data_loader("dataset/1k_images/host_data/valid_data/binary_mask/annotations_binary_mask.csv", EyeBinaryMaskDataset),
        "test": create_data_loader("dataset/1k_images/host_data/test_data/binary_mask/annotations_binary_mask.csv", EyeBinaryMaskDataset)
    }
    
    host = Host(unet, data_loader)
    host.inital_training(50)

    # model = load_model("models/host_model_unet_320.pth")
    # test_loader = create_data_loader("dataset/1k_images/host_data/test_data/binary_mask/annotations_binary_mask.csv",
    #                                EyeBinaryMaskDataset, batch_size=5)
    # visualize_predictions(model, test_loader, num_examples=5)