import torch
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

from CNN import CNN
from DataSet import create_data_loader

class Host:
    def __init__(self, model: torch.nn.Module, data_loader: dict[str, torch.utils.data.DataLoader]):
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Host Device: {self.device}")
        self.train_losses = []
        self.valid_losses = []
        self.test_metrics = None
        
    def __evaluate_model(self, dataset):
        self.model.eval()
        criterion = torch.nn.MSELoss()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in dataset:
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1, 1).float()
                output = self.model(data)
                total_loss += criterion(output, target).item()
                
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(dataset)
        
        predictions = torch.tensor(predictions)
        targets = torch.tensor(targets)
        mae = torch.mean(torch.abs(predictions - targets))
        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
        
        return {
            'mse': avg_loss,
            'mae': mae.item(),
            'rmse': rmse.item()
        }
            
    def __save_model(self, path: str = "models_2"):
        os.makedirs(path, exist_ok=True)
        
        model_path = os.path.join(path, "host_model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    def __save_training_data(self, path: str = "training_process_data_2"):
        os.makedirs(path, exist_ok=True)
        
        # Save training metrics
        np.save(os.path.join(path, "train_losses.npy"), np.array(self.train_losses))
        np.save(os.path.join(path, "valid_losses.npy"), np.array(self.valid_losses))
        
        # Save test metrics if available
        if self.test_metrics is not None:
            np.save(os.path.join(path, "test_metrics.npy"), np.array([
                self.test_metrics['mse'],
                self.test_metrics['mae'],
                self.test_metrics['rmse']
            ]))
        
        print(f"Training data saved to {path}/")
            
    def inital_training(self, iterations: int):
        torch.cuda.empty_cache()
        self.model.to(self.device)
        
        train_dataset = self.data_loader["train"]
        train2_dataset = self.data_loader["train2"] # TODO: only for training, since there is too few data by now
        valid_dataset = self.data_loader["valid"]
        test_dataset = self.data_loader["test"]
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # train the model over the iterations and visualize via tqdm
        for epoch in range(iterations):
            self.model.train()
            train_loss = 0
            for data, target in tqdm.tqdm(train_dataset):
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1, 1).float()
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # TODO: only for training, since there is too few data by now
            for data, target in tqdm.tqdm(train2_dataset):
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1, 1).float()
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # TODO end    
            
            
            self.model.eval()
            valid_loss = 0
            with torch.no_grad():
                for data, target in valid_dataset:
                    data, target = data.to(self.device), target.to(self.device)
                    target = target.view(-1, 1).float()
                    output = self.model(data)
                    valid_loss += criterion(output, target).item()
            
            # Store losses for this epoch
            epoch_train_loss = train_loss / len(train_dataset)
            epoch_valid_loss = valid_loss / len(valid_dataset)
            self.train_losses.append(epoch_train_loss)
            self.valid_losses.append(epoch_valid_loss)
            
            print(f"Epoch {epoch + 1}/{iterations}, Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}")
            
        self.__save_model()
        
        print("\nEvaluating final model performance on test set...")
        self.test_metrics = self.__evaluate_model(test_dataset)
        
        print("\nFinal Test Results:")
        print("=" * 50)
        print(f"Mean Squared Error (MSE): {self.test_metrics['mse']:.6f}")
        print(f"Mean Absolute Error (MAE): {self.test_metrics['mae']:.6f}")
        print(f"Root Mean Squared Error (RMSE): {self.test_metrics['rmse']:.6f}")
        print("=" * 50)
        
        # Save all training data
        self.__save_training_data()
        
    def federated_training(self):
        pass    


  
def load_model(model_path: str = "models/host_model.pth"):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def visualize_predictions(model, data_loader, num_examples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Get some random test examples
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        labels = labels.view(-1, 1).float()
        predictions = model(images)
        
    # Move everything to CPU for visualization
    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_examples, figsize=(20, 4))
    
    for idx in range(num_examples):
        # Get the image and reshape it
        img = images[idx].numpy().transpose(1, 2, 0)  # Change from CxHxW to HxWxC
        
        # Display the image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Add title with true label and prediction
        true_val = labels[idx].item()
        pred_val = predictions[idx].item()
        axes[idx].set_title(f'True: {true_val:.2f}\nPred: {pred_val:.2f}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # model = load_model()
    # test_loader = create_data_loader("dataset/1k_images/host_data/test_data/diameter/annotations_diameter.csv")
    # visualize_predictions(model, test_loader)
    
    cnn = CNN()
    data_loader = {
        "train": create_data_loader("dataset/1k_images/host_data/training_data/diameter/annotations_diameter.csv"),
        "train2": create_data_loader("dataset/1k_images/client_data/training_data/diameter/annotations_diameter.csv"),  # TODO: only for training, since there is too few data by now
        "valid": create_data_loader("dataset/1k_images/host_data/valid_data/diameter/annotations_diameter.csv"),
        "test": create_data_loader("dataset/1k_images/host_data/test_data/diameter/annotations_diameter.csv")
    }
    
    host = Host(cnn, data_loader)
    host.inital_training(100)
    
