import torch
import tqdm
import os

from CNN import CNN
from DataSet import create_data_loader

class Host:
    def __init__(self, model: torch.nn.Module, data_loader: dict[str, torch.utils.data.DataLoader]):
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Host Device: {self.device}")
        
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
            
    def __save_model(self, path: str = "models"):
        os.makedirs(path, exist_ok=True)
        
        model_path = os.path.join(path, "host_model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    def inital_training(self, iterations: int):
        torch.cuda.empty_cache()
        self.model.to(self.device)
        
        train_dataset = self.data_loader["train"]
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
                # Target-Tensor umformen und Datentyp anpassen
                target = target.view(-1, 1).float()  # Neue Zeile
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            valid_loss = 0
            with torch.no_grad():
                for data, target in valid_dataset:
                    data, target = data.to(self.device), target.to(self.device)
                    target = target.view(-1, 1).float()
                    output = self.model(data)
                    valid_loss += criterion(output, target).item()
            
            print(f"Epoch {epoch + 1}/{iterations}, Train Loss: {train_loss / len(train_dataset)}, Valid Loss: {valid_loss / len(valid_dataset)}")
            
        self.__save_model()
        
        print("\nEvaluating final model performance on test set...")
        test_metrics = self.__evaluate_model(test_dataset)
        
        print("\nFinal Test Results:")
        print("=" * 50)
        print(f"Mean Squared Error (MSE): {test_metrics['mse']:.6f}")
        print(f"Mean Absolute Error (MAE): {test_metrics['mae']:.6f}")
        print(f"Root Mean Squared Error (RMSE): {test_metrics['rmse']:.6f}")
        print("=" * 50)
        
    def federated_training(self):
        pass    
    
if __name__ == "__main__":
    def load_model(model_path: str = "models/host_model.pth"):
        model = CNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    cnn = CNN()
    data_loader = {
        "train": create_data_loader("dataset/1k_images/host_data/training_data/diameter/annotations_diameter.csv"),
        "valid": create_data_loader("dataset/1k_images/host_data/valid_data/diameter/annotations_diameter.csv"),
        "test": create_data_loader("dataset/1k_images/host_data/test_data/diameter/annotations_diameter.csv")
    }
    
    host = Host(cnn, data_loader)
    host.inital_training(100)