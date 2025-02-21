import torch
import tqdm

from nn_util.Metrics import DiceBCELoss

class Client:
    def __init__(self, name: str, model: torch.nn.Module, data_loader: dict[str, torch.utils.data.DataLoader]):
        self.name = name
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{self.name} Device: {self.device}")
        self.train_losses = []
        self.test_metrics = None
        
    def train(self, iterations: int):
        self.model.to(self.device)
        
        train_dataset = self.data_loader
        
        criterion = DiceBCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(iterations):
            self.model.train()
            total_loss = 0
            num_samples = 0
            
            for data, _ in tqdm.tqdm(train_dataset, desc=f"Epoch {epoch+1}/{iterations}"):
                data = data.float().to(self.device)
                
                # Monte Carlo Inference to create own labels
                info = self.model.monte_carlo_inference(data)
                target = info['mean_prediction']
                target = target.float().to(self.device)
                uncertainty = info['uncertainty']  # Jetzt uncertainty statt entropy
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                
                # Skaliere den Loss pixel-weise mit der Unsicherheit
                scaling_factor = 1.0 / (1.0 + uncertainty) 
                scaled_loss = loss * scaling_factor
                scaled_loss = scaled_loss.mean()  # Mitteln über alle Pixel
                
                scaled_loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += 1
            
            avg_loss = total_loss / num_samples
            self.train_losses.append(avg_loss)
            
            print(f"Epoch {epoch + 1}/{iterations} - Train Loss: {avg_loss:.4f}")
            
    def fetch_parameters(self):
        return self.model.state_dict()
    
    def update_model(self, new_parameters: dict):
        self.model.load_state_dict(new_parameters, strict=True)
    