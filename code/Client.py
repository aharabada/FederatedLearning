import torch
import tqdm
import random
import copy

from nn_util.Metrics import DiceBCELoss

class Client:
    def __init__(self, name: str, model: torch.nn.Module, data_loader: dict[str, torch.utils.data.DataLoader]):
        self.name = name
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{self.name} Device: {self.device}")
        self.train_losses = []
        
    def train(self, iterations: int, n_datapoints: int = 0, labeling_method: str = "true labels"):
        if n_datapoints <= 0:
            n_datapoints = len(self.data_loader.dataset)
            
        self.model.to(self.device)
        
        # randomly choose n_datapoints from the dataset
        train_dataset = copy.deepcopy(self.data_loader)
        idx = random.sample(range(len(train_dataset.dataset)), n_datapoints)
        train_dataset.dataset.data = [train_dataset.dataset.data[i] for i in idx]
        train_dataset.dataset.labels = [train_dataset.dataset.labels[i] for i in idx]
        
        criterion = DiceBCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(iterations):
            self.model.train()
            total_loss = 0
            num_samples = 0
            
            for data, target in tqdm.tqdm(train_dataset, desc=f"Epoch {epoch+1}/{iterations}"):
                data = data.float().to(self.device)
                
                if labeling_method == "monte carlo dropout":
                    # Monte Carlo Inference to create own labels
                    #info = self.model.monte_carlo_inference(data, num_samples=16)
                    #target = info['mean_prediction']
                    #uncertainty = info['uncertainty']  # Jetzt uncertainty statt entropy
                    loss = self.model.mc_consistency_loss(data)
                elif labeling_method == "true labels":
                    uncertainty = 0
                elif labeling_method == "contrastive learning":
                    # TODO: implement me
                    uncertainty = 0
                else:
                    raise Exception(f"Unknown labeling method: {labeling_method}")
                    
                #target = target.float().to(self.device)
                
                optimizer.zero_grad()
                #output = self.model(data)
                #loss = criterion(output, target)
                
                # Skaliere den Loss pixel-weise mit der Unsicherheit
                #scaling_factor = 1.0 / (1.0 + uncertainty)
                
                
                #caled_loss = loss * scaling_factor
                #scaled_loss = scaled_loss.mean()  # Mitteln über alle Pixel

                #scaled_loss.backward()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += 1
            
            avg_loss = total_loss / num_samples
            self.train_losses.append(avg_loss)
            
            print(f"Epoch {epoch + 1}/{iterations} - Train Loss: {avg_loss}")
            
    def fetch_parameters(self):
        return self.model.state_dict()
    
    def update_model(self, new_parameters: dict):
        self.model.load_state_dict(new_parameters, strict=True)
    