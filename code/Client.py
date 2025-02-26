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
        self.latest_gradients = []
        self.dataset_counter = 0
       
    def train(self, iterations: int, n_datapoints: int = 0, use_mcd: bool = True, store_gradients: bool = False):
        if 0 >= n_datapoints > len(self.data_loader.dataset):
            n_datapoints = len(self.data_loader.dataset)
           
        self.model.to(self.device)
       
        train_dataset = copy.deepcopy(self.data_loader)
        start_index = ((self.dataset_counter) * n_datapoints) % len(self.data_loader.dataset)
        end_index = start_index + n_datapoints
        if end_index > len(self.data_loader.dataset):
            end_index = len(self.data_loader.dataset)
            # reset dataset counter. since we add +1 at the end, we need to set it to -1
            self.dataset_counter = -1
       
        train_dataset.dataset.data = [train_dataset.dataset.data[i] for i in range(start_index, end_index)]
        train_dataset.dataset.labels = [train_dataset.dataset.labels[i] for i in range(start_index, end_index)]
       
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
              
        for epoch in range(iterations):
            self.model.train()
            total_loss = 0
            num_samples = 0
            
            # Clear previous gradients list to avoid memory build-up
            self.latest_gradients = []
            
            for data, target in tqdm.tqdm(train_dataset, desc=f"Epoch {epoch+1}/{iterations}"):
                data = data.float().to(self.device)
                
                optimizer.zero_grad()
                
                if use_mcd:
                    loss = self.model.mc_consistency_loss(data)
                else:
                    target = target.float().to(self.device)
                    criterion = DiceBCELoss()
                    loss = criterion(self.model(data), target)
                
                loss.backward()
                
                # Store the latest gradients
                if store_gradients:
                    current_gradients = {}
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            current_gradients[name] = param.grad.clone()#.detach().cpu()
                            param.grad = None
                    
                    self.latest_gradients.append(current_gradients)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_samples += 1
                
                # Optional: force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_samples
            self.train_losses.append(avg_loss)
            
            print(f"Epoch {epoch + 1}/{iterations} - Train Loss: {avg_loss:.4f}")
        
        self.dataset_counter += 1
                  
    def fetch_parameters(self):
        return self.model.state_dict()
    
    def fetch_gradients(self):
        return self.latest_gradients
   
    def update_model(self, new_parameters: dict):
        self.model.load_state_dict(new_parameters, strict=True)