import torch
import tqdm
import copy
from nn_util.Metrics import DiceBCELoss

class Client:
    """
    The Client class represents a federated learning client that trains a local model on its own data and communicates with a central server.
    
    Methods:
        train(iterations, n_datapoints=0, use_mcd=True, store_gradients=False): Trains the local model for a given number of iterations.
        fetch_parameters(): Fetches the local model parameters.
        fetch_gradients(): Fetches the gradients of the local model.
        update_model(new_parameters): Updates the local model with new parameters.
    """
    def __init__(self, name: str, model: torch.nn.Module, data_loader: dict[str, torch.utils.data.DataLoader]):
        self.name = name
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{self.name} Device: {self.device}")
        self.train_losses = []
        self.latest_gradients = []
        self.dataset_counter = 0
       
    def train(self, iterations: int, n_datapoints: int = 0, use_mcd: bool = True, store_gradients: bool = False) -> None:
        """
        Trains the local model for a given number of iterations.

        Args:
            iterations (int): Number of training iterations (epochs).
            n_datapoints (int, optional): Number of data points to use for training. Defaults to 0, which means using the entire dataset.
            use_mcd (bool, optional): Whether to use Monte Carlo Dropout for consistency loss. Defaults to True.
            store_gradients (bool, optional): Whether to store the gradients after each iteration. Defaults to False. (Used for FedSGD)

        Returns:
            None
        """
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
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_samples
            self.train_losses.append(avg_loss)
            
            print(f"Epoch {epoch + 1}/{iterations} - Train Loss: {avg_loss:.4f}")
        
        self.dataset_counter += 1
                  
    def fetch_parameters(self) -> dict:
        """
        Fetches the current parameters of the local model.

        Returns:
            dict: A dictionary containing the model's state_dict, which includes all the parameters of the model.
        """
        return self.model.state_dict()
    
    def fetch_gradients(self) -> list[dict[str, torch.Tensor]]:
        """
        Fetches all gradients of the local model after the last training iteration.

        Returns:
            list: A list of dictionaries containing the gradients of the model's parameters.
        """
        return self.latest_gradients
   
    def update_model(self, new_parameters: dict):
        """
        Updates the local model with new parameters.

        Args:
            new_parameters (dict): A dictionary containing the new parameters to update the model with.

        Returns:
            None
        """
        self.model.load_state_dict(new_parameters, strict=True)