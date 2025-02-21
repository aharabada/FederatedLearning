import torch

from nn_util.Metrics import DiceBCELoss, calculate_iou

class Client:
    def __init__(self, model: torch.nn.Module, data_loader: dict[str, torch.utils.data.DataLoader]):
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Host Device: {self.device}")
        self.train_losses = []
        self.valid_losses = []
        self.test_metrics = None
        
    def train(self, iterations: int):
        self.model.to(self.device)
        
        train_dataset = self.data_loader["train"]
        valid_dataset = self.data_loader["valid"]
        test_dataset = self.data_loader["test"]
        
        criterion = DiceBCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(iterations):
            self.model.train()
            total_loss = 0
            num_samples = 0
            
            for data, target in train_dataset:
                data = data.float().to(self.device)
                target = target.float().to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_samples += 1
                
            avg_loss = total_loss / num_samples
            self.train_losses.append(avg_loss)
            
            valid_metrics = self.__evaluate_model(valid_dataset)
            self.valid_losses.append(valid_metrics['loss'])
            
            print(f"Epoch {epoch + 1}/{iterations} - Train Loss: {avg_loss:.4f} - Valid Loss: {valid_metrics['loss']:.4f}")
            
        self.test_metrics = self.__evaluate_model(test_dataset)
        print(f"Test Loss: {self.test_metrics['loss']:.4f} - Test IoU: {self.test_metrics['iou']:.4f}")
        
        self.__save_model()
        self.__save_training_data()
        
    def recieve_parameters_from_host(self):
        pass
        
    def sent_parameters_to_host(self):
        pass