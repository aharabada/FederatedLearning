from copy import deepcopy
import random
import torch

from Client import Client
from Host import Host
from nn_util.UNet import load_model
from nn_util.DataSet import create_data_loader, EyeBinaryMaskDataset


class FedAvgController:
    client_names = [
        "Hans", "Fritz", "Günther", "Klaus", "Wolfgang",
        "Bernd", "Uwe", "Dieter", "Heinz", "Horst",
        "Jürgen", "Rainer", "Gerhard", "Manfred", "Erwin",
        "Otto", "Wilhelm", "Karl", "Ludwig", "Siegfried",
        "Herbert", "Norbert", "Egon", "Helmut", "Reinhard",
        "Monika", "Gisela", "Brigitte", "Ursula", "Renate",
        "Ingrid", "Elke", "Helga", "Gertrud", "Marianne",
        "Roswitha", "Waltraud", "Hannelore", "Margot", "Erika",
        "Bärbel", "Christa", "Anneliese", "Ilse", "Edith",
        "Heidemarie", "Petra", "Elfriede", "Dorothea", "Friederike"
    ]
    N_CLIENTS = 2
    CLIENT_ITERATIONS = 1
    CLIENT_BATCH_SIZE = 4
    
    def __init__(self, path_to_host_model: str, config: dict = {}):
        # load config
        self.N_CLIENTS = config.get("N_CLIENTS", self.N_CLIENTS)
        self.CLIENT_ITERATIONS = config.get("CLIENT_ITERATIONS", self.CLIENT_ITERATIONS)
        self.CLIENT_BATCH_SIZE = config.get("CLIENT_BATCH_SIZE", self.CLIENT_BATCH_SIZE)
        # prepare host
        self.host = Host(load_model(model_path=path_to_host_model), None)
        self.host.model.to(self.host.device)

        # prepare client dataloaders and clients
        client_dataloader = create_data_loader("dataset/1k_images/client_data/training_data/binary_mask/annotations_binary_mask.csv", 
                                               dataset_type=EyeBinaryMaskDataset, 
                                               batch_size=self.CLIENT_BATCH_SIZE, 
                                               preprocess=True)
        self.clients = []

        n_samples_per_client = len(client_dataloader.dataset) // self.N_CLIENTS
        for i in range(self.N_CLIENTS):
            dataloader = create_data_loader("", dataset_type=EyeBinaryMaskDataset, batch_size=self.CLIENT_BATCH_SIZE, preprocess=False, length=n_samples_per_client)
            dataloader.dataset.data = client_dataloader.dataset.data[(i * n_samples_per_client):((i + 1) * n_samples_per_client)]
            dataloader.dataset.labels = client_dataloader.dataset.labels[(i * n_samples_per_client):((i + 1) * n_samples_per_client)]
            
            self.clients.append(Client(self.client_names.pop(random.randint(0, len(self.client_names) - 1)), deepcopy(self.host.model), dataloader))

        del client_dataloader
        
    def __aggregate_parameters(self, client_parameters: list[dict]):
        new_parameters = {}
        n_clients = len(client_parameters)
        
        # Initialisiere mit den Parametern des ersten Clients
        for name, param in client_parameters[0].items():
            new_parameters[name] = torch.zeros_like(param, dtype=torch.float32)
            
            # Einfacher Durchschnitt über alle Clients
            for client_params in client_parameters:
                new_parameters[name] += client_params[name].float() / n_clients
                
            # Konvertiere zurück zum ursprünglichen Typ wenn nötig
            if param.dtype != torch.float32:
                new_parameters[name] = new_parameters[name].to(dtype=param.dtype)
        
        return deepcopy(new_parameters)
        
    
    def test_model(self):
        test_dataset = create_data_loader("dataset/1k_images/final_test/binary_mask/annotations_binary_mask.csv", 
                                          dataset_type=EyeBinaryMaskDataset, 
                                          batch_size=self.CLIENT_BATCH_SIZE, 
                                          preprocess=True)
        
        print("\nEvaluating model performance on test set...")
        test_metrics = self.host.evaluate_model(test_dataset)
        
        print("\nTest Results:")
        print("=" * 50)
        print(f"Test Loss: {test_metrics['loss']:.6f}")
        print(f"Test IoU: {test_metrics['iou']:.6f}")
        print("=" * 50)
        
    def run(self, iterations):
        for iteration in range(iterations):
            print(f"\nStarting round {iteration + 1}/{iterations}...")
            
            # train each client
            for i, client in enumerate(self.clients):
                print(f"\nTraining Client {client.name} ({i + 1}/{self.N_CLIENTS})...")
                client.train(self.CLIENT_ITERATIONS)
                
            # fetch client data
            print("\nFetching client data...")
            client_parameters = []
            for client in self.clients:
                client_parameters.append(client.fetch_parameters())
            
            # aggregate client parameters
            print("\nAggregating client parameters...")
            new_parameters = self.__aggregate_parameters(client_parameters)
            
            # send new parameters to host
            print("\nUpdating host model")
            self.host.update_model(new_parameters)
            
            # send new parameters to clients
            print("\nSending updated model to clients...")
            for client in self.clients:
                client.update_model(self.host.model.state_dict())





        
        
if __name__ == "__main__":
    ROUNDS = 1
    controller = FedAvgController("models/host_model_unet_320_dropout_75.pth")
    print(f"\nPerformence before FedAvg (Rounds: {ROUNDS}, Clients: {controller.N_CLIENTS}, Client Iterations: {controller.CLIENT_ITERATIONS} Client BatchSize: {controller.CLIENT_BATCH_SIZE}):")
    controller.test_model()
    controller.run(ROUNDS)
    print(f"Performence after FedAvg (Rounds: {ROUNDS}, Clients: {controller.N_CLIENTS}, Client Iterations: {controller.CLIENT_ITERATIONS} Client BatchSize: {controller.CLIENT_BATCH_SIZE}):")
    controller.test_model()