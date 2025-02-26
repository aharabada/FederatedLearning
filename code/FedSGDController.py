from copy import deepcopy, copy
import random
import torch
import os
import gc
import json

from Client import Client
from Host import Host
from DataCollector import data_collector
from nn_util.UNet import load_model
from nn_util.DataSet import create_data_loader, EyeBinaryMaskDataset
from nn_util.Names import names


class FedSGDController:
    """
    FedSGDController orchestrates the federated learning process using Stochastic Gradient Descent (SGD).
    
    This class handles the initialization of the host and clients, the aggregation of gradients, 
    and the overall training and evaluation process across multiple rounds and runs.
    
    Methods:
        run(test: bool, data: DataCollector) -> None:
            Runs the federated learning process for the specified number of rounds and runs.
        test_model() -> dict:
            Evaluates the host model on the test set and returns the loss and IoU.
        
        __initiate_host_and_clients() -> None: Initializes the host and clients.
        __aggregate_gradients(client_gradients: list[dict]) -> dict: Aggregates the gradients of the clients.
        __save_config(path: str) -> None: Saves the configuration of the experiment.
    """
    RUNS = 1
    N_CLIENTS = 4
    N_DATAPOINTS_PER_ROUND = 64
    CLIENT_ITERATIONS = 1
    ROUNDS = 10
    USE_MCD = True
    
    def __init__(self, path_to_host_model: str, config: dict = {}):
        self.client_names = copy(names)
        self.path_to_host_model = path_to_host_model
        self.host = None
        self.clients = None
        # load config
        self.N_CLIENTS = config.get("N_CLIENTS", self.N_CLIENTS)
        self.CLIENT_ITERATIONS = config.get("CLIENT_ITERATIONS", self.CLIENT_ITERATIONS)
        self.N_DATAPOINTS_PER_ROUND = config.get("N_DATAPOINTS_PER_ROUND", self.N_DATAPOINTS_PER_ROUND)
        self.ROUNDS = config.get("ROUNDS", self.ROUNDS)
        self.USE_MCD = config.get("USE_MCD", self.USE_MCD)
        self.RUNS = config.get("RUNS", self.RUNS)
        # generate experiment name
        self.experiment_name = f"FedSTD_{"MCD" if self.USE_MCD else "GT"}_N{self.RUNS}_R{self.ROUNDS}_C{self.N_CLIENTS}_I{self.CLIENT_ITERATIONS}_D{self.N_DATAPOINTS_PER_ROUND}"
        
    def __initiate_host_and_clients(self):
        """
        Initializes the host and clients for the Federated Averaging process.
        """
        # delete old host and clients
        if self.host is not None:
            del self.host
        if self.clients is not None and len(self.clients) > 0:
            for client in self.clients:
                del client
            self.clients = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # prepare host
        self.host = Host(load_model(model_path=self.path_to_host_model), None)
        self.host.model.to(self.host.device)

        # prepare client dataloaders and clients
        client_dataloader = create_data_loader("dataset/1k_images/client_data/training_data/binary_mask/annotations_binary_mask.csv", 
                                               dataset_type=EyeBinaryMaskDataset, 
                                               batch_size=1, 
                                               preprocess=True)
        self.clients = []

        n_samples_per_client = len(client_dataloader.dataset) // self.N_CLIENTS
        for i in range(self.N_CLIENTS):
            dataloader = create_data_loader("", dataset_type=EyeBinaryMaskDataset, batch_size=1, preprocess=False, length=n_samples_per_client)
            dataloader.dataset.data = client_dataloader.dataset.data[(i * n_samples_per_client):((i + 1) * n_samples_per_client)]
            dataloader.dataset.labels = client_dataloader.dataset.labels[(i * n_samples_per_client):((i + 1) * n_samples_per_client)]
            
            self.clients.append(Client(self.client_names.pop(random.randint(0, len(self.client_names) - 1)), deepcopy(self.host.model), dataloader))

        del client_dataloader
        
    def __aggregate_gradients(self, client_gradients: list[dict]):
        """
        Aggregates the gradients of all clients.
        """
        new_parameters = {}
        for name, param in self.host.model.named_parameters():
            new_parameters[name] = torch.stack([client_gradient[name] for client_gradient in client_gradients]).mean(dim=0)
        return new_parameters
    
    def __save_config(self, path: str):
        """
        Saves the configuration of the experiment.
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as file:
            json.dump({
                "RUNS": self.RUNS,
                "ROUNDS": self.ROUNDS,
                "N_CLIENTS": self.N_CLIENTS,
                "CLIENT_ITERATIONS": self.CLIENT_ITERATIONS,
                "N_DATAPOINTS_PER_ROUND": self.N_DATAPOINTS_PER_ROUND,
                "USE_MCD": self.USE_MCD
            }, file)
        
    def test_model(self) -> dict:
        """
        Tests the model performance on a test dataset.

        Returns:
            dict: A dictionary containing the test loss and IoU metrics.
        """
        test_dataset = create_data_loader("dataset/1k_images/final_test/binary_mask/annotations_binary_mask.csv", 
                                          dataset_type=EyeBinaryMaskDataset, 
                                          batch_size=8, 
                                          preprocess=True)
        
        print("\nEvaluating model performance on test set...")
        
        test_metrics = self.host.evaluate_model(test_dataset)
        
        print("\nTest Results:")
        print("=" * 50)
        print(f"Test Loss: {test_metrics["loss"]:.6f}")
        print(f"Test IoU: {test_metrics["iou"]:.6f}")
        print("=" * 50)
        return test_metrics
        
    @data_collector
    def run(self, test: bool, data) -> None:
        """
        Runs the Federated Averaging process.

        Args:
            test (bool): If True, the model will be tested after each round.
            data (DataCollector): An instance of DataCollector to store experiment data.
        """
        # setting up data collector
        data.experiment = self.experiment_name
        data.plot_data = {"Loss": [], "IoU": []}
        
        # save config
        self.__save_config(os.path.join(data.save_path, data.experiment))

        for _ in range(self.RUNS):
            self.__initiate_host_and_clients()
            
            losses = []
            ious = []
            
            # initial testing (if requested)
            if test:
                test_metrics = self.test_model()
                losses.append(test_metrics["loss"])
                ious.append(test_metrics["iou"])
                    
            for iteration in range(self.ROUNDS):
                print(f"\nStarting round {iteration + 1}/{self.ROUNDS}...")
                print("=" * 50)
                            
                # train each client
                for i, client in enumerate(self.clients):
                    print(f"\nTraining Client {client.name} ({i + 1}/{self.N_CLIENTS})...")
                    client.train(self.CLIENT_ITERATIONS,
                                 n_datapoints=self.N_DATAPOINTS_PER_ROUND,
                                 use_mcd=self.USE_MCD,
                                 store_gradients=True)
                    
                # fetch client data
                print("\nFetching client data...")
                client_gradients = []
                for client in self.clients:
                    client_gradients.extend(client.fetch_gradients())
                
                # aggregate client parameters
                print("Aggregating client parameters...")
                new_gradient = self.__aggregate_gradients(client_gradients)
                
                del client_gradients
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # send new parameters to host
                print("Updating host model")
                self.host.update_model_by_gradient(new_gradient)
                
                # send new parameters to clients
                print("Sending updated model to clients...")
                for client in self.clients:
                    client.update_model(self.host.model.state_dict())
                    
                # testing host (if test is True)
                if test:
                    test_metrics = self.test_model()
                    losses.append(test_metrics["loss"])
                    ious.append(test_metrics["iou"])
                    
            data.plot_data["Loss"].append(losses)
            data.plot_data["IoU"].append(ious)
