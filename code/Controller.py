from copy import deepcopy

from Client import Client
from Host import Host
from nn_util.UNet import load_model


N_CLIENTS = 3
N_ITERATIONS_CLIENTS = 10

# TODO: Dataloader erstellen und bei den Clients und dem Host im init übergeben

host = Host(load_model(model_path="models/host_model_unet_320.pth"))

clients = [Client(deepcopy(host.model)) for _ in range(N_CLIENTS)]