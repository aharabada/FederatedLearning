import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

class CollectedData:
    experiment: str = "Unnamed Experiment"
    plot_data: dict[str, np.ndarray] = {}
    other_data: dict[str, any] = {}
    show_plots: bool = True
    save_plots: bool = True
    save_data: bool = True
    save_path: str = os.path.join(os.getcwd(), "..", "results")
    
    def save(self):
        if self.save_data:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, self.experiment), exist_ok=True)
            with open(os.path.join(self.save_path, self.experiment, f"{self.experiment}_collected_data.pkl"), "wb") as file:
                pickle.dump(self, file)
                
    def plot(self):
        for key, value in self.plot_data.items():
            if isinstance(value, list):
                value = np.array(value)
                
            if value.ndim == 2:
                # First plot: Mean and std
                plt.figure(figsize=(10, 6))
                mean = np.mean(value, axis=0)
                std = np.std(value, axis=0)
                plt.plot(mean, label=f"Mean (N={value.shape[0]})")
                plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
                plt.axhline(mean[0], color="red", linestyle="--", label="Initial Performance")
                plt.legend()
                plt.xlabel("Epochs")
                plt.ylabel(key)
                plt.title(f"{self.experiment} - {key} (Mean and Std)")
                if self.save_plots:
                    os.makedirs(self.save_path, exist_ok=True)
                    os.makedirs(os.path.join(self.save_path, self.experiment), exist_ok=True)
                    plt.savefig(os.path.join(self.save_path, self.experiment, f"{self.experiment}_{key}_mean_std.png"))
                if self.show_plots:
                    plt.show()
                plt.close()
                
                # Second plot: Individual lines
                plt.figure(figsize=(10, 6))
                for i in range(value.shape[0]):
                    plt.plot(value[i], label=f"Run {i+1}")
                plt.axhline(mean[0], color="red", linestyle="--", label="Initial Performance")
                plt.legend()
                plt.xlabel("Epochs")
                plt.ylabel(key)
                plt.title(f"{self.experiment} - {key} (Individual Runs)")
                if self.save_plots:
                    plt.savefig(os.path.join(self.save_path, self.experiment, f"{self.experiment}_{key}_individual.png"))
                if self.show_plots:
                    plt.show()
                plt.close()
            else:
                plt.figure(figsize=(10, 6))
                plt.axhline(value[0], color="red", linestyle="--", label="Initial Performance")
                plt.plot(value)
                plt.legend()
                plt.xlabel("Epochs")
                plt.ylabel(key)
                plt.title(f"{self.experiment} - {key}")
                if self.save_plots:
                    os.makedirs(self.save_path, exist_ok=True)
                    os.makedirs(os.path.join(self.save_path, self.experiment), exist_ok=True)
                    plt.savefig(os.path.join(self.save_path, self.experiment, f"{self.experiment}_{key}.png"))
                if self.show_plots:
                    plt.show()
                plt.close()
    
    def __str__(self):
        out = "=" * 50 + "\n"
        out += f"Experiment: {self.experiment}\nPlot Data:\n"
        for key, value in self.plot_data.items():
            out += f"{key}: {value}\n"
        out += "Other Data:\n"
        for key, value in self.other_data.items():
            out += f"{key}: {value}\n"
        out += "=" * 50
        return out


def data_collector(func):
    """
    Function decorator to collect data from the functions experiments. Access the data object
    via 'data' parameter in the function. The data object has the following attributes:
    
    - experiment: str
    - plot_data: dict[str, np.ndarray]
    - other_data: dict[str, any]
    - show_plots: bool = True
    - save_plots: bool = True
    - save_data: bool = True
    - save_path: str = cwd + "/results"
    
    Data stored in plot_data will be plotted after the function has been executed.
    
    It is mandatory when using this decorator to pass the data object as a parameter to the function.        
    """
    def wrapper(*args, **kwargs):
        data = CollectedData()
        func(*args, **kwargs, data=data)
        if data.save_data:
            print(f"\nCollected data {"and generated plots " if data.save_plots else ""}saved by 'data_collector' to '{os.path.join("results", data.experiment)}'")
        else:
            print()    
        print(data)
        data.plot()
        data.save()
            
    return wrapper
