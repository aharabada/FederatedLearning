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
    save_path: str = os.path.join(os.getcwd(), "training_process_data")
    
    def save(self):
        if self.save_data:
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, "collected_data.pkl"), "wb") as file:
                pickle.dump(self, file)
                
    def plot(self):
        for key, value in self.plot_data.items():
            if isinstance(value, list):
                value = np.array(value)
            plt.plot(value)
            plt.xlabel("Epochs")
            plt.ylabel(key)
            plt.title(f"{self.experiment} - {key}")
            if self.save_plots:
                os.makedirs(self.save_path, exist_ok=True)
                plt.savefig(os.path.join(self.save_path, f"{self.experiment}_{key}.png"))
            if self.show_plots:
                plt.show()
            plt.close()
    
    def __str__(self):
        out = "\n" + "=" * 50 + "\n"
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
    - save_path: str = cwd + "/training_process_data"
    
    Data stored in plot_data will be plotted after the function has been executed.
    
    It is mandatory when using this decorator to pass the data object as a parameter to the function.        
    """
    def wrapper(*args, **kwargs):
        data = CollectedData()
        func(*args, **kwargs, data=data)
        print(data)
        data.plot()
        data.save()
            
    return wrapper
