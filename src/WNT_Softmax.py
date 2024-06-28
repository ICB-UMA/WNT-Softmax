import torch
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

class NTSoftmaxDataset(torch.utils.data.Dataset):
    """
    Dataset for storing features and labels for training the NTSoftmax model.

    Attributes:
        X (np.ndarray): The input features for the model.
        y (np.ndarray): The labels corresponding to the input features.
    """
    def __init__(self, X, y):
        """
        Initialize the dataset with features and labels.

        Parameters:
            X (np.ndarray): The input features.
            y (np.ndarray): The corresponding labels.
        """
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        """
        Retrieve an item by index.

        Parameters:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the feature and label at the specified index.
        """
        return self.X[idx], self.y[idx]

    def __len__(self):
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items.
        """
        return len(self.X)
    
class NTSoftmax(torch.nn.Module):
    """
    A neural network module implementing the NTSoftmax classification layer.

    Attributes:
        temperature (float): Temperature scaling factor applied to softmax.
        weight_vectors (torch.nn.Parameter): Trainable weight vectors for classes.
    """
    def __init__(self, arr_weight_vectors_init=np.array([]), temperature=1.0, num_classes=None, embed_dim=768):
        """
        Initialize the NTSoftmax layer.

        Parameters:
            arr_weight_vectors_init (np.ndarray, optional): Initial weight vectors for the classes.
            temperature (float, optional): Temperature scaling factor.
            num_classes (int, optional): Number of classes if initializing without predefined weights.
            embed_dim (int, optional): Dimensionality of the embedding.
        """
        super().__init__()
        self.temperature = temperature
        if arr_weight_vectors_init.size == 0:
            self.weight_vectors = torch.nn.Parameter(
                torch.empty((num_classes, embed_dim), dtype=torch.float32)
            )
            torch.nn.init.kaiming_uniform_(self.weight_vectors, a=math.sqrt(5))
        else:
            self.weight_vectors = torch.nn.Parameter(torch.from_numpy(arr_weight_vectors_init), requires_grad=True)

    def forward(self, x):
        """
        Perform the forward pass of the NTSoftmax layer.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output of the softmax layer after applying temperature scaling.
        """
        x = torch.mm(torch.nn.functional.normalize(x, p=2.0), torch.nn.functional.normalize(self.weight_vectors, p=2.0).T) / self.temperature
        return x

    def fit(self, dataloader, loss_fn, optimizer, device):
        """
        Train the model using the given dataloader and optimizer.

        Parameters:
            dataloader (DataLoader): DataLoader providing the training batches.
            loss_fn (callable): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (torch.device): The device tensors will be sent to.
        """
        size = len(dataloader.dataset)
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = self(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def get_candidates(self, dataloader, device, label_encoder):
        """
        Get predictions for each item in the dataloader using a list of lists.

        Parameters:
            dataloader (DataLoader): DataLoader providing the data.
            device (torch.device): The device tensors will be sent to.
            label_encoder (LabelEncoder): Encoder to transform indices to labels.

        Returns:
            list of lists: List of predicted labels for each sample.
        """
        self.eval()  
        predictions = []  
        
        with torch.no_grad():
            for X, _ in dataloader:
                batch_predictions = self(X.to(device)).detach().cpu()
                top_k_indices = torch.topk(batch_predictions, k=200, dim=-1).indices.numpy()
                decoded_predictions = [label_encoder.inverse_transform(indices) for indices in top_k_indices]
                predictions.extend(decoded_predictions)
        
        return predictions