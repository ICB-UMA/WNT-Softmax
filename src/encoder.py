import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
import faiss
import pandas as pd


"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

class Encoder:
    """
    A class to encode text using a specified transformer model and fit/search using FAISS indices.

    Attributes:
        model (AutoModel): The pre-trained transformer model.
        tokenizer (AutoTokenizer): Tokenizer for the transformer model.
        f_type (str): Type of FAISS index to use. Options include "FlatL2" and "FlatIP".
        vocab (DataFrame): A pandas DataFrame containing terms and their corresponding codes.
        arr_text (list): List of terms from vocab.
        arr_codes (list): List of codes corresponding to terms in arr_text.
        arr_text_id (ndarray): Array of indices for arr_text.
        device (str): Device to run the model on.
        faiss_index (Index): The FAISS index for searching encoded texts.
    """
    def __init__(self, MODEL_NAME: str, MAX_LENGTH: int):
        """
        Initializes the encoder with a model and tokenizer

        Parameters:
            MODEL_NAME (str): The name or path of the pre-trained model.
            MAX_LENGTH (int): Maximum length of tokens for the tokenizer.
        """
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_length = MAX_LENGTH

        # Setup device and DataParallel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

    def encode(self, texts, batch_size):
        """
        Encodes the given texts into embeddings using the transformer model.

        Parameters:
            texts (list): List of text strings to encode.
            batch_size (int): The size of each batch for processing.

        Returns:
            ndarray: A numpy array of embeddings.
        """
        all_embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        for batch_idx in tqdm(range(num_batches), desc="Encoding"):
            batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
                all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)