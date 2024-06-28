import torch
import os
import numpy as np
import encoder as enc
from WNT_Softmax import NTSoftmax
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def generate_and_convert_embeddings(encoder:enc, dataframe:pd.DataFrame(), batch_size:int, logger=None):
    """
    Generates embeddings for terms in a dataframe using a encoder,
    and ensures all embeddings are converted to float32 format.

    :param encoder: The model used to generate embeddings.
    :param dataframe: A pandas DataFrame containing a "term" column with the terms to be encoded.
    :param batch_size: The batch size for processing the terms.
    :param logger: Logger module.
    :return: A NumPy array with the embeddings converted to float32.
    """
    if logger:
        logger.info("Generating and converting embeddings.")
    embeddings = encoder.encode(dataframe["term"].tolist(), batch_size=batch_size)

    converted_embeddings = []
    for embedding in embeddings:
        try:
            converted_embedding = np.array(embedding, dtype=np.float32)
            converted_embeddings.append(converted_embedding)
        except ValueError as e:
            print(f"Error converting embedding: {e}")
            print(embedding)

    if len(converted_embeddings) == len(embeddings):
        embeddings_array = np.stack(converted_embeddings)
        return embeddings_array
    else:
        raise ValueError("Not all embeddings could be successfully converted.")


def prepare_ntsoftmax_model(df_train, label_enc, train_emb, w_type, device):
    """
    Prepares and initializes the NTSoftmax model based on provided training data and embeddings.

    Parameters:
        df_train (pandas.DataFrame): DataFrame containing training data with codes.
        label_enc (LabelEncoder): An instance of LabelEncoder with fitted class labels.
        train_emb (numpy.ndarray): Array of embeddings corresponding to the training data.
        w_type (bool): Flag indicating whether to use weighted embeddings.
        device (str or torch.device): Device to which the model should be allocated.

    Returns:
        torch.nn.Module: An instance of NTSoftmax possibly wrapped in DataParallel if multiple GPUs are available.
    """
    if w_type:
        # Create a dictionary to hold indices for each class
        dict_train_gaz_code_int_term = {code: [] for code in label_enc.classes_}
        for index, row in tqdm(df_train.iterrows(), desc="Generating Weights", total=df_train.shape[0]):
            code = row['code']
            dict_train_gaz_code_int_term[code].append(index)
        
        # Compute mean embeddings for each class
        arr_train_gaz_code_emb = []
        for code in label_enc.classes_:
            indices = dict_train_gaz_code_int_term[code]
            if indices:
                embeddings = train_emb[indices]
                mean_embedding = np.mean(embeddings, axis=0)
                arr_train_gaz_code_emb.append(mean_embedding)
            else:
                arr_train_gaz_code_emb.append(np.zeros_like(train_emb[0]))

        arr_train_gaz_code_emb = np.array(arr_train_gaz_code_emb)
        model = NTSoftmax(arr_weight_vectors_init=arr_train_gaz_code_emb, temperature=1.0)
    else:
        # Initialize without specific weights
        EMBED_DIM = train_emb.shape[-1]
        num_classes = len(label_enc.classes_)
        model = NTSoftmax(arr_weight_vectors_init=None, temperature=1.0, embed_dim=EMBED_DIM, num_classes=num_classes)

    # Utilize DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Move the model to the specified device
    model.to(device)

    return model

def set_seed(seed_value:int=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility across multiple runs. It configures seeds for numpy,
    Python, and PyTorch (both CPU and GPU), and sets deterministic algorithms for CUDA operations.

    Args:
        seed_value (int): The seed number to use for random number generation. Default is 42.

    Note:
        Setting `torch.backends.cudnn.deterministic` to True can have a performance impact, especially on GPUs. It ensures
        that CUDA convolution operations are deterministic, but at the cost of potential slowdowns.
    """
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    os.environ['PYTHONHASHSEED'] = str(seed_value)  

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TranslationDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for handling the tokenization and encoding of terms for translation.

    Attributes:
        terms (list): A list of terms (strings) to be translated.
        tokenizer: A tokenizer instance compatible with the terms' language and model.
        max_length (int): Maximum token length for each term. Defaults to 256.

    Methods:
        __len__(): Returns the number of terms in the dataset.
        __getitem__(idx): Returns the tokenized and encoded data for the term at the provided index.
    """
    def __init__(self, terms, tokenizer, max_length=256):
        self.terms = terms
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.terms)

    def __getitem__(self, idx):
        term = self.terms[idx]
        inputs = self.tokenizer(term, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        return inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0)

def collate_fn(batch):
    """
    A function to collate data samples into batch tensors, handling padding for variable lengths.

    Args:
        batch (list): A list of tuples, each containing tokenized input IDs and attention masks.

    Returns:
        tuple: Two tensors; the padded input IDs and the corresponding attention masks, both batch-first.
    """
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return input_ids_padded, attention_mask_padded

def translate_dataframe_in_batches(dataframe, model, tokenizer, batch_size=32, logger=None):
    """
    Translates terms in a DataFrame in batches using a specified model and tokenizer.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing terms to be translated under the "term" column.
        model: The translation model, already loaded and configured.
        tokenizer: The tokenizer for the model, set up for the specific language pair.
        batch_size (int): The size of each batch for translation. Defaults to 32.
        logger (logging.Logger or Custom Logger): Logger for logging messages. If provided, will log the translation process.

    Returns:
        pd.DataFrame: The DataFrame with the "term" column updated to the translated terms.
    """
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model.cuda()
    device = next(model.parameters()).device
    dataset = TranslationDataset(dataframe["term"].tolist(), tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    translated_terms = []
    model.eval()
    if logger:
        logger.info(f'Starting batch translation with batch size {batch_size}.')
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Translating in batches"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

            with torch.cuda.amp.autocast():
                outputs = actual_model.generate(input_ids=input_ids, attention_mask=attention_mask)
            
            for output in outputs:
                translated_terms.append(tokenizer.decode(output, skip_special_tokens=True))

    dataframe["term"] = translated_terms
    if logger:
        logger.info('Batch translation completed.')
    return dataframe

def load_and_split_dataframe(file_path, test_size:int=0.3, stratify:str="code", logger=None):
    """
    Loads a DataFrame from a tab-separated file, preprocesses it, and splits into training and testing sets.

    Args:
        file_path (str): The file path to load the DataFrame from.
        test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.3.
        stratify (str): The column to use for stratifying the split. Defaults to "code".
        logger (logging.Logger or Custom Logger): Logger for logging messages. If provided, will log the translation process.

    Returns:
        tuple: Two DataFrames, the training DataFrame and the testing DataFrame.
    """
    if logger:
        logger.info(f'Loading and splitting dataframe from {file_path}.')
    dataframe = pd.read_csv(file_path, sep="\t", dtype={"code": str})
    dataframe.rename(columns={'text': 'term'}, inplace=True)
    dataframe = dataframe.loc[dataframe['code'] != 'NO_CODE']
    dataframe = dataframe[dataframe['code'] != 'NOMAP']
    dataframe['term'] = dataframe['term'].str.replace(r"«|»", "", regex=True)
    dataframe = dataframe[~dataframe['code'].str.contains("\+")]
    dataframe['term'] = dataframe['term'].str.lower()

    dataframe = dataframe.reset_index(drop=True)
    unique_codes = dataframe['code'].value_counts()[dataframe['code'].value_counts() == 1].index.tolist()
    unique_cases = dataframe[dataframe['code'].isin(unique_codes)]
    df_to_split = dataframe[~dataframe['code'].isin(unique_codes)]
    train_df, test_df = train_test_split(df_to_split, test_size=test_size, random_state=42, stratify=df_to_split[stratify])
    train_df = pd.concat([train_df, unique_cases]).drop_duplicates().reset_index(drop=True)
    if logger:
        logger.info('Dataframe loaded and split completed.')
    return train_df, test_df

def add_semantic_tag(df, code_to_sem_tag, logger = None):
    """
    Adds semantic tags to a DataFrame based on a mapping from codes to semantic tags.

    This function iterates over each row of the DataFrame. If a code in the 'code' column of the DataFrame
    is found in the `code_to_sem_tag` dictionary, it appends the corresponding semantic tag to the 'term'
    column of the DataFrame within square brackets.

    Parameters:
    - df (pd.DataFrame): The DataFrame to which semantic tags will be added. Assumes columns 'code' and 'term'.
    - code_to_sem_tag (dict): A dictionary mapping codes (keys) to semantic tags (values).
    - logger (logging.Logger, optional): A logger for logging the process. If provided, logs the process of adding tags.

    Returns:
    - pd.DataFrame: The modified DataFrame with semantic tags added to the 'term' column.
    """
    if logger:
        logger.info('Adding semantic tags to dataframe directly based on the code.')
    df = df.copy()
    for index, row in df.iterrows():
        if row['code'] in code_to_sem_tag:
            df.loc[index, 'term'] = f"{row['term']} [{code_to_sem_tag[row['code']]}]"
    return df