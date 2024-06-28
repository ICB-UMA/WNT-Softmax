import os
import sys
import argparse
import gc
import time
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import generate_and_convert_embeddings, prepare_ntsoftmax_model, set_seed
from logger import setup_custom_logger
import encoder as enc
from WNT_Softmax import  NTSoftmaxDataset
from metrics import calculate_topk_accuracy

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Train and save weights for the NT-Softmax model and LabelEncoder.")
    parser.add_argument("--lang", type=str, default="es", help="Language of the model to train.")
    parser.add_argument("--model_save_path", type=str, default="../models/", help="Directory to save the model weights.")
    parser.add_argument("--model_path", type=str, default="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large", help="Path to the bi-encoder model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for randomness control.")
    parser.add_argument("--batch_size_be", type=int, default=32, help="Batch size for BiEncoder.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--batch_size_nt", type=int, default=32, help="Batch size for NTSoftmax training.")
    parser.add_argument("--w_type", type=bool, default=True, help="Weighted or random initialization.")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs to use, separated by commas, e.g., '0' or '0,1'.")
    parser.add_argument('--log_file', type=str, default=None, help='File to log to (defaults to console if not provided)')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length for the model')
    parser.add_argument('--corpus', type=str, default='SympTEMIST', help='Name of the corpus to process')
    parser.add_argument('--st', type=bool, default=False, help='Use semantic tag')
    parser.add_argument('--save_model', type=bool, default=False, help='Whether to save the model weights after training')
    parser.add_argument('--save_preds', type=bool, default=True, help='Save prediction results')
    return parser.parse_args()

def load_corpus_data(args):
    """Load testing data and gazetteer based on the specified corpus."""
    if args.corpus == "SympTEMIST":
        if args.lang == "es":
            test_df = pd.read_csv("../../../data/SympTEMIST/symptemist-complete_240208/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2.tsv", sep="\t", header=0, dtype={"code": str})
        else:
            test_df = pd.read_csv(f"../../../data/SympTEMIST/symptemist-complete_240208/symptemist_test/subtask3-experimental_multilingual/symptemist_task3_{args.lang}_test.tsv", sep="\t", header=0, dtype={"code": str})
        test_df = test_df.rename(columns={'text': 'term'})
        df_gaz = pd.read_csv("../../../data/SympTEMIST/symptemist-complete_240208/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv", sep="\t", header=0, dtype={"code": str})
    elif args.corpus == "MedProcNER":
        test_df = pd.read_csv("../../../data/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_test/tsv/medprocner_tsv_test_subtask2.tsv", sep="\t", header=0, dtype={"code": str})
        test_df = test_df.rename(columns={'text': 'term'})
        df_gaz = pd.read_csv("../../../data/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_gazetteer/gazzeteer_medprocner_v1_noambiguity.tsv", sep="\t", header=0, dtype={"code": str})
    elif args.corpus == "DisTEMIST":
        test_df = pd.read_csv("../../../data/DisTEMIST/distemist_zenodo/test_annotated/subtrack2_linking/distemist_subtrack2_test_linking.tsv", sep="\t", header=0, dtype={"code": str})
        test_df = test_df.rename(columns={'span': 'term'})
        df_gaz = pd.read_csv("../../../data/DisTEMIST/dictionary_distemist.tsv", sep="\t", header=0, dtype={"code": str})
    else:
        raise ValueError(f"Unsupported corpus: {args.corpus}")
    
    return test_df, df_gaz

def evaluate_and_log_performance(test_dataloader, val_dataloader, model, device, label_enc, logger, args, epoch, test_df, val_df, best_accuracy, best_epoch, output_path):
    """Evaluate and log the performance of the model on both test and validation datasets."""
    codes = model.get_candidates(test_dataloader, device, label_enc)
    test_df["codes"] = codes
    res_test = calculate_topk_accuracy(test_df, [1, 5, 25, 50, 100, 200])

    codes = model.get_candidates(val_dataloader, device, label_enc)
    val_df["codes"] = codes
    res_val = calculate_topk_accuracy(val_df, [1, 5, 25, 50, 100, 200])
    
    logger.info(f"Epoch {epoch+1} Test Results: {res_test}")
    logger.info(f"Epoch {epoch+1} Validation Results: {res_val}")

    current_accuracy = res_val[1]  
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_epoch = epoch + 1

        if args.save_model:
            torch.save(model.state_dict(), output_path)
            logger.info(f"Best model saved at: {output_path}")

    return best_accuracy, best_epoch


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    logger = setup_custom_logger('WNT-Training', log_file=args.log_file)    
    logger.info(f"Seed --> {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    output_path = os.path.join(args.model_save_path, f"wnt-softmax-{args.lang}-{args.corpus}.pth")
    data_path = f"../data/{args.corpus}/{args.lang}/train_df_st.tsv" if args.st else f"../data/{args.corpus}/{args.lang}/train_df.tsv"
    train_df = pd.read_csv(data_path, sep="\t", dtype={"code": str})
    val_df = pd.read_csv(data_path.replace("train", "val"), sep="\t", dtype={"code": str})
    val_df.drop_duplicates(inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    test_df, df_gaz = load_corpus_data(args)

    encoder = enc.Encoder(args.model_path, args.max_length)

    train_emb = generate_and_convert_embeddings(encoder, train_df, args.batch_size_be)
    val_emb = generate_and_convert_embeddings(encoder, val_df, args.batch_size_be)
    test_emb = generate_and_convert_embeddings(encoder, test_df, args.batch_size_be)

    del encoder
    torch.cuda.empty_cache()
    gc.collect()

    label_enc = LabelEncoder()
    label_enc.fit(train_df["code"].tolist())
    arr_train_gaz_label = label_enc.transform(train_df["code"].tolist())

    train_data = NTSoftmaxDataset(train_emb, arr_train_gaz_label)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_nt, shuffle=True)

    val_dataset = NTSoftmaxDataset(X=val_emb, y=val_df["code"])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_nt, shuffle=False)

    test_dataset = NTSoftmaxDataset(X=test_emb, y=test_df["code"])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_nt, shuffle=False)

    model = prepare_ntsoftmax_model(train_df, label_enc, train_emb, args.w_type, device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    start_time = time.time()
    logger.info(f"BiEncoder evaluation")

    codes = model.get_candidates(test_dataloader, device, label_enc)
    test_df["codes"] = codes
    res_test = calculate_topk_accuracy(test_df, [1, 5, 25, 50, 100, 200])

    codes = model.get_candidates(val_dataloader, device, label_enc)
    val_df["codes"] = codes
    res_val = calculate_topk_accuracy(val_df, [1, 5, 25, 50, 100, 200])

    logger.info(f"Epoch 0 Test Results: {res_test}")
    logger.info(f"Epoch 0 Validation Results: {res_val}\n")

    best_accuracy = 0
    best_epoch = -1

    for t in range(args.epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        model.fit(train_dataloader, loss_fn, optimizer, device)
        best_accuracy, best_epoch = evaluate_and_log_performance(
            test_dataloader, val_dataloader, model, device, label_enc, logger, args, t,
            test_df, val_df, best_accuracy, best_epoch, output_path
        )
    end_time = time.time()
    logger.info(f"Training Completed -> {end_time - start_time:.2f} seconds")
    if args.save_preds:
        codes = model.get_candidates(test_dataloader, device, label_enc)
        codes = [[str(code) for code in sublist] for sublist in codes]
        test_df["codes"] = codes
        #test_df["codes"] = test_df["codes"].apply(lambda lst: [str(code) for code in lst])
        print(test_df.head())
        pred_dir = f"../preds/{args.lang}/"
        os.makedirs(pred_dir, exist_ok=True)
        test_df.to_csv(os.path.join(pred_dir, f"wnt-softmax-{args.corpus}.tsv"), index=False, sep="\t")

if __name__ == "__main__":
    main()