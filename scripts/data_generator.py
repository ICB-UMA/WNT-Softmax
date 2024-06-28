
from tqdm.auto import tqdm
import os
import sys
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import *
from logger import setup_custom_logger

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def parse_args():
    """
    Parses command line arguments necessary for the script.

    Returns:
        argparse.Namespace: The namespace containing all the arguments.
    """
    parser = argparse.ArgumentParser(description="Process training data and translation settings.")

    parser.add_argument('--data_path', type=str,
                        default='../../../data/SympTEMIST/symptemist-complete_240208/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2.tsv',
                        help='Path to the training dataset. Default is ../../../data/SympTEMIST/symptemist-complete_240208/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2.tsv')
    parser.add_argument('--gaz_path', type=str,
                        default='../../../data/SympTEMIST/symptemist-complete_240208/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv',
                        help='Path to the provided gazetteer. Default is "../../../data/SympTEMIST/symptemist-complete_240208/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv"')
    parser.add_argument('--lang', type=str, default='es',
                        help='Language to which the dataset is to be translated. Default is "es".')
    args, unknown = parser.parse_known_args()
    if args.lang == "es":
        default_link_path = f'../../../data/SympTEMIST/symptemist-complete_240208/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2_complete.tsv'
    else:
        default_link_path = f'../../../data/SympTEMIST/symptemist-complete_240208/symptemist_train/subtask3-experimental_multilingual/symptemist_task3_{args.lang}.tsv'

    parser.add_argument('--link_path', type=str, default=default_link_path, 
                        help=f'Path to the provided gazetteer. Default is {default_link_path}')
    parser.add_argument('--model', type=str, default=None,
                        help='Translation model to use. Default is None, which implies that no specific model is selected.')

    parser.add_argument('--corpus', type=str, default='SympTEMIST',
                        help='Corpus to which the dataset belongs. Default is "SympTEMIST".')
    parser.add_argument('--from_spanish', type=str, default='true', choices=['true', 'false'],
                        help='Need data from another language different to spanish?.')
    parser.add_argument('--log_file', type=str, default=None,
                        help='File to log to (defaults to console if not provided)')
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs to use, separated by commas, e.g., '0' or '0,1'.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for BiEncoder.")


    return parser.parse_args()


def main(args):
    logger = setup_custom_logger('data_generator', log_file=args.log_file)
    output_dir = f'../data/{args.corpus}/{args.lang}/'    
    logger.info(f'Processing file: {os.path.basename(args.data_path)} - {args.lang}')
    df_gaz = pd.read_csv(args.gaz_path, sep='\t', dtype={"code": str})
    link_df = pd.read_csv(args.link_path, sep='\t', dtype={"code": str})
    link_df = link_df.rename(columns={'text': 'term'})
    print(link_df.head())
    code_to_sem_tag = pd.Series(df_gaz.semantic_tag.values, index=df_gaz.code).to_dict()
    if args.from_spanish == "false":
        complete_train_df = pd.read_csv(args.data_path, sep="\t", dtype={"code": str})
        val_df = pd.read_csv(args.data_path.replace("train", "val"), sep="\t", dtype={"code": str})
    else:
        train_df, val_df = load_and_split_dataframe(file_path=args.data_path, logger=logger)
        val_df = val_df.drop_duplicates().reset_index(drop=True)
        complete_train_df = pd.concat([train_df[['code', 'term']], df_gaz[['code', 'term']]], axis=0)
        complete_train_df = complete_train_df.drop_duplicates().reset_index(drop=True)
    
    os.makedirs(output_dir, exist_ok=True)


    if args.model:
        tokenizer = MarianTokenizer.from_pretrained(args.model)
        model = MarianMTModel.from_pretrained(args.model)     
        logger.info(f'Translating terms for {args.lang}.')
        
        complete_train_df = translate_dataframe_in_batches(complete_train_df, model, tokenizer, batch_size=args.batch_size, logger=logger)[['code', 'term']] 
        val_df = translate_dataframe_in_batches(val_df, model, tokenizer, batch_size=args.batch_size, logger=logger)[['code', 'term']] 

    else:
        complete_train_df = pd.concat([train_df[['code', 'term']], df_gaz[['code', 'term']]], axis=0)
        complete_train_df = complete_train_df.drop_duplicates().reset_index(drop=True)
    complete_train_df = pd.concat([complete_train_df[['code', 'term']], link_df[['code', 'term']]], axis=0)
    # Including Semantic tags
    train_df_st = add_semantic_tag(complete_train_df, code_to_sem_tag, logger)
    train_df_st = train_df_st.drop_duplicates().reset_index(drop=True)

    val_df_st = add_semantic_tag(val_df, code_to_sem_tag, logger)
    val_df_st = val_df_st.drop_duplicates().reset_index(drop=True)

    
    complete_train_df.to_csv(os.path.join(output_dir, 'train_df.tsv'), sep='\t', index=False)
    val_df[['code', 'term']].to_csv(os.path.join(output_dir, 'val_df.tsv'), sep='\t', index=False)
    train_df_st.to_csv(os.path.join(output_dir, 'train_df_st.tsv'), sep='\t', index=False)
    val_df_st.to_csv(os.path.join(output_dir, 'val_df_st.tsv'), sep='\t', index=False)

    logger.info(f'Generation data process completed. Files in: {output_dir}')


if __name__ == "__main__":
    args = parse_args()
    main(args)