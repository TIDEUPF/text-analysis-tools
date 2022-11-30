from torch.utils.data import DataLoader,ConcatDataset
import model
import torch
import pandas as pd
import dataset as dt
import argparse
from pathlib import Path

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', required=True, type=Path, help='Path to test directory')
    parser.add_argument('--model',required=True, type=Path, help='Path to model')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    MAX_TOKEN_LEN = 200

    args = _parse_arguments()
    print('Initializing model...', flush=True)
    device = torch.cuda.current_device() if torch.cuda.is_available() else None

    model = model.BertClassifier.load_from_checkpoint(checkpoint_path=str(args.model),strict=False).to(device)

    df = pd.read_csv(args.test)
    a = df[df['sentences'].isna()]
    df.drop(a.index,inplace=True)
    data_module = dt.CSdatamodule(
        df,
        df,
        df,
        tokenizer,
        batch_size=args.batch_size,
        max_token_len=MAX_TOKEN_LEN)

    data_module.setup()


    model.eval()
    result = [0]*31280
    with torch.no_grad():
        i = 0
        for x in data_module.test_dataloader():
            prediction = model(x['input_ids'].to(device, non_blocking=True),x['attention_mask'].to(device, non_blocking=True))
            result[i]=int(torch.round(prediction[1]))
            i = i + 1
            print(i)



    df['category_index'] = result
    df.to_csv('finalClassbySentencesSimple21.csv')












