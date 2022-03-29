
import os
import json
import argparse

import torch
import datasets
import pandas as pd
from transformers import AutoTokenizer, AutoModel, set_seed
from tqdm.auto import tqdm
from transformers.optimization import get_linear_schedule_with_warmup

from config import CUSTOM_PATHS, CUSTOM_PARAMS
from common import mean_pooling


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pass in parameters.')

    parser.add_argument(
        "--num_train_epochs", type=int,
        default=1
    )

    parser.add_argument(
        "--batch_size", type=int,
        default=10
    )
 
    return parser.parse_args()

def convert_to_mnli_format(df):
    ans_strs = df["ans_str"].tolist()
    query_strs = df["query_str"].tolist()

    mnli_fmt_data = []
    for i, (ans_str, query_str) in enumerate(zip(ans_strs, query_strs)):
        mnli_fmt_data.append(
            {
                "premise": ans_str,
                "hypothesis": query_str,
                "label": 0,
                "idx": i
            },
        )

    return {"data": mnli_fmt_data}

def main():
    """
    Taken from: https://www.pinecone.io/learn/fine-tune-sentence-transformers-mnr/
    """
    set_seed(5)

    args = parse_args()

    # Generate training data
    generated_qas = pd.read_csv(CUSTOM_PATHS["GEN_QA_DATA_PATH"])
    mnli_fmt_data = convert_to_mnli_format(generated_qas)
    with open(CUSTOM_PATHS["MNLI_FORMAT_DATA_PATH"], "w") as f:
        f.write(json.dumps(mnli_fmt_data, indent=4))

    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_PARAMS["MODEL_NAME"])
    model = AutoModel.from_pretrained(CUSTOM_PARAMS["MODEL_NAME"])

    dataset = datasets.load_dataset('json', data_files=CUSTOM_PATHS["MNLI_FORMAT_DATA_PATH"], field="data", split="train")

    dataset = dataset.filter(
        lambda x: True if x['label'] == 0 else False
    )

    dataset = dataset.map(
        lambda x: tokenizer(
                x['premise'], max_length=128, padding='max_length',
                truncation=True
            ), batched=True
    )

    dataset = dataset.rename_column('input_ids', 'anchor_ids')
    dataset = dataset.rename_column('attention_mask', 'anchor_mask')

    dataset = dataset.map(
        lambda x: tokenizer(
                x['hypothesis'], max_length=128, padding='max_length',
                truncation=True
        ), batched=True
    )

    dataset = dataset.rename_column('input_ids', 'positive_ids')
    dataset = dataset.rename_column('attention_mask', 'positive_mask')

    dataset = dataset.remove_columns(['premise', 'hypothesis', 'label'])

    dataset.set_format(type='torch', output_all_columns=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    cos_sim = torch.nn.CosineSimilarity()

    # set device and move model there
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f'moved to {device}')

    # define layers to be used in multiple-negatives-ranking
    cos_sim = torch.nn.CosineSimilarity()
    loss_func = torch.nn.CrossEntropyLoss()
    scale = 20.0  # we multiply similarity score by this scale value
    # move layers to device
    cos_sim.to(device)
    loss_func.to(device)

    # initialize Adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)

    # setup warmup for first ~10% of steps
    total_steps = int(len(loader) / args.batch_size)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps-warmup_steps
    )

    # 1 epoch should be enough, increase if wanted
    for epoch in range(args.num_train_epochs):
        model.train()  # make sure model is in training mode
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # zero all gradients on each new step
            optim.zero_grad()
            # prepare batches and more all to the active device
            anchor_ids = batch['anchor_ids'].to(device)
            anchor_mask = batch['anchor_mask'].to(device)
            pos_ids = batch['positive_ids'].to(device)
            pos_mask = batch['positive_mask'].to(device)
            # extract token embeddings from BERT
            a = model(
                anchor_ids, attention_mask=anchor_mask
            )
            p = model(
                pos_ids, attention_mask=pos_mask
            )
            # get the mean pooled vectors
            a = mean_pooling(a, anchor_mask)
            p = mean_pooling(p, pos_mask)
            # calculate the cosine similarities
            scores = torch.stack([
                cos_sim(
                    a_i.reshape(1, a_i.shape[0]), p
                ) for a_i in a])
            # get label(s) - we could define this before if confident of consistent batch sizes
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
            # and now calculate the loss
            loss = loss_func(scores*scale, labels)
            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()
            # update learning rate scheduler
            scheduler.step()
            # update the TDQM progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    if not os.path.exists(CUSTOM_PATHS["GEN_QA_MODEL_OUTPUT_PATH"]):
        os.makedirs(CUSTOM_PATHS["GEN_QA_MODEL_OUTPUT_PATH"])

    model.save_pretrained(CUSTOM_PATHS["GEN_QA_MODEL_OUTPUT_PATH"])


if __name__ == "__main__":
    main()