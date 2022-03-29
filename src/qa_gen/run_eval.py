import os
import argparse
import json

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from config import CUSTOM_PATHS, CUSTOM_VARIABLES
from common import mean_pooling


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pass in parameters.')

    parser.add_argument(
        "--model_path", type=str,
        default="sentence-transformers/msmarco-distilbert-cos-v5"
    )
    parser.add_argument(
        "--save_preds", type=bool,
        default=False
    )
    parser.add_argument(
        "--save_preds_path", type=str
    )

    parser.add_argument(
        "--corpus_path", type=str,
        default=None
    )

    parser.add_argument(
        "--test_data_path", type=str,
        default=None
    )
 
    return parser.parse_args()


class Evaluator():
    # def __init__(self, model_path):
    def __init__(self, **kwargs):
        self.model_path = kwargs['model_path']
        self.corpus_file = kwargs['corpus_path']
        self.test_file = kwargs['test_data_path']

    def compute_scores(self, query_emb, doc_emb):
        scores = torch.mm(query_emb, doc_emb.transpose(0, 1)).cpu()
        return scores

    def encode(self, tokenizer, model, texts):
        # Tokenize sentences
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

    def calc_mrr(self, query_emb, doc_emb, ans_idx):
        
        # Compute dot score between query and all document embeddings
        # Original version: Scores for all queries calculated togther
        scores = self.compute_scores(query_emb, doc_emb)
        
        # Alternate version: If there are memory issues, can use this version 
        # which calculates the scores for each query individually and combines 
        # together so that it has the same output as the original version.        
        # for i, x in enumerate(query_emb):
        #     score = self.compute_scores(x.unsqueeze(0), doc_emb)
        #     if i == 0:
        #         scores = score
        #     else:
        #         scores = torch.cat((scores, score),0)

        ranks = []
        for i in range(len(query_emb)):
            q_orders = (-scores[i]).argsort()
            q_ranks = q_orders.argsort()
            q_rank = q_ranks[ans_idx[i]]
            ranks.append(q_rank)
        
        return ranks, sum([1/(v+1) for v in ranks])/len(ranks)

    def _generate_predictions(self, save_pred_path, query_emb, queries, doc_emb, docs, ranks, model_path, mrr):

        scores = self.compute_scores(query_emb, doc_emb)

        predictions = []
        #Combine docs & scores
        for i in range(len(query_emb)):
            rank = ranks[i]
            doc_score_pairs = list(zip(docs, scores[i]))

            #Sort by decreasing score
            doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

            #Output passages & scores
            for doc_idx, (doc, score) in enumerate(doc_score_pairs):
                if doc_idx == rank:
                    is_actual_ans = "Y"
                else:
                    is_actual_ans = ""
                predictions.append((queries[i], doc, score.item(), is_actual_ans, mrr.item()))

            if i >= CUSTOM_VARIABLES["TOP_K"]:
                break

        pred_df = pd.DataFrame(predictions, columns=["query_str", "ans_str", "score", "is_actual_ans", "MRR"])

        if not os.path.exists(save_pred_path):
            os.makedirs(save_pred_path)
        
        model_name = model_path.split("/")[-1]
        pred_df.to_csv(f"{save_pred_path}/{model_name}.csv", index=False)

    def run_eval(self, save_preds, save_preds_path):
        # Sentences we want sentence embeddings for
        # raw_df = pd.read_csv(CUSTOM_PATHS["RAW_DATA_PATH"])
        # eval_df = pd.read_csv(CUSTOM_PATHS["TEST_DATA_PATH"])
        raw_df = pd.read_csv(self.corpus_file)
        eval_df = pd.read_csv(self.test_file)

        docs = raw_df["ans_str"].tolist()

        eval_queries = eval_df["query_str"].tolist()
        eval_ans_idx = eval_df["idx_of_ans"].tolist()

        print("Loading tokenizer and model")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5")
        model = AutoModel.from_pretrained(self.model_path)

        # Encode query and docs
        print("Encoding queries")
        query_emb = self.encode(tokenizer, model, eval_queries) # 188x768
        print("Encoding corpus")
        doc_emb = self.encode(tokenizer, model, docs) # 50 x 768 x 4 bytes

        print("Calculating scores")
        ranks, mrr = self.calc_mrr(query_emb, doc_emb, eval_ans_idx)
        print(f"mrr for model located at {self.model_path}: {mrr}")

        if save_preds:
            self._generate_predictions(save_preds_path, query_emb, eval_queries, doc_emb, docs, ranks, self.model_path, mrr)

        return {'model': self.model_path, 'mrr': mrr.item()}


def main():

    # CHANGE HERE: Name of output file
    results_filename = 'results_mod_origq_1.json'
    args = parse_args()

    args.corpus_path = "data/data.csv"
    print(args.corpus_path)

    # CHANGE HERE: Path to model, either local folder or from model zoo
    # If this param is commented, then default model used (defined in parser object)
    # above
    args.model_path = "outputs/model/model_with_gen_q/1o/"
    
    all_results = [] # Save results in a list of dicts
    for i in range(1,6):
        # Enumerate over each test dataset
        args.test_data_path = "data/test_"+str(i)+".csv"
        print(args.test_data_path)

        ev = Evaluator(**vars(args))
        results = ev.run_eval(args.save_preds, args.save_preds_path)
        results['data_idx'] = i

        all_results.append(results)

    print(results)

    # save to file
    with open(results_filename, 'w') as outfile:
        json.dump(all_results, outfile)
    

if __name__ == "__main__":
    main()







