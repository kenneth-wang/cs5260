import json

import pandas as pd
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
)
import torch

MODEL_PATH = "./outputs/model/qa-gen/0.439/pretrained"
RAW_DATA_PATH = "./data/raw/answers.csv"
GEN_Q_PATH = "./data/generated_qns.csv"

model_type = "t5"


class QuestionAnswerGenerator:
    def __init__(
        self,
        model_path,
        tokenizer,
        cuda=0,
        prefix="generate qna"
    ):

        print("model_path", model_path)
        self.model = T5ForConditionalGeneration.\
            from_pretrained(model_path, from_tf=True)
        self.tokenizer = T5Tokenizer.\
            from_pretrained(tokenizer)

        self.model.eval()

        if cuda >= 0:
            self.model.to('cuda:%i' % cuda)
        self.cuda = cuda

        self.prefix = prefix

    def generate(
            self, text,
            max_length=64,
            num_return_sequences=10,
            decode=False
        ):
        try:
            input_txt = text.lower()
            if self.prefix:
                input_txt = "%s: %s" % (self.prefix, text.lower())

            input_ids = self.tokenizer.encode(input_txt, return_tensors='pt')
            if self.cuda >= 0:
                input_ids=input_ids.to('cuda:%i' % self.cuda)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=num_return_sequences
                )
        except Exception as e:
            print("An error has occured predicting %s" % text)
            print(e)
            return []

        all_queries = []

        for x in outputs:
            queries = self.tokenizer.decode(x, skip_special_tokens=True)
            if decode:
                queries = self.decode(queries)
            all_queries.append(queries)

        return all_queries

    def decode(self, text):

        try:
            return json.loads('''{%s}''' % text)
        except Exception:
            print("Error decoding ==== %s ====" % text)
            return None

    def __call__(self, text):
        return self.generate(text)



def main():

    df = pd.read_csv(RAW_DATA_PATH)

    generator = QuestionAnswerGenerator(MODEL_PATH, "t5-small", cuda=-1)

    texts = df["context"].tolist()

    txts = []
    proposed_qns = []
    proposed_ans = []
    idx_of_ans = []
    idx_of_qns = []
    q_idx = 0
    for i, text in enumerate(texts):
        qa = generator.generate(
            text,
            max_length=64, # max number of tokens for generation
            num_return_sequences=20, # number of qa-pairs to generate
            decode=False # decodes output string into dict
        )

        for q in qa:
            txts.append(text)
            idx_of_ans.append(i)
            idx_of_qns.append(q_idx)
            proposed_qns.append(q.split('", a: "')[0].split('q: "')[1])
            proposed_ans.append(q.split('", a: "')[1].split('", i:')[0])
            q_idx += 1

        df_with_qns = pd.DataFrame(
            list(
                zip(txts, proposed_qns, proposed_ans, idx_of_ans, idx_of_qns)
            ),
            columns=["ans_str", "query_str", "ans_str_span", "idx_of_ans", "idx_of_qns"]
        )
        df_with_qns.to_csv(GEN_Q_PATH, index=False)

if __name__ == "__main__":
    main()