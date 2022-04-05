# CS5260 Project
## Semester II 2021/2022

---

Team Members
- Raimi bin Karim: raimi.karim@u.nus.edu
- Azmi bin Mohamed Ridwan: E0576209@u.nus.edu
- Wang Tian Ming Kenneth: E0573193@u.nus.edu

## Overview
Information retrieval models generally require large amount of training data to perform well. Unfortunately, most companies do not have sufficient data for training as they might not have data collection pipelines in place to collect training data. To solve the lack of training data, we generate questions for the training set by using the Text-to-text Transfer Transformer (T5) model. We believe that the performance of the information retrieval (DistillBert) model will improve after being trained on the generated questions.

## Installation
1. Clone this repository
2. [Recommended] Create a virtual environment using either pyenv or conda
3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download and prepare data
    - See this [doc](data/README.md)

## How to Run
1. Prepare the configuration files

2. Finetune the Question generator model using SQuaD dataset
```bash
python -m src.qa_gen.finetune
```

3. Generate questions using the corpus answers

```bash
python -m src.qa_gen.generate
```

4. Finetune the pre-trained Distilbert models using original and generated questions

Create output folder based on the path defined in the configuration file (GEN_QA_MODEL_OUTPUT_PAT)

```bash
python -m src.qa_gen.finetune_with_gen_qa
```

5. Evaluate the model using the dataset

```bash
python -m src.qa_gen.run_eval
```
