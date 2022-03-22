# cs5260

## To finetune Question Generator model

python -m src.qa_gen.finetune

## To generate questions

python -m src.qa_gen.generate


## Azmi's Notes
### Information Retrieval Model Finetuning using Generated Questions

Before running, need to create output folder:
> outputs\model\model_with_gen_q

[Maybe TODO] Modify code to create output folder if it doesnt exist

**Run command**

Execute the following command in top project directory

> python src\qa_gen\finetune_with_gen_qa.py --num_train_epochs 1 --batch_size 10

The 2 arguments are needed in the command cos No defaults or not in config file
[Maybe TODO] Add the 2 parameters into the config file

### Model Evaluation

**Run command**

Execute the following command in top project directory

> python src\qa_gen\run_eval.py