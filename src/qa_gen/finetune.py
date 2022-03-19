import json
import argparse
import logging

import numpy as np
import tensorflow as tf

from src.qa_gen.model import ModelTF, QuestionAnswerGenerator

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", type=str, default="data/raw", help="Directory for training data")
parser.add_argument("--output_dir", type=str, default="outputs/model/qa-gen/", help="Directory to save models")
parser.add_argument("--base_model", type=str, default="t5-small", help="Base model to start training from")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batchsize_train", type=int, default=1, help="Batch size for training")
parser.add_argument("--batchsize_test", type=int, default=1, help="Batch size for testing")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")

args = parser.parse_args()


model_output_dir = args.output_dir

def replace_keys(parse_dict, keypairs):
    new_dict = parse_dict.replace('"%s":' % keypairs[0], '%s:' % keypairs[1])
    return new_dict

def load_train_squad():

    dict_pairs = [
        ["question", "q"],
        ["answer", "a"],
        ["answer_start", "i"],
        ["text", "t"]
    ]

    with open('%s/train-v2.0.json' % args.input_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return_data = []

    for qna_group in data["data"]:
        for paragraph in qna_group['paragraphs']:
            for qna in paragraph["qas"]:
                if len(qna["answers"]) == 1:
                    new_dict = {}
                    new_dict['question'] = qna['question']
                    new_dict['answer'] = qna['answers'][0]['text']
                    new_dict["answer_start"] = qna['answers'][0]['answer_start']
                    new_dict = json.dumps(new_dict)
                    for keypairs in dict_pairs:
                        new_dict = replace_keys(new_dict, keypairs)
                    return_data.append(
                        ['generate qna: ' + paragraph['context'].lower() + "</s>",
                        new_dict.lower() + "</s>"]
                    )

    return return_data

def create_train_data():
    squad = load_train_squad()
    return np.array(squad)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self.model.save_pretrained("%s/%.3f/pretrained" % (model_output_dir, logs['loss']))
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

if __name__ == '__main__':
    print("Starting training script")

    # Start experiment
    log_level = logging.INFO
    logging.basicConfig(level=log_level)

    # Initialize model
    logger.info("Initializing model")

    strategy = tf.distribute.MirroredStrategy()
    logger.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Initialze callbacks
    learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    T5 = ModelTF(args.base_model)
    with strategy.scope():
        T5.model = QuestionAnswerGenerator.from_pretrained(args.base_model)
        T5.model.compile(optimizer=optimizer)

    checkpoint_filepath = model_output_dir + "/T5-{epoch:04d}-{loss:.4f}.ckpt"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=False,
        save_freq=5000
    )

    logger.info("Preparing training data")

    # Create Dataset
    train_dataloader = create_train_data()
    train_data = T5.encode(train_dataloader)
    train_data = T5.to_tf_dataset(train_data)
    train_data = train_data.shuffle(len(train_dataloader)).batch(args.batchsize_train)
    train_data = train_data.repeat(args.epochs)

    ntrain = len(train_dataloader)
    steps = int(np.ceil(ntrain//args.batchsize_train))

    logger.info("Begin model training")
    T5.model.fit(
        train_data, epochs=args.epochs, steps_per_epoch=steps,
        callbacks=[model_checkpoint_callback, CustomCallback()],
        initial_epoch=0,
        batch_size=args.batchsize_train
    )
