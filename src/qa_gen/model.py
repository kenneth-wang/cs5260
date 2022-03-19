import logging

import tensorflow as tf

from transformers import AutoTokenizer, TFT5ForConditionalGeneration

model_type = "t5"

class QuestionAnswerGenerator(TFT5ForConditionalGeneration):
    """
    Trains a question-answer generator model using tensorflow
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker= tf.keras.metrics.Mean(name="loss")

    def train_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        lr = self.optimizer._decayed_lr(tf.float32)

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({"lr": lr})

        return metrics

    def test_step(self, data):
        x = data
        y = x["target_text"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

class ModelTF:
    """
    T5 model in Tensorflow
    """
    def __init__(self, model_name=None):

        self.logger = logging.getLogger(__name__)
        self.model = None
        if model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, data, encoder_max_len=512, decoder_max_len=64):

        input_text = [i[0] for i in data]
        target_text = [i[1] for i in data]

        self.logger.info("Encoding Inputs...")
        encoder_inputs = self.tokenizer(
            input_text, truncation=True,
            return_tensors="tf", max_length=encoder_max_len,
            pad_to_max_length=True
        )

        self.logger.info("Decoding Inputs...")
        decoder_inputs = self.tokenizer(
            target_text, truncation=True,
            return_tensors="tf", max_length=decoder_max_len,
            pad_to_max_length=True
        )

        outputs = []

        for i, _ in enumerate(encoder_inputs["input_ids"]):
            outputs.append(
                {
                    "input_ids":encoder_inputs["input_ids"][i],
                    "attention_mask": encoder_inputs["attention_mask"][i],
                    "labels":decoder_inputs["input_ids"][i],
                    "decoder_attention_mask": decoder_inputs["attention_mask"][i]
                }
            )

        return outputs

    def to_tf_dataset(self, dataset):
        return_types = {
            "input_ids":tf.int32,
            "attention_mask":tf.int32,
            "labels":tf.int32,
            "decoder_attention_mask":tf.int32
        }
        return_shapes = {
            "input_ids": tf.TensorShape([None]),
            "attention_mask": tf.TensorShape([None]),
            "labels": tf.TensorShape([None]),
            "decoder_attention_mask":tf.TensorShape([None])
        }
        ds = tf.data.Dataset.from_generator(lambda : dataset, return_types, return_shapes)
        return ds

    def create_dataset(self, dataset, cache_path=None, batch_size=4,
                        buffer_size=1000, shuffle=True):

        self.logger.info("Creating dataset")

        train_dataset = map(self.encode, dataset)
        train_dataset = self.to_tf_dataset(train_dataset)
        if cache_path:
            train_dataset = train_dataset.cache(cache_path)
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
        return train_dataset

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=7056):
        super().__init__()
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        m = tf.maximum(self.warmup_steps, step)
        m = tf.cast(m, tf.float32)
        lr = tf.math.rsqrt(m)

        return lr
