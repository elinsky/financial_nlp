from typing import Tuple, Dict

from transformers import AutoTokenizer, BertForMaskedLM, BertTokenizerFast
from transformers import BertTokenizer, TFBertModel
import tensorflow_datasets as tfds

import tensorflow as tf
from config import *
from filter_words import filter_words
from tqdm import tqdm, trange
from model import BERTLinearTF, LQLoss
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def _init_dict(items: list) -> Tuple[dict, dict]:
    lookup = {}
    inv_lookup = {}
    for idx, item in enumerate(items):
        lookup[idx] = item
        inv_lookup[item] = idx
    return lookup, inv_lookup


class Trainer:

    def __init__(self):
        self.domain: str = config['domain']
        self.bert_type: str = bert_mapper[self.domain]
        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path: str = path_mapper[self.domain]
        self.categories: list[str] = aspect_category_mapper[self.domain]
        self.polarities: list[str] = sentiment_category_mapper[self.domain]
        self.model: BERTLinearTF = BERTLinearTF(self.bert_type, len(self.categories), len(self.polarities))

        self.polarity_dict: Dict[int, str]
        self.inv_polarity_dict: Dict[str, int]
        self.polarity_dict, self.inv_polarity_dict = _init_dict(self.polarities)

        self.aspect_dict: Dict[int, str]
        self.inv_aspect_dict: Dict[str, int]
        self.aspect_dict, self.inv_aspect_dict = _init_dict(self.categories)

    def load_training_data(self):
        sentences = []
        cats = []
        pols = []
        with open(f'{self.root_path}/label.txt', 'r') as f:  # Loop through the labeled data.
            for idx, line in enumerate(f):
                if idx % 2 == 1:  # If the line is a label
                    cat, pol = line.strip().split()  # Get the labeled category and polarity for the previous sentence
                    cats.append(self.inv_aspect_dict[cat])  # Add label to categories list
                    pols.append(self.inv_polarity_dict[pol])  # Add label to polarities list
                else:  # If the line is a sentence
                    sentences.append(line.strip())
        encoded_dict = self.tokenizer(  # Pass sentence into the tokenizer.
            sentences,  # This encoded_dict dictionary has 3 keys: input_ids, token_type_ids, and attention_mask
            padding=True,
            return_tensors='tf',
            max_length=128,
            return_attention_mask=True,
            truncation=True)  # So in this step, we basically take all the sentences in the dataset and tokenize them.
        # Padding = True will pad to the longest sequence in the batch.
        # We truncate to 128 tokens.
        # Return attention mask / attention_mask - So this is a tensor of shape (batch_size, 128). This is 1s for the real tokens, and 0s for the padded tokens. This tells the downstream model which tokens to 'attend' to, and which it can ignore.
        # token_type_ids - one of the return values. Shape is (batch size, 128). This appears to be used when you need to pass into a downstream model multiple sentences. E.g. question answering. You want to pass in two sentences (say a context sentence. and a question). And you want to indicate to the model which sentence is which. I'm not sure I need this.
        # input_ids - one of the return values. Shape is (batch size, 128). So basically these are the tokenized sentences with padding on the end.
        labels_cat = tf.convert_to_tensor(cats)
        labels_pol = tf.convert_to_tensor(pols)
        input_ids = encoded_dict['input_ids']
        token_type_ids = encoded_dict['token_type_ids']
        attention_mask = encoded_dict['attention_mask']
        tf.cast(attention_mask, dtype=tf.float32)

        labels_cat_dataset = tf.data.Dataset.from_tensor_slices(labels_cat, name='labels_cat')
        labels_pol_dataset = tf.data.Dataset.from_tensor_slices(labels_pol, name='labels_pol')
        input_ids_dataset = tf.data.Dataset.from_tensor_slices(input_ids, name='input_ids')
        token_type_ids_dataset = tf.data.Dataset.from_tensor_slices(token_type_ids, name='token_type_ids')
        attention_mask_dataset = tf.data.Dataset.from_tensor_slices(attention_mask, name='attention_mask')

        dataset = tf.data.Dataset.zip(
            (input_ids_dataset, token_type_ids_dataset, attention_mask_dataset, labels_pol_dataset, labels_cat_dataset))
        dataset.batch(32)

        # Shuffle
        # dataset = dataset.shuffle(1000000, seed=0)

        # n = len(dataset)
        # n_val = int(n * 0.10)
        # validation_set = dataset.take(n_val)
        # train_set = dataset.skip(n_val)

        # validation_set.batch(batch_size)
        # train_set.batch(batch_size)

        return dataset

        # TODO - the authors have a max length of 128. So no sentences can be longer than that. Do I want to change this parameter?
        # TODO - can I get rid of token_type_ids? I don't think I need to pass it in to the model.
        # TODO - shouldn't I include position_ids as something I pass in to the downstream model? Nevermind. It looks like if you don't pass in position_ids, then the model will automatically create them for you.

    def set_seed(self, value: int):
        random.seed(value)
        np.random.seed(value)
        tf.random.set_seed(value)

    def train_model(self, dataset: tf.data.Dataset, epochs=epochs):
        model = self.model
        self.set_seed(0)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        loss_fn = LQLoss()

        for epoch in range(epochs):
            print("Start of epoch %d" % (epoch,))
            print_loss = 0
            batch_loss = 0
            cnt = 0

            for x_batch_train in dataset.batch(32):
                input_ids, token_type_ids, attention_mask, labels_pol, labels_cat = x_batch_train

                with tf.GradientTape() as tape:
                    preds_cat, preds_pol = model.call(input_ids, token_type_ids, attention_mask)
                    loss = loss_fn.call(preds_cat, labels_cat, preds_pol, labels_pol)
                    print('loss: ', loss)

                    # TODO 2022-01-22 - Get training to work with actual backpropagation.
                    # TODO 2022-01-28 - Let's ignore the loss function for now, let's get other stuff working.
                    # TODO - get Tensorboard working


if __name__ == '__main__':
    pass
