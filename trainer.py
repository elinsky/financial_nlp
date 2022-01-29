import random
from typing import Tuple, Dict, List

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, BertTokenizerFast
from transformers import PreTrainedTokenizerBase

from config import *
from model import BERTLinearTF, LQLoss


def create_categorical_encoder_and_decoder(categorical_variables: list[str]) -> Tuple[dict, dict]:
    """
    Given a list of categorical variables, returns an encoder and decoder.
    Encoder Key = category. Value = integer encoding.
    Decoder Key = integer encoding. Value = category.
    """
    decoder = {}
    encoder = {}
    for idx, variable in enumerate(categorical_variables):
        decoder[idx] = variable
        encoder[variable] = idx
    return decoder, encoder


def parse_labels_file(root_path: str, category_encoder: dict, polarity_encoder: dict) -> Tuple[list, list, list]:
    """
    Encoders are dictionaries. Key = category/polarity string. value is integer encoding.
    Parses the labeled dataset.

    Returns a tuple of lists. The lists are of
    sentences, category label for that sentence, polarity label for that sentence.
    """
    sentences = []
    categories = []
    polarities = []

    with open(f'{root_path}/label.txt', 'r') as f:  # Loop through the labeled data.
        for idx, line in enumerate(f):
            if idx % 2 == 1:  # If the line is a label
                cat, pol = line.strip().split()  # Get the labeled category and polarity for the previous sentence
                categories.append(category_encoder[cat])  # Add label to categories list
                polarities.append(polarity_encoder[pol])  # Add label to polarities list
            else:  # If the line is a sentence
                sentences.append(line.strip())
    return sentences, categories, polarities


def tokenize_sentences(sentences: List[str], tokenizer: PreTrainedTokenizerBase) -> Tuple[
    tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Given a list of sentences and a tokenizer, this tokenizes the sentences
    and returns a tuple of input_ids, token_type_ids, and attention_mask as
    tensors.
    """
    encoded_dict = tokenizer(
        sentences,
        padding=True,  # Padding = True will pad to the longest sequence in the batch.
        return_tensors='tf',
        max_length=128,  # We truncate to 128 tokens.
        return_attention_mask=True,
        truncation=True)

    # Unpack encoded dict
    input_ids = encoded_dict['input_ids']  # (batch size, 128). Integer encoded sentences with padding on the end.
    token_type_ids = encoded_dict[
        'token_type_ids']  # (batch size, 128). This appears to be used when you need to pass into a downstream model multiple sentences. E.g. question answering. You want to pass in two sentences (say a context sentence. and a question). And you want to indicate to the model which sentence is which.
    attention_mask = encoded_dict[
        'attention_mask']  # (batch_size, 128). This is 1s for the real tokens, and 0s for the padded tokens. This tells the downstream model which tokens to 'attend' to, and which it can ignore.

    tf.cast(attention_mask, dtype=tf.float32)

    return input_ids, token_type_ids, attention_mask


def create_dataset(input_ids, token_type_ids, attention_mask, labels_pol, labels_cat):
    # Create dataset
    labels_cat_dataset = tf.data.Dataset.from_tensor_slices(labels_cat, name='labels_cat')
    labels_pol_dataset = tf.data.Dataset.from_tensor_slices(labels_pol, name='labels_pol')
    input_ids_dataset = tf.data.Dataset.from_tensor_slices(input_ids, name='input_ids')
    token_type_ids_dataset = tf.data.Dataset.from_tensor_slices(token_type_ids, name='token_type_ids')
    attention_mask_dataset = tf.data.Dataset.from_tensor_slices(attention_mask, name='attention_mask')

    dataset = tf.data.Dataset.zip(
        (input_ids_dataset, token_type_ids_dataset, attention_mask_dataset, labels_pol_dataset, labels_cat_dataset))

    # TODO - clean all this up
    dataset.batch(32)

    # Shuffle
    dataset = dataset.shuffle(1000000, seed=0)

    n = len(dataset)
    n_val = int(n * 0.10)
    validation_set = dataset.take(n_val)
    train_set = dataset.skip(n_val)

    validation_set.batch(batch_size)
    train_set.batch(batch_size)

    return dataset


class Trainer:

    def __init__(self):
        self.domain: str = config['domain']
        self.bert_type: str = bert_mapper[self.domain]
        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path: str = path_mapper[self.domain]
        self.categories: list[str] = aspect_category_mapper[self.domain]
        self.polarities: list[str] = sentiment_category_mapper[self.domain]
        self.model: BERTLinearTF = BERTLinearTF(self.bert_type, len(self.categories), len(self.polarities))

        self.polarity_decoder: Dict[int, str]
        self.polarity_encoder: Dict[str, int]
        self.polarity_decoder, self.polarity_encoder = create_categorical_encoder_and_decoder(self.polarities)

        self.aspect_decoder: Dict[int, str]
        self.aspect_encoder: Dict[str, int]
        self.aspect_decoder, self.aspect_encoder = create_categorical_encoder_and_decoder(self.categories)

    def load_training_data(self):
        sentences, cats, pols = parse_labels_file(self.root_path, self.aspect_encoder, self.polarity_encoder)
        labels_cat, labels_pol = tf.convert_to_tensor(cats), tf.convert_to_tensor(pols)
        input_ids, token_type_ids, attention_mask = tokenize_sentences(sentences, self.tokenizer)
        dataset = create_dataset(input_ids, token_type_ids, attention_mask, labels_pol, labels_cat)

        return dataset

        # TODO - the authors have a max length of 128. So no sentences can be longer than that. Do I want to change this parameter?
        # TODO - can I get rid of token_type_ids? I don't think I need to pass it in to the model.
        # TODO - shouldn't I include position_ids as something I pass in to the downstream model? Nevermind. It looks like if you don't pass in position_ids, then the model will automatically create them for you.

    def set_seed(self, value: int):
        random.seed(value)
        np.random.seed(value)
        tf.random.set_seed(value)

    # @tf.function TODO - consider getting this to work. This provides a performance speed up.
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

                # Open a GradientTape to record the operations run during the
                # forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:
                    # Run the forward pass through the model.
                    # The operations that the model applies to its inputs are
                    # going to be recorded on the GradientTape.
                    preds_cat, preds_pol = model(input_ids, token_type_ids, attention_mask)

                    # Compute the loss value for this mini-batch.
                    loss = loss_fn.call(preds_cat, labels_cat, preds_pol, labels_pol)
                    print('loss: ', loss)

                    # TODO - 1 - get Tensorboard working
                    # TODO - 1.5 - Look into logging more metrics like accuracy and F1
                    # TODO - 2.5 - get model checkpointing and model saving working
                    # TODO - 3 - clean up code. clean up comments
                    # TODO - 4 - Compare results with your code to researchers. Make sure metrics match or are close.
                    # TODO - 5 - Get model working with financial dataset
                    # TODO - someday maybe - plug in FinBert
                    # TODO - someday maybe - get a larger BERT model, or a newer transformer architecture

                # Use the gradient tape to automatically retrieve the gradients
                # of the trainable variables with respect to the loss.
                # grads is a list of Tensors to be differentiated.
                # It includes the weights in BERT, and the weights from my layers on top
                grads = tape.gradient(target=loss, sources=model.trainable_weights)

                # Run one step of gradient descent by updating the value of the
                # variables to minimize the loss
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
