import random
from datetime import datetime
from typing import Tuple, Dict, List

import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BertTokenizerFast
from transformers import PreTrainedTokenizerBase
from sklearn.metrics import classification_report

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


def parse_test_file(root_path: str) -> Tuple[list, list, list]:
    """
    Given the path to a test file, this function parses the file, and returns three lists: sentences, category labels,
    and polarity labels.
    """
    test_sentences = []
    test_categories = []
    test_polarities = []

    with open(f'{root_path}/test.txt', 'r') as f:
        for line in f:  # Read in each line. Parse the category and polarity. Add values to lists.
            _, category, polarity, sentence = line.strip().split('\t')
            category = int(category)
            polarity = int(polarity)
            test_categories.append(category)
            test_polarities.append(polarity)
            test_sentences.append(sentence)

    return test_sentences, test_categories, test_polarities


def load_test_dataset(root_path: str, tokenizer: PreTrainedTokenizerBase) -> tf.data.Dataset:
    test_sentences, test_categories, test_polarities = parse_test_file(root_path)
    # TODO - the paper doesn't truncate test sequences. Here I am truncating at 128. Does this make a difference?
    input_ids, token_type_ids, attention_mask = tokenize_sentences(test_sentences, tokenizer)

    # Convert test labels to tensors
    test_categories = tf.convert_to_tensor(test_categories)
    test_polarities = tf.convert_to_tensor(test_polarities)

    dataset = create_dataset(input_ids, token_type_ids, attention_mask, test_polarities, test_categories)

    return dataset


def tokenize_sentences(sentences: List[str], tokenizer: PreTrainedTokenizerBase) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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
    # input_ids (batch size, 128). Integer encoded sentences with padding on the end.
    # token_type_ids (batch size, 128). This appears to be used when you need to pass into a downstream model multiple
    # sentences. E.g. question answering. You want to pass in two sentences (say a context sentence. and a question).
    # And you want to indicate to the model which sentence is which.
    # attention_mask (batch_size, 128). This is 1s for the real tokens, and 0s for the padded tokens. This tells the
    # downstream model which tokens to 'attend' to, and which it can ignore.
    input_ids = encoded_dict['input_ids']
    token_type_ids = encoded_dict['token_type_ids']
    attention_mask = encoded_dict['attention_mask']

    tf.cast(attention_mask, dtype=tf.float32)

    return input_ids, token_type_ids, attention_mask


def create_dataset(input_ids: tf.Tensor, token_type_ids: tf.Tensor, attention_mask: tf.Tensor, labels_pol: tf.Tensor,
                   labels_cat: tf.Tensor) -> tf.data.Dataset:
    labels_cat_dataset = tf.data.Dataset.from_tensor_slices(labels_cat, name='labels_cat')
    labels_pol_dataset = tf.data.Dataset.from_tensor_slices(labels_pol, name='labels_pol')
    input_ids_dataset = tf.data.Dataset.from_tensor_slices(input_ids, name='input_ids')
    token_type_ids_dataset = tf.data.Dataset.from_tensor_slices(token_type_ids, name='token_type_ids')
    attention_mask_dataset = tf.data.Dataset.from_tensor_slices(attention_mask, name='attention_mask')

    dataset = tf.data.Dataset.zip(
        (input_ids_dataset, token_type_ids_dataset, attention_mask_dataset, labels_pol_dataset, labels_cat_dataset))
    dataset = dataset.shuffle(1000000, seed=0)
    dataset.batch(32)

    return dataset


def split_dataset(dataset: tf.data.Dataset, split_perc: float) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Given a dataset and a percentage, splits the dataset into two. The first dataset returned has split_perc of the
    data. The dataset will be shuffled before splitting.
    """
    n = len(dataset)
    shuffled_dataset = dataset.shuffle(buffer_size=n, seed=0)
    n_first = int(n * split_perc)
    first_dataset = shuffled_dataset.take(n_first)
    second_dataset = shuffled_dataset.skip(n_first)

    return first_dataset, second_dataset


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

    def load_training_data(self): # TODO - probably rename this to something like 'load_labeled_training_data'. Make it clear that this is the unsupervised labeled data.
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

    def train_model(self, dataset: tf.data.Dataset, epochs=epochs):
        # TODO - factor out functions for train_step and validate_step, test_step https://www.tensorflow.org/tensorboard/get_started
        validation_dataset, train_dataset = split_dataset(dataset, 0.10)
        model = self.model
        self.set_seed(0)

        # Set up TensorBoard
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        loss_fn = LQLoss()

        # Define our metrics
        train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        validation_loss_metric = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)

        for epoch in trange(epochs):
            print("Start of epoch %d" % (epoch,))

            # Iterate over batches of the training dataset
            for input_ids, token_type_ids, attention_mask, labels_pol, labels_cat in tqdm(train_dataset.batch(32)):
                with tf.GradientTape() as tape:
                    preds_cat, preds_pol = model(input_ids, token_type_ids, attention_mask)
                    train_loss = loss_fn.call(preds_cat, labels_cat, loss_fn.aspect_weights) + \
                                 loss_fn.call(preds_pol, labels_pol, loss_fn.sentiment_weights)

                grads = tape.gradient(target=train_loss, sources=model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric
                train_loss_metric(train_loss)

            # Display metrics at the end of each epoch
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss_metric.result(), step=epoch)

            # Run an evaluation loop at the end of each epoch
            for val_input_ids, val_token_type_ids, val_attention_mask, val_labels_pol, val_labels_cat in validation_dataset.batch(
                    32):
                val_preds_cat, val_preds_pol = model(val_input_ids, val_token_type_ids, val_attention_mask)
                val_loss = loss_fn.call(val_preds_cat, val_labels_cat, loss_fn.aspect_weights) + \
                           loss_fn.call(val_preds_pol, val_labels_pol, loss_fn.sentiment_weights)

                # Update validation metric
                validation_loss_metric(val_loss)

            # Display validation metrics
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', validation_loss_metric.result(), step=epoch)

            # Reset metrics every epoch
            train_loss_metric.reset_states()
            validation_loss_metric.reset_states()
            # train_cat_precision_metric.reset_states()

        # Save model
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M")
        self.save_model(dt_string + '-model')

        # Evaluate on the test set
        n_categories = 3  # TODO - remove hardcodings
        n_polarities = 2  # TODO - remove hardcodings

        test_predicted_category_probabilities = tf.zeros((0, n_categories), dtype=tf.float32)
        test_predicted_polarity_probabilities = tf.zeros((0, n_polarities), dtype=tf.float32)
        test_actual_category_classes = tf.zeros((0, 1), dtype=tf.int32)
        test_actual_polarity_classes = tf.zeros((0, 1), dtype=tf.int32)

        test_dataset = load_test_dataset(self.root_path, self.tokenizer)
        for test_input_ids_batch, test_token_type_ids_batch, test_attention_mask_batch, test_labels_pol_batch, test_labels_cat_batch in test_dataset.batch(32):
            test_preds_cat_batch, test_preds_pol_batch = model(test_input_ids_batch, test_token_type_ids_batch, test_attention_mask_batch)
            test_loss_batch = loss_fn.call(test_preds_cat_batch, test_labels_cat_batch, loss_fn.aspect_weights) + \
                              loss_fn.call(test_preds_pol_batch, test_labels_pol_batch, loss_fn.sentiment_weights)

            test_predicted_category_probabilities = tf.concat([test_predicted_category_probabilities, test_preds_cat_batch], 0)
            test_predicted_polarity_probabilities = tf.concat([test_predicted_polarity_probabilities, test_preds_pol_batch], 0)
            test_actual_category_classes = tf.concat([test_actual_category_classes, tf.expand_dims(test_labels_cat_batch, 1)], 0)
            test_actual_polarity_classes = tf.concat([test_actual_polarity_classes, tf.expand_dims(test_labels_pol_batch, 1)], 0)

        # Convert predicted probabilities to hard classes
        test_predicted_category_classes = tf.math.argmax(test_predicted_category_probabilities, axis=1)
        test_predicted_polarity_classes = tf.math.argmax(test_predicted_polarity_probabilities, axis=1)

        # Convert to numpy
        test_predicted_polarity_classes_np = test_predicted_polarity_classes.numpy()
        test_predicted_category_classes_np = test_predicted_category_classes.numpy()
        test_actual_polarity_classes_np = test_actual_polarity_classes.numpy()
        test_actual_category_classes_np = test_actual_category_classes.numpy()

        print("Category classification report")
        print(classification_report(test_actual_category_classes_np, test_predicted_category_classes_np))

        print("Polarity classification report")
        print(classification_report(test_actual_polarity_classes_np, test_predicted_polarity_classes_np))


    def save_model(self, name):
        self.model.save(f'{self.root_path}/{name}.tf')

    def load_model(self, name):
        self.model = tf.keras.models.load_model(f'{self.root_path}/{name}.tf')

# TODO - 1 - Look into logging more metrics like accuracy and F1, precision, recall
# TODO - 2 - clean up code. clean up comments. and refactor as much as you can
#            TODO - do the authors fine-tune BERT, or just the top layers?
# TODO - 4 - clean up console warnings
# TODO - 5 - Get model working with financial dataset
# TODO - someday maybe - plug in FinBert
# TODO - someday maybe - get a larger BERT model, or a newer transformer architecture
# TODO - someday maybe - checkpoint your model
# TODO - someday maybe - look into classification metrics for imbalanced datasets.
