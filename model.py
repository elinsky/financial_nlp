import tensorflow as tf
from transformers import TFBertModel


class BERTLinearTF(tf.keras.Model):
    def __init__(self, bert_type: str, num_cat: int, num_pol: int, **kwargs):
        super(BERTLinearTF, self).__init__(name='BERTLinearTF')
        self.bert = TFBertModel.from_pretrained(bert_type, output_hidden_states=True)
        self.ff_category = tf.keras.layers.Dense(units=num_cat, activation='relu', use_bias=True)
        self.ff_polarity = tf.keras.layers.Dense(units=num_pol, activation='relu', use_bias=True)

    def call(self, input_ids: tf.Tensor, token_type_ids: tf.Tensor, attention_mask: tf.Tensor):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        # TODO - run an experiment comparing the last hidden layer to second to last. https://github.com/hanxiao/bert-as-service#q-why-not-the-last-hidden-layer-why-second-to-last
        # TODO - also test using the pooled output. Or also concatenating a hidden states from the last 4 layers.
        x = outputs.last_hidden_state  # x is (32, 128, 768)

        attention_mask = tf.expand_dims(attention_mask, 2)  # (32, 128) -> (32, 128, 1).
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)

        se = x * attention_mask  # (batch_size, 128, 768) * (batch_size, 128, 1) -> (batch_size, 128, 768)
        den = tf.math.reduce_sum(attention_mask, axis=1)  # (batch_size, 128, 1) -> (batch size, 1)
        se = tf.math.reduce_sum(se, axis=1) / den  # (batch_size, 128, 768) -> (batch size, 768)

        category_logits = self.ff_category(se)  # (batch size, num_categories)
        polarity_logits = self.ff_polarity(se)  # (batch size, num_polarities)

        category_predictions = tf.nn.softmax(category_logits)  # (batch_size, 3)
        polarity_predictions = tf.nn.softmax(polarity_logits)  # (batch_size, 2)

        return category_predictions, polarity_predictions


class LQLoss(tf.keras.Model):
    def __init__(self, q=0.4, alpha=0.0):
        super(LQLoss, self).__init__()
        self.q = q
        self.alpha = alpha
        # TODO - I need to update these weights to reflect my dataset
        # TODO - look into how these are used. It looks like this is the class distribution.
        self.aspect_weights = tf.convert_to_tensor([345, 67, 201],
                                                   dtype=tf.float32)  # aspect category distribution (restaurant)
        self.sentiment_weights = tf.convert_to_tensor([231, 382],
                                                      dtype=tf.float32)  # sentiment category distribution (restaurant)
        self.aspect_weights = tf.constant(tf.nn.softmax(tf.math.log(1 / self.aspect_weights)))
        self.sentiment_weights = tf.constant(tf.nn.softmax(tf.math.log(1 / self.sentiment_weights)))

    def call(self, category_predictions: tf.Tensor, category_labels: tf.Tensor, polarity_predictions: tf.Tensor,
             polarity_labels: tf.Tensor):
        # preds_cat is a tensor of shape (bsz, 3). Each row has the probability of each class
        # labels_cat is a tensor of shape (bsz, 1)
        bsz = len(category_predictions)
        n_pols = polarity_predictions.shape[1]
        n_cats = category_predictions.shape[1]

        category_labels = tf.expand_dims(category_labels, axis=1)  # (batch_size,) -> (batch_size, 1)
        polarity_labels = tf.expand_dims(polarity_labels, axis=1)  # (batch_size,) -> (batch_size, 1)

        # https://stackoverflow.com/questions/52129909/tensorflow-equivalent-of-torch-gather/52642327
        idx_cat = tf.stack([tf.range(tf.shape(category_labels)[0]), category_labels[:, 0]], axis=-1)  # (batch_size, 2)
        idx_pol = tf.stack([tf.range(tf.shape(polarity_labels)[0]), polarity_labels[:, 0]], axis=-1)  # (batch_size, 2)

        # Get the model's probability for the correct category/polarity class
        cat_pred_prob = tf.gather_nd(category_predictions, idx_cat)  # (batch_size,)
        pol_pred_prob = tf.gather_nd(polarity_predictions, idx_pol)  # (batch_size,)

        lq_cat = (1 - tf.pow(cat_pred_prob, self.q)) / self.q  # (batch_size,)
        lq_pol = (1 - tf.pow(pol_pred_prob, self.q)) / self.q  # (batch_size,)

        bsz_t = tf.constant([bsz])

        # Reshape sentiment weights from (2,) -> (batch_size, 2) and aspect weights from (3,) -> (batch_size, 3)
        sentiment_weights_bsz = tf.reshape(tf.tile(self.sentiment_weights, bsz_t), (bsz, n_pols))
        aspect_weights_bsz = tf.reshape(tf.tile(self.aspect_weights, bsz_t), (bsz, n_cats))

        sentiment_weights_bsz = tf.gather_nd(sentiment_weights_bsz,
                                             idx_pol)  # get weight for the label. New shape is (32, 1)
        aspect_weights_bsz = tf.gather_nd(aspect_weights_bsz, idx_pol)  # get weight for the label. New shape is (32, 1)

        cat_loss = tf.math.reduce_mean(self.alpha * lq_cat + (1 - self.alpha) * lq_cat * aspect_weights_bsz)
        pol_loss = tf.math.reduce_mean(self.alpha * lq_pol + (1 - self.alpha) * lq_pol * sentiment_weights_bsz)

        return cat_loss + pol_loss
