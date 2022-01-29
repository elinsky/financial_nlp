import tensorflow as tf
from transformers import TFBertModel


class BERTLinearTF(tf.keras.Model):
    def __init__(self, bert_type: str, num_cat: int, num_pol: int, **kwargs):
        super(BERTLinearTF, self).__init__(name='BERTLinearTF')
        self.bert: TFBertModel = TFBertModel.from_pretrained(bert_type, output_hidden_states=True)
        self.ff_cat = tf.keras.layers.Dense(units=num_cat, activation='relu', use_bias=True)
        self.ff_pol = tf.keras.layers.Dense(units=num_pol, activation='relu', use_bias=True)

    def call(self, input_ids: tf.Tensor, token_type_ids: tf.Tensor, attention_mask: tf.Tensor):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        # TODO - run an experiment comparing the last hidden layer to second to last. https://github.com/hanxiao/bert-as-service#q-why-not-the-last-hidden-layer-why-second-to-last
        # TODO - also test using the pooled output. Or also concatenating a hidden states from the last 4 layers.
        x = outputs.last_hidden_state  # shape of x is (32, 128, 768)

        attention_mask = tf.expand_dims(attention_mask, 2)  # Turns attention mask from (32, 128) to (32, 128, 1).
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)

        se = x * attention_mask  # se is (32, 128, 768)
        den = tf.math.reduce_sum(attention_mask, axis=1)  # (batch size, 1)
        se = tf.math.reduce_sum(se, axis=1) / den  # (batch size, 768)

        logits_cat = self.ff_cat(se)  # (batch size, num_cat)
        logits_pol = self.ff_pol(se)  # (batch size, num_pol)

        preds_cat = tf.nn.softmax(logits_cat)
        preds_pol = tf.nn.softmax(logits_pol)

        return preds_cat, preds_pol


class LQLoss(tf.keras.Model):
    def __init__(self, q=0.4, alpha=0.0):
        # should I be subclassing keras.losses.Loss instead?
        # And do I need to subclass anything, or can I just create a normal class?
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

    def call(self, preds_cat: tf.Tensor, labels_cat: tf.Tensor, preds_pol: tf.Tensor, labels_pol: tf.Tensor):
        # preds_cat is a tensor of shape (bsz, 3). Each row has the probability of each class
        # labels_cat is a tensor of shape (bsz, 1)
        bsz = len(preds_cat)
        n_pols = preds_pol.shape[1]
        n_cats = preds_cat.shape[1]

        labels_cat = tf.expand_dims(labels_cat, axis=1)  # change shape from (32,) to (32, 1)
        labels_pol = tf.expand_dims(labels_pol, axis=1)

        # https://stackoverflow.com/questions/52129909/tensorflow-equivalent-of-torch-gather/52642327
        idx_cat = tf.stack([tf.range(tf.shape(labels_cat)[0]), labels_cat[:, 0]], axis=-1)
        idx_pol = tf.stack([tf.range(tf.shape(labels_pol)[0]), labels_pol[:, 0]], axis=-1)

        cat_pred_prob = tf.gather_nd(preds_cat, idx_cat)  # Get model's probability for the correct category class
        pol_pred_prob = tf.gather_nd(preds_pol, idx_pol)  # Get model's probability for the correct polarity class

        lq_cat = (1 - tf.pow(cat_pred_prob, self.q)) / self.q
        lq_pol = (1 - tf.pow(pol_pred_prob, self.q)) / self.q

        bsz_t = tf.constant([bsz])
        sentiment_weights_bsz = tf.reshape(tf.tile(self.sentiment_weights, bsz_t),
                                           (bsz, n_pols))  # Reshape into (32, 2)
        aspect_weights_bsz = tf.reshape(tf.tile(self.aspect_weights, bsz_t), (bsz, n_cats))  # Reshape into (32, 3)

        sentiment_weights_bsz = tf.gather_nd(sentiment_weights_bsz,
                                             idx_pol)  # get weight for the label. New shape is (32, 1)
        aspect_weights_bsz = tf.gather_nd(aspect_weights_bsz, idx_pol)  # get weight for the label. New shape is (32, 1)

        cat_loss = tf.math.reduce_mean(self.alpha * lq_cat + (
                1 - self.alpha) * lq_cat * aspect_weights_bsz)  # aspect weights should be shape (32, 1). For some reason mine are (32, 3).
        pol_loss = tf.math.reduce_mean(self.alpha * lq_pol + (1 - self.alpha) * lq_pol * sentiment_weights_bsz)

        return cat_loss + pol_loss


if __name__ == '__main__':
    pass
