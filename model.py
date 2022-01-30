import tensorflow as tf
from transformers import TFBertModel


class BERTLinearTF(tf.keras.Model):
    def __init__(self, bert_type: str, num_cat: int, num_pol: int, **kwargs):
        super(BERTLinearTF, self).__init__(name='BERTLinearTF')
        self.bert = TFBertModel.from_pretrained(bert_type, output_hidden_states=True).bert
        # In order to save the model, the bert layer needs to be a subclass of tf.keras.layers.Layer. Hence, the need to
        # use the .bert attribute, which is a huggingface 'MainLayer' object and has the @keras_serializable decorator.
        # https://github.com/huggingface/transformers/blob/0f69b924fbda6a442d721b10ece38ccfc6b67275/src/transformers/models/bert/modeling_tf_bert.py#L696
        assert isinstance(self.bert, tf.keras.layers.Layer)
        self.ff_category = tf.keras.layers.Dense(units=num_cat, activation='relu', use_bias=True)
        self.ff_polarity = tf.keras.layers.Dense(units=num_pol, activation='relu', use_bias=True)

    def call(self, input_ids: tf.Tensor, token_type_ids: tf.Tensor, attention_mask: tf.Tensor):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        # TODO - run an experiment comparing the last hidden layer to second to last:
        # https://github.com/hanxiao/bert-as-service#q-why-not-the-last-hidden-layer-why-second-to-last
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
        # TODO - read this in regards to class weighting:
        # https://www.tensorflow.org/guide/keras/train_and_evaluate#using_sample_weighting_and_class_weighting

        # aspect category distribution (restaurant)
        self.aspect_weights = tf.convert_to_tensor([345, 67, 201], dtype=tf.float32)
        # sentiment category distribution (restaurant)
        self.sentiment_weights = tf.convert_to_tensor([231, 382], dtype=tf.float32)

        self.aspect_weights = tf.constant(tf.nn.softmax(tf.math.log(1 / self.aspect_weights)))
        self.sentiment_weights = tf.constant(tf.nn.softmax(tf.math.log(1 / self.sentiment_weights)))

    def call(self, predictions: tf.Tensor, labels: tf.Tensor, class_weights: tf.Tensor):
        # predictions is a tensor of shape (bsz, n_classes). Each row has the probability of each class
        # labels is a tensor of shape (bsz, 1)
        bsz = len(predictions)
        bsz_t = tf.constant([bsz])
        n_classes = predictions.shape[1]

        labels = tf.expand_dims(labels, axis=1)  # (batch_size,) -> (batch_size, 1)

        # For each sample, get the index of the correct label
        # https://stackoverflow.com/questions/52129909/tensorflow-equivalent-of-torch-gather/52642327
        idx = tf.stack([tf.range(tf.shape(labels)[0]), labels[:, 0]], axis=-1)  # (batch_size, n_classes)

        # Get the model's probability for the correct category/polarity class
        pred_prob = tf.gather_nd(predictions, idx)  # (batch_size,)

        lq = (1 - tf.pow(pred_prob, self.q)) / self.q  # (batch_size,)

        # Reshape sentiment weights from (n_classes,) -> (batch_size, n_classes)
        class_weights_bsz = tf.reshape(tf.tile(class_weights, bsz_t), (bsz, n_classes))
        # Get the weight for each label.
        class_weights_bsz = tf.gather_nd(class_weights_bsz, idx)  # New shape is (batch_size, 1)
        # Calculate loss
        loss = tf.math.reduce_mean(self.alpha * lq + (1 - self.alpha) * lq * class_weights_bsz)

        return loss
