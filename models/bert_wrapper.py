from models.bert import BertModel, BertConfig
import tensorflow as tf
from models.utils import add_train_op
from models.model import BaseModel


class BertSetting:
    batch_size = 32
    title_len = 30
    detail_len = 100
    class_num = 25551
    fc_hidden_dim = 1024
    lr = 0.001
    max_epoch = 50
    max_seq_length = 128
    train_data_size = 649447
    dev_data_size = 72161
    test_data_size = 8946


class BertWrapper(BaseModel):

    def __init__(self):
        super(BertWrapper, self).__init__('BERT')
        self.settings = BertSetting()

        with tf.name_scope('Inputs'):
            self.input_ids = tf.placeholder(tf.int32, [None, self.settings.max_seq_length], name='input_ids')
            self.input_mask = tf.placeholder(tf.int32, [None, self.settings.max_seq_length], name='input_mask')
            self.segment_ids = tf.placeholder(tf.int32, [None, self.settings.max_seq_length], name='segment_ids')
            self.labels = tf.placeholder(tf.float32, [None, self.settings.class_num], name='labels')
        bert_config = BertConfig.from_json_file('chinese_L-12_H-768_A-12/bert_config.json')

        self.model = BertModel(
            config=bert_config,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        output_layer = self.model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value
        output_weights = tf.get_variable(
            "output_weights", [self.settings.class_num, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [self.settings.class_num], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            output_layer = tf.nn.dropout(output_layer, keep_prob=1 - self.model.hidden_dropout_prob)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            self.sigmoid_y_pred = tf.nn.sigmoid(logits)
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits)
            self.loss = tf.reduce_mean(per_example_loss)

        with tf.variable_scope('training_ops'):
            self.train_op = add_train_op(lr=self.settings.lr, loss=self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=1, name=self.model_name)

    def create_feed_dic(self, batch_data, is_training=True):
        max_seq_length = self.settings.max_seq_length

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_labels = batch_data['class_input']
        for tokens_a, tokens_b in zip(batch_data['title_input'], batch_data['detail_input']):
            tokens_a = list(tokens_a)
            tokens_b = list(tokens_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            input_ids = []
            segment_ids = []
            input_ids.append(102)  # [CLS]
            segment_ids.append(0)
            for token in tokens_a:
                input_ids.append(token)
                segment_ids.append(0)
            input_ids.append(103)  # [SEP]
            segment_ids.append(0)
            for token in tokens_b:
                input_ids.append(token)
                segment_ids.append(1)
            input_ids.append(103)  # [SEP]
            segment_ids.append(1)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)

        return {
            self.input_ids: all_input_ids,
            self.input_mask: all_input_mask,
            self.segment_ids: all_segment_ids,
            self.labels: all_labels,
            self.model.hidden_dropout_prob: self.model.config.hidden_dropout_prob if is_training else 0,
            self.model.attention_probs_dropout_prob: self.model.config.attention_probs_dropout_prob if is_training else 0
        }


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
