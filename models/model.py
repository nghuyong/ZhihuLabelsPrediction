import tensorflow as tf


class BaseModel(object):
    max_f1 = 0.0
    global_step = tf.Variable(0, trainable=False, name='global_step')

    def __init__(self, model_name):
        self.model_name = model_name
        tf.gfile.MakeDirs(f'logs/{self.model_name}/train')
        tf.gfile.MakeDirs(f'logs/{self.model_name}/dev')
        self.train_writer = tf.summary.FileWriter(f'logs/{self.model_name}/train')
        self.dev_writer = tf.summary.FileWriter(f'logs/{self.model_name}/dev')

    def create_feed_dic(self, batch_data):
        raise NotImplementedError()
