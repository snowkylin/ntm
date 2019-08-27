import tensorflow as tf
import numpy as np
import json
from model_v2 import CopyModel
from utils import generate_random_strings


class SequenceCrossEntropyLoss(tf.keras.losses.Loss):
    eps = 1e-8

    def call(self, y_true, y_pred):
        return -tf.reduce_mean(   # cross entropy function
            y_true * tf.math.log(y_pred + self.eps) + (1 - y_true) * tf.math.log(1 - y_pred + self.eps)
        )


def train(config):
    model = CopyModel(
        batch_size=config['batch_size'],
        vector_dim=config['vector_dim'],
        model_type=config['model_type'],
        cell_params=config['cell_params'][config['model_type']]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    sequence_loss_func = SequenceCrossEntropyLoss()
    for batch_index in range(config['num_batches']):
        seq_length = tf.constant(np.random.randint(1, config['max_seq_length'] + 1), dtype=tf.int32)
        x = generate_random_strings(config['batch_size'], seq_length, config['vector_dim'])
        with tf.GradientTape() as tape:
            y_pred = model((x, seq_length))
            loss = sequence_loss_func(y_true=x, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if batch_index % 100 == 0:
            x = generate_random_strings(config['batch_size'], config['test_seq_length'], config['vector_dim'])
            y_pred = model((x, config['test_seq_length']))
            loss = sequence_loss_func(y_true=x, y_pred=y_pred)
            print("batch %d: loss %f" % (batch_index, loss))
            print("original string sample: ", x[0])
            print("copied string sample: ", y_pred[0])


if __name__ == '__main__':
    with open('copy_task_config.json') as f:
        config = json.load(f)
    print(config)
    if config['mode'] == 'train':
        train(config)
