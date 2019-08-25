import tensorflow as tf
import numpy as np


class NTMCopyModel(tf.keras.Model):
    def __init__(self, batch_size, vector_dim, model_type, cell_params):
        super().__init__()
        eof = np.zeros([batch_size, vector_dim + 1])
        eof[:, vector_dim] = np.ones([batch_size])
        eof = tf.constant(eof, dtype=tf.float32)
        zero = tf.constant(np.zeros([batch_size, vector_dim + 1]), dtype=tf.float32)

        if model_type == 'LSTM':
            self.cell = tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.LSTMCell(units=cell_params['rnn_size']) for _ in range(cell_params['rnn_num_layers'])])
        elif model_type == 'NTM':
            import ntm.ntm_cell_v2 as ntm_cell
            self.cell = ntm_cell.NTMCell(rnn_size=cell_params['rnn_size'],
                                         memory_size=cell_params['memory_size'],
                                         memory_vector_dim=cell_params['memory_vector_dim'],
                                         read_head_num=1,
                                         write_head_num=1,
                                         addressing_mode='content_and_location',
                                         output_dim=vector_dim)
        else:
            raise ValueError('Model type not supported')

        state = self.cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)


    def call(self, inputs):

