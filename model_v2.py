import tensorflow as tf


class CopyModel(tf.keras.Model):
    def __init__(self, batch_size, vector_dim, model_type, cell_params):
        super().__init__()
        self.batch_size = batch_size
        self.vector_dim = vector_dim

        self.eof = tf.one_hot([self.vector_dim] * batch_size, depth=self.vector_dim+1)
        self.zero = tf.zeros([batch_size, vector_dim + 1], dtype=tf.float32)
        self.model_type = model_type
        self.cell_params = cell_params

        if self.model_type == 'LSTM':
            self.cell = tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.LSTMCell(units=self.cell_params['rnn_size']) for _ in range(self.cell_params['rnn_num_layers'])])
        elif self.model_type == 'NTM':
            from ntm.ntm_cell_v2 import NTMCell
            self.cell = NTMCell(rnn_size=self.cell_params['rnn_size'],
                                memory_size=self.cell_params['memory_size'],
                                memory_vector_dim=self.cell_params['memory_vector_dim'],
                                read_head_num=self.cell_params['read_head_num'],
                                write_head_num=self.cell_params['write_head_num'],
                                addressing_mode='content_and_location',
                                output_dim=self.vector_dim)
        else:
            raise ValueError('Model type not supported')

    @tf.function
    def call(self, inputs):
        x, seq_length = inputs

        x_list = tf.TensorArray(dtype=tf.float32, size=seq_length)
        x_list = x_list.unstack(tf.transpose(x, perm=[1, 0, 2]))
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        for t in range(seq_length):
            output, state = self.cell(tf.concat([x_list.read(t), tf.zeros([self.batch_size, 1])], axis=1), state)

        output, state = self.cell(self.eof, state)

        output_list = tf.TensorArray(dtype=tf.float32, size=seq_length)
        for t in range(seq_length):
            output, state = self.cell(self.zero, state)
            output_list = output_list.write(t, output[:, 0:self.vector_dim])
        y_pred = tf.sigmoid(tf.transpose(output_list.stack(), perm=[1, 0, 2]))

        return y_pred

