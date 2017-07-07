import tensorflow as tf
import numpy as np

class NTM_model():
    def __init__(self, args, seq_length, reuse=False):
        self.x = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, seq_length, args.vector_dim])
        self.y = self.x
        eof = np.zeros([args.batch_size, args.vector_dim + 1])
        eof[:, args.vector_dim] = np.ones([args.batch_size])
        eof = tf.constant(eof, dtype=tf.float32)
        zero = tf.constant(np.zeros([args.batch_size, args.vector_dim + 1]), dtype=tf.float32)

        if args.model == 'LSTM':
            #single_cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_size)
            # cannot use [single_cell] * 3 in tensorflow 1.2
            def rnn_cell(rnn_size):
                return tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(args.rnn_size) for _ in range(args.rnn_num_layers)])
        elif args.model == 'DNC_DEEPMIND':
            import dnc_deepmind.dnc as dnc
            access_config = {
                "memory_size": 16,
                "word_size": 16,
                "num_reads": 4,
                "num_writes": 1,
            }
            controller_config = {
                "hidden_size": args.rnn_size,
            }
            cell = dnc.DNC(access_config, controller_config, args.vector_dim + 1, 20)
        elif args.model == 'NTM':
            import ntm.ntm_cell as ntm_cell
            cell = ntm_cell.NTMCell(args.rnn_size, args.memory_size, args.memory_vector_dim, 1, 1,
                                    addressing_mode='content_and_location', reuse=reuse)

        state = cell.zero_state(args.batch_size, tf.float32)
        for t in range(seq_length):
            output, state = cell(tf.concat([self.x[:, t, :], np.zeros([args.batch_size, 1])], axis=1), state)
        output, state = cell(eof, state)

        self.o = []
        for t in range(seq_length):
            output, state = cell(zero, state)
            self.o.append(output[:, 0:args.vector_dim])
        self.o = tf.sigmoid(tf.transpose(self.o, perm=[1, 0, 2]))

        # self.copy_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.o), reduction_indices=[1, 2]))
        eps = 1e-5
        self.copy_loss = -tf.reduce_mean(
            self.y * tf.log(self.o + eps) + (1 - self.y) * tf.log(1 - self.o + eps)
        )   # cross entropy function
        with tf.variable_scope('optimizer', reuse=reuse):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            self.train_op = self.optimizer.minimize(self.copy_loss)
