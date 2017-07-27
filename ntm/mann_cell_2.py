import tensorflow as tf
import numpy as np

class MANNCell():
    def __init__(self, rnn_size, memory_size, memory_vector_dim, head_num, gamma=0.95, k_strategy='separate'):
        self.rnn_size = rnn_size
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.head_num = head_num                                    # #(read head) == #(write head)
        self.controller = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        self.step = 0
        self.gamma = gamma
        self.k_strategy = k_strategy

        self.W_key = tf.Variable(
            tf.random_uniform([self.head_num, self.controller.output_size, self.memory_vector_dim], minval=-0.1, maxval=0.1)
        )
        self.b_key = tf.Variable(
            tf.random_uniform([self.head_num, self.memory_vector_dim], minval=-0.1, maxval=0.1)
        )
        self.W_add = tf.Variable(
            tf.random_uniform([self.head_num, self.controller.output_size, self.memory_vector_dim], minval=-0.1, maxval=0.1)
        )
        self.b_add = tf.Variable(
            tf.random_uniform([self.head_num, self.memory_vector_dim], minval=-0.1, maxval=0.1)
        )
        self.W_sigma = tf.Variable(
            tf.random_uniform([self.head_num, self.controller.output_size, 1], minval=-0.1, maxval=0.1)
        )
        self.b_sigma = tf.Variable(
            tf.random_uniform([self.head_num, 1], minval=-0.1, maxval=0.1)
        )

    def __call__(self, x, prev_state):
        r_tm1 = prev_state['r']
        c_tm1 = prev_state['c']
        wu_tm1 = prev_state['wu']  # (batches, memory_size)
        wr_tm1 = prev_state['wr']  # (batches, heads, memory_size)
        M_tm1 = prev_state['M']     # (batches, memory_size, vector_dim)

        controller_input = tf.concat([x, r_tm1], axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            h_t, c_t = self.controller(controller_input, c_tm1) # (batches, controller_output_size)

        k_t = tf.tanh(tf.tensordot(h_t, self.W_key, axes=[[1], [1]]) + tf.expand_dims(self.b_key, axis=0)) # (batches, heads, vector_dims)
        a_t = tf.tanh(tf.tensordot(h_t, self.W_add, axes=[[1], [1]]) + tf.expand_dims(self.b_add, axis=0)) # (batches, heads, vector_dims)
        sigma_t = tf.sigmoid(tf.tensordot(h_t, self.W_sigma, axes=[[1], [1]]) + tf.expand_dims(self.b_sigma, axis=0)) # (batches, heads, 1)

        _, wlu_tm1_ = tf.nn.top_k(wu_tm1, k=self.memory_size)
        wlu_tm1 = tf.reduce_sum(tf.one_hot(wlu_tm1_[:, -self.head_num:], depth=self.memory_size), axis=1) # (batches, head_num, memory_size) -> (batches, memory_size)
        ww_t = sigma_t * wr_tm1 + (1. - sigma_t) * tf.expand_dims(wlu_tm1, axis=1)  # (batches, 1 -> head_num, memory_size)

        M_t_ = M_tm1 * (1. - tf.expand_dims(tf.one_hot(wlu_tm1_[:, -1], depth=self.memory_size), axis=2)) # (batches, memory_size, 1 -> memory_dim)
        M_t = M_t_ + tf.reduce_sum(tf.matmul(tf.expand_dims(ww_t, 3), tf.expand_dims(a_t, 2)), axis=1) # (batches, head_num, memory_size, vector_dim) -> (batches, memory_size, vector_dim)

        inner_product = tf.matmul(k_t, tf.transpose(M_t, [0, 2, 1])) # (batches, heads, memory_size)
        norm_product = tf.sqrt(tf.reduce_sum(tf.square(k_t), axis=2, keep_dims=True)) * \
                       tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(M_t), axis=2)), axis=1) # (batches, heads, memory_size)
        K_t = inner_product / (norm_product + 1e-6)
        wr_t = tf.nn.softmax(K_t) # (batches, heads, memory_size)
        r_t_ = tf.matmul(wr_t, M_t)  # (batches, heads, vector_dim)
        r_t = tf.reshape(r_t_, shape=[-1, self.head_num * self.memory_vector_dim])

        wu_t = self.gamma * wu_tm1 + tf.reduce_sum(wr_t, axis=1) + tf.reduce_sum(ww_t, axis=1)

        return tf.concat([h_t, r_t], axis=1), {'r': r_t, 'c': c_t, 'wu': wu_t, 'wr': wr_t, 'M': M_t}

    def zero_state(self, batch_size, dtype):
        def expand(x, dim, N):
            return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)
        def one_hot_weight_vector(shape):
            one_hot_weight_vector = np.zeros(shape)
            one_hot_weight_vector[..., 0] = 1
            return tf.constant(one_hot_weight_vector, dtype=tf.float32)

        with tf.variable_scope('init', reuse=self.reuse):
            state = {
                'c': self.controller.zero_state(batch_size, dtype),
                'r': tf.zeros([batch_size, self.head_num * self.memory_vector_dim]),
                'wr': one_hot_weight_vector([batch_size, self.head_num, self.memory_size]),
                'wu': one_hot_weight_vector([batch_size, self.memory_size]),
                'M': tf.constant(np.ones([batch_size, self.memory_size, self.memory_vector_dim]) * 1e-6, dtype=tf.float32)
            }
            return state

