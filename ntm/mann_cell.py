import tensorflow as tf
import numpy as np

class MANNCell():
    def __init__(self, rnn_size, memory_size, memory_vector_dim, head_num, gamma=0.99,
                 reuse=False, output_dim=None, usage_strategy='summary'):
        self.rnn_size = rnn_size
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.head_num = head_num                                    # #(read head) == #(write head)
        self.reuse = reuse
        self.controller = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        self.step = 0
        self.output_dim = output_dim
        self.gamma = gamma
        self.usage_strategy = usage_strategy

    def __call__(self, x, prev_state):
        prev_read_vector_list = prev_state['read_vector_list']      # read vector (the content that is
                                                                    # read out, length = memory_vector_dim)
        prev_controller_state = prev_state['controller_state']      # state of controller (LSTM hidden state)

        # x + prev_read_vector -> controller (RNN) -> controller_output

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller(controller_input, prev_controller_state)

        # controller_output     -> k (dim = memory_vector_dim, compared to each vector in M)
        #                       -> alpha (scalar, combination of w_r and w_lu)

        num_parameters_per_head = self.memory_vector_dim + 1
        total_parameter_num = num_parameters_per_head * self.head_num
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            o2p_w = tf.get_variable('o2p_w', [controller_output.get_shape()[1], total_parameter_num],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            o2p_b = tf.get_variable('o2p_b', [total_parameter_num],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            parameters = tf.nn.xw_plus_b(controller_output, o2p_w, o2p_b)
        head_parameter_list = tf.split(parameters, self.head_num, axis=1)

        # k, prev_M -> w_r
        # alpha, prev_w_r, prev_w_lu -> w_w

        prev_w_r_list = prev_state['w_r_list']      # vector of weightings (blurred address) over locations
        prev_w_lu_list = prev_state['w_lu_list']    # least-used weight (usage_strategy = 'seperate')
        prev_w_u_list = prev_state['w_u_list']      # usage weight (usage_strategy = 'seperate')
        prev_M = prev_state['M']
        prev_w_u = prev_state['w_u_summary']        # usage weight (usage_strategy = 'summary')
        prev_w_lu = prev_state['w_lu_summary']      # least-used weight (usage_strategy = 'summary')
        w_r_list = []
        w_w_list = []
        w_u_list = []
        w_lu_list = []
        w_u_summary = None
        w_lu_summary = None
        k_list = []
        p_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            with tf.variable_scope('addressing_head_%d' % i):
                k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim], name='k')
                sig_alpha = tf.sigmoid(head_parameter[:, -1:], name='sig_alpha')
                w_r = self.read_head_addressing(k, prev_M)
                if self.usage_strategy == 'separate':
                    w_w = self.write_head_addressing(sig_alpha, prev_w_r_list[i], prev_w_lu_list[i])
                    w_u = self.gamma * prev_w_u_list[i] + w_r + w_w                     # eq (20)
                    w_lu = self.least_used(w_u)
                elif self.usage_strategy == 'summary':
                    w_w = self.write_head_addressing(sig_alpha, prev_w_r_list[i], prev_w_lu)
            w_r_list.append(w_r)
            w_w_list.append(w_w)
            if self.usage_strategy == 'separate':
                w_u_list.append(w_u)
                w_lu_list.append(w_lu)
            k_list.append(k)
            p_list.append({'k': k, 'sig_alpha': sig_alpha})

        if self.usage_strategy == 'summary':
            w_u_summary = self.gamma * prev_w_u + tf.add_n(w_r_list) + tf.add_n(w_w_list)
            w_lu_summary = self.least_used(w_u_summary)

        # Reading

        read_vector_list = []
        with tf.variable_scope('reading'):
            for i in range(self.head_num):
                read_vector = tf.reduce_sum(tf.expand_dims(w_r_list[i], dim=2) * prev_M, axis=1)
                read_vector_list.append(read_vector)

        # Writing

        M = prev_M
        with tf.variable_scope('writing'):
            for i in range(self.head_num):
                w = tf.expand_dims(w_w_list[i], axis=2)
                k = tf.expand_dims(k_list[i], axis=1)
                M = M + tf.matmul(w, k)

        # controller_output -> NTM output

        if not self.output_dim:
            output_dim = x.get_shape()[1]
        else:
            output_dim = self.output_dim
        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):
            o2o_w = tf.get_variable('o2o_w', [controller_output.get_shape()[1], output_dim],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            o2o_b = tf.get_variable('o2o_b', [output_dim],
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            NTM_output = tf.nn.xw_plus_b(controller_output, o2o_w, o2o_b)

        state = {
            'controller_state': controller_state,
            'read_vector_list': read_vector_list,
            'w_r_list': w_r_list,
            'w_u_list': w_u_list,
            'w_lu_list': w_lu_list,
            'w_u_summary': w_u_summary,
            'w_lu_summary': w_lu_summary,
            'p_list': p_list,
            'M': M
        }

        self.step += 1
        return NTM_output, state

    def read_head_addressing(self, k, prev_M):
        with tf.variable_scope('read_head_addressing'):

            # Cosine Similarity

            k = tf.expand_dims(k, axis=2)
            inner_product = tf.matmul(prev_M, k)
            k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True))
            M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keep_dims=True))
            norm_product = M_norm * k_norm
            K = tf.squeeze(inner_product / (norm_product + 1e-8))                   # eq (17)

            # Calculating w^c

            K_exp = tf.exp(K)
            w = K_exp / tf.reduce_sum(K_exp, axis=1, keep_dims=True)                # eq (18)

            return w

    def write_head_addressing(self, sig_alpha, prev_w_r, prev_w_lu):
        with tf.variable_scope('write_head_addressing'):

            # Write to (1) the place that was read in t-1 (2) the place that was least used in t-1

            return sig_alpha * prev_w_r + (1 - sig_alpha) * prev_w_lu               # eq (22)

    def least_used(self, w_u):
        _, indices = tf.nn.top_k(w_u, k=self.memory_size)
        indices = indices[:, -self.head_num:]
        w_lu = tf.reduce_sum(tf.one_hot(indices, depth=self.memory_size), axis=1)
        return w_lu

    def zero_state(self, batch_size, dtype):
        def expand(x, dim, N):
            return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

        with tf.variable_scope('init', reuse=self.reuse):
            state = {
                # 'controller_state': expand(tf.tanh(tf.get_variable('init_state', self.rnn_size,
                #                             initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                #                   dim=0, N=batch_size),
                'controller_state': self.controller.zero_state(batch_size, dtype),
                'read_vector_list': [expand(tf.nn.softmax(tf.get_variable('init_r_%d' % i, [self.memory_vector_dim],
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                                  dim=0, N=batch_size)
                           for i in range(self.head_num)],
                'w_r_list': [expand(tf.nn.softmax(tf.get_variable('init_w_r_%d' % i, [self.memory_size],
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                                  dim=0, N=batch_size)
                           for i in range(self.head_num)],
                # 'w_u_list': [expand(tf.nn.softmax(tf.get_variable('init_w_u_%d' % i, [self.memory_size],
                #                             initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                #                   dim=0, N=batch_size)
                #            for i in range(self.head_num)],
                'w_u_list': [tf.zeros([batch_size, self.memory_size]) for i in range(self.head_num)],
                # 'w_lu_list': [expand(tf.nn.softmax(tf.get_variable('init_w_lu_%d' % i, [self.memory_size],
                #                             initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                #                   dim=0, N=batch_size)
                #            for i in range(self.head_num)],
                'w_lu_list': [tf.zeros([batch_size, self.memory_size]) for i in range(self.head_num)],
                'w_u_summary': tf.zeros([batch_size, self.memory_size]),
                'w_lu_summary': tf.zeros([batch_size, self.memory_size]),
                'M': expand(tf.tanh(tf.get_variable('init_M', [self.memory_size, self.memory_vector_dim],
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                                  dim=0, N=batch_size)
            }
            return state