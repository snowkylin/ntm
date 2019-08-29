import tensorflow as tf


class NTMCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, rnn_size, memory_size, memory_vector_dim, read_head_num, write_head_num,
                 addressing_mode='content_and_location', shift_range=1, output_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.rnn_size = rnn_size
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.addressing_mode = addressing_mode
        self.output_dim = output_dim
        self.shift_range = shift_range

        # Controller RNN layer
        self.controller = tf.keras.layers.LSTMCell(units=self.rnn_size)

        # Calculate number of parameters per read/write head
        # controller_output     -> k (dim = memory_vector_dim, compared to each vector in M, Sec 3.1)
        #                       -> beta (positive scalar, key strength, Sec 3.1)                -> w^c
        #                       -> g (scalar in (0, 1), blend between w_prev and w^c, Sec 3.2)  -> w^g
        #                       -> s (dim = shift_range * 2 + 1, shift weighting, Sec 3.2)      -> w^~
        #                            (not memory_size, that's too wide)
        #                       -> gamma (scalar (>= 1), sharpen the final result, Sec 3.2)     -> w    * num_heads
        # controller_output     -> erase, add vector (dim = memory_vector_dim, \in (0, 1), Sec 3.2)     * write_head_num

        self.num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        self.num_heads = self.read_head_num + self.write_head_num
        self.total_parameter_num = self.num_parameters_per_head * self.num_heads + self.memory_vector_dim * 2 * self.write_head_num

        # From controller output to parameters:
        self.controller_output_to_params = tf.keras.layers.Dense(units=self.total_parameter_num, use_bias=True)

        # From controller output to NTM output:
        self.controller_output_to_ntm_output = tf.keras.layers.Dense(units=self.output_dim, use_bias=True)

        self.init_memory_state = self.add_weight(name='init_memory_state',
                                                 shape=[self.rnn_size],
                                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
        self.init_carry_state = self.add_weight(name='init_carry_state',
                                                shape=[self.rnn_size],
                                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
        self.init_r = [self.add_weight(name='init_r_%d' % i,
                                       shape=[self.memory_vector_dim],
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
                       for i in range(self.read_head_num)]
        self.init_w = [self.add_weight(name='init_w_%d' % i,
                                       shape=[self.memory_size],
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
                       for i in range(self.read_head_num + self.write_head_num)]
        self.init_M = self.add_weight(name='init_M',
                                      shape=[self.memory_size, self.memory_vector_dim],
                                      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))

    @property
    def state_size(self):
        return self.rnn_size

    def build(self, inputs_shape):
        if not self.output_dim:
            self.output_dim = inputs_shape[1]

    def call(self, inputs, states):
        prev_read_vector_list = states['read_vector_list']  # read vector in Sec 3.1 (the content that is
                                                            # read out, length = memory_vector_dim)
        prev_controller_state = states['controller_state']  # state of controller (LSTM hidden state)
        controller_input = tf.concat([inputs] + prev_read_vector_list, axis=1)
        controller_output, controller_state = self.controller(controller_input, prev_controller_state)
        parameters = self.controller_output_to_params(controller_output)
        head_parameter_list = tf.split(parameters[:, :self.num_parameters_per_head * self.num_heads], self.num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, self.num_parameters_per_head * self.num_heads:], 2 * self.write_head_num, axis=1)

        prev_w_list = states['w_list']  # vector of weightings (blurred address) over locations
        prev_M = states['M']
        w_list = []
        p_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            # Some functions to constrain the parameters in specific range
            # exp(x)                -> x > 0
            # sigmoid(x)            -> x \in (0, 1)
            # softmax(x)            -> sum_i x_i = 1
            # log(exp(x) + 1) + 1   -> x > 1

            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = tf.sigmoid(head_parameter[:, self.memory_vector_dim]) * 10  # do not use exp, it will explode!
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])
            s = tf.nn.softmax(
                head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
            )
            gamma = tf.math.log(tf.exp(head_parameter[:, -1]) + 1) + 1

            # Addressing mechanism in figure 2
            w = self._addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])

            w_list.append(w)
            p_list.append({'k': k, 'beta': beta, 'g': g, 's': s, 'gamma': gamma})

        # Reading (Sec 3.1)
        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], axis=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)

        # Writing (Sec 3.2)
        write_w_list = w_list[self.read_head_num:]
        M = prev_M
        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
            M = M * (tf.ones(M.get_shape()) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)

        ntm_output = self.controller_output_to_ntm_output(controller_output)

        state = {
            'controller_state': controller_state,
            'read_vector_list': read_vector_list,
            'w_list': w_list,
            # 'p_list': p_list,
            'M': M
        }
        return ntm_output, state

    def _addressing(self, k, beta, g, s, gamma, prev_M, prev_w):
        # Sec 3.3.1 Focusing by Content
        # Cosine Similarity
        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(prev_M, k)
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keepdims=True))
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keepdims=True))
        norm_product = M_norm * k_norm
        K = tf.squeeze(inner_product / (norm_product + 1e-8))                   # eq (6)

        # Calculating w^c
        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keepdims=True)   # eq (5)

        if self.addressing_mode == 'content':                                   # Only focus on content
            return w_c

        # Sec 3.3.2 Focusing by Location
        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w                                        # eq (7)

        s = tf.concat([s[:, :self.shift_range + 1],
                       tf.zeros([s.get_shape()[0], self.memory_size - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1)
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
            axis=1
        )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)      # eq (8)
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keepdims=True)         # eq (9)

        return w

    @staticmethod
    def _expand(x, dim, N):
        return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initial_state = {
            'controller_state': [self._expand(tf.tanh(self.init_memory_state), dim=0, N=batch_size),
                                 self._expand(tf.tanh(self.init_carry_state), dim=0, N=batch_size)],
            'read_vector_list': [self._expand(tf.nn.tanh(self.init_r[i]), dim=0, N=batch_size)
                                 for i in range(self.read_head_num)],
            'w_list': [self._expand(tf.nn.softmax(self.init_w[i]), dim=0, N=batch_size)
                       for i in range(self.read_head_num + self.write_head_num)],
            'M': self._expand(tf.tanh(self.init_M), dim=0, N=batch_size)
        }
        return initial_state
