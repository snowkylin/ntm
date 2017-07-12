import tensorflow as tf
import numpy as np
import argparse
from model import NTM_model
from utils import generate_random_strings
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="NTM")
    parser.add_argument('--rnn_size', default=64)
    parser.add_argument('--rnn_num_layers', default=3)
    parser.add_argument('--max_seq_length', default=10)
    parser.add_argument('--memory_size', default=20)
    parser.add_argument('--memory_vector_dim', default=4)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--vector_dim', default=4)
    parser.add_argument('--num_epoches', default=100000)
    parser.add_argument('--learning_rate', default=0.0001)
    args = parser.parse_args()
    train(args)

def train(args):
    model_list = [NTM_model(args, 1)]
    for seq_length in range(2, args.max_seq_length + 1):
        model_list.append(NTM_model(args, seq_length, reuse=True))
    # model = NTM_model(args, args.max_seq_length)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
        tf.global_variables_initializer().run()
        plt.ion()
        plt.show()
        for b in range(args.num_epoches):
            seq_length = np.random.randint(1, args.max_seq_length + 1)
            model = model_list[seq_length - 1]
            # seq_length = args.max_seq_length
            x = generate_random_strings(args.batch_size, seq_length, args.vector_dim)
            feed_dict = {model.x: x}
            # print(sess.run([model.state_list, model.output_list], feed_dict=feed_dict))
            if b % 100 == 0:        # test
                p = 0               # select p th sample in the batch to show
                print(x[p, :, :])
                print(sess.run(model.o, feed_dict=feed_dict)[p, :, :])
                state_list = sess.run(model.state_list, feed_dict=feed_dict)
                w_plot = []
                M_plot = np.concatenate([state['M'][p, :, :] for state in state_list])
                for state in state_list:
                    w_plot.append(np.concatenate([state['w_list'][0][p, :], state['w_list'][1][p, :]]))
                plt.imshow(w_plot, interpolation='nearest', cmap='gray')
                plt.draw()
                plt.pause(0.001)
                copy_loss = sess.run(model.copy_loss, feed_dict=feed_dict)
                merged_summary = sess.run(model.copy_loss_summary, feed_dict=feed_dict)
                train_writer.add_summary(merged_summary, b)
                print('batches %d, loss %g' % (b, copy_loss))
            else:                   # train
                sess.run(model.train_op, feed_dict=feed_dict)

if __name__ == '__main__':
    main()