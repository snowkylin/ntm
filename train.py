import tensorflow as tf
import numpy as np
import argparse
from model import NTM_model
from utils import generate_random_strings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="NTM")
    # parser.add_argument('--model', default="DNC_DEEPMIND")
    parser.add_argument('--rnn_size', default=256)
    parser.add_argument('--rnn_num_layers', default=3)
    parser.add_argument('--max_seq_length', default=5)
    parser.add_argument('--memory_size', default=10)
    parser.add_argument('--memory_vector_dim', default=4)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--vector_dim', default=4)
    parser.add_argument('--num_epoches', default=10000)
    parser.add_argument('--learning_rate', default=0.0001)
    args = parser.parse_args()
    train(args)

def train(args):
    model_list = [NTM_model(args, 1)]
    for seq_length in range(2, args.max_seq_length + 1):
        model_list.append(NTM_model(args, seq_length, reuse=True))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for b in range(args.num_epoches):
            seq_length = np.random.randint(1, args.max_seq_length + 1)
            x = generate_random_strings(args.batch_size, seq_length, args.vector_dim)
            model = model_list[seq_length - 1]
            feed_dict = {model.x: x}
            #print(sess.run([model.state_list, model.output_list], feed_dict=feed_dict))
            sess.run(model.train_op, feed_dict=feed_dict)
            if b % 100 == 0:
                print(x[0, :, :])
                print(sess.run(model.o, feed_dict=feed_dict)[0, :, :])
                copy_loss = sess.run(model.copy_loss, feed_dict=feed_dict)
                print('batches %d, loss %g' % (b, copy_loss))

if __name__ == '__main__':
    main()