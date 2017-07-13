from utils import OmniglotDataLoader
import tensorflow as tf
import numpy as np
import argparse
from model import NTMOneShotLearningModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--test_seq_length', type=int, default=20)
    parser.add_argument('--model', default="LSTM")
    parser.add_argument('--rnn_size', default=64)
    parser.add_argument('--image_width', default=20)
    parser.add_argument('--image_height', default=20)
    parser.add_argument('--rnn_num_layers', default=3)
    parser.add_argument('--n_classes', default=5)
    parser.add_argument('--seq_length', default=25)
    parser.add_argument('--memory_size', default=20)
    parser.add_argument('--memory_vector_dim', default=10)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--vector_dim', default=4)
    parser.add_argument('--num_epoches', default=1000000)
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--save_dir', default='./save/one_shot_learning')
    parser.add_argument('--tensorboard_dir', default='./summary/one_shot_learning')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    # elif args.mode == 'test':
    #     test(args)

def train(args):
    model = NTMOneShotLearningModel(args)
    data_loader = OmniglotDataLoader(image_size=(args.image_width, args.image_height))
    with tf.Session() as sess:
        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
        for b in range(args.num_epoches):
            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            if b % 100 == 0:        # test
                learning_loss = sess.run(model.learning_loss, feed_dict=feed_dict)
                merged_summary = sess.run(model.learning_loss_summary, feed_dict=feed_dict)
                train_writer.add_summary(merged_summary, b)
                print('batches %d, loss %g' % (b, learning_loss))
            else:                   # train
                sess.run(model.train_op, feed_dict=feed_dict)
            if b % 5000 == 0 and b > 0:
                saver.save(sess, args.save_dir + '/model.tfmodel', global_step=b)

if __name__ == '__main__':
    main()