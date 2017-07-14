from utils import OmniglotDataLoader, one_hot_decode
import tensorflow as tf
import argparse
from model import NTMOneShotLearningModel
from tensorflow.python import debug as tf_debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--restore_training', default=False)
    parser.add_argument('--model', default="LSTM")
    parser.add_argument('--rnn_size', default=128)
    parser.add_argument('--image_width', default=20)
    parser.add_argument('--image_height', default=20)
    parser.add_argument('--rnn_num_layers', default=3)
    parser.add_argument('--n_classes', default=5)
    parser.add_argument('--seq_length', default=50)
    parser.add_argument('--memory_size', default=128)
    parser.add_argument('--memory_vector_dim', default=20)
    parser.add_argument('--shift_range', default=5)
    parser.add_argument('--read_head_num', default=1)
    parser.add_argument('--write_head_num', default=1)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--test_batch_num', default=10)
    parser.add_argument('--num_epoches', default=100000)
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--save_dir', default='./save/one_shot_learning')
    parser.add_argument('--tensorboard_dir', default='./summary/one_shot_learning')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)

def train(args):
    model = NTMOneShotLearningModel(args)
    data_loader = OmniglotDataLoader(image_size=(args.image_width, args.image_height))
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(args.tensorboard_dir + '/' + args.model, sess.graph)
        for b in range(args.num_epoches):
            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            if b % 100 == 0:        # test
                learning_loss = sess.run(model.learning_loss, feed_dict=feed_dict)
                merged_summary = sess.run(model.learning_loss_summary, feed_dict=feed_dict)
                state_list = sess.run(model.state_list, feed_dict=feed_dict)
                print(state_list)
                train_writer.add_summary(merged_summary, b)
                print('batches %d, loss %g' % (b, learning_loss))
            else:                   # train
                sess.run(model.train_op, feed_dict=feed_dict)
            if b % 5000 == 0 and b > 0:
                saver.save(sess, args.save_dir + '/' + args.model + '/model.tfmodel', global_step=b)

def test(args):
    model = NTMOneShotLearningModel(args)
    data_loader = OmniglotDataLoader(image_size=(args.image_width, args.image_height))
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        correct = [0] * args.seq_length
        total = [0] * args.seq_length
        for b in range(args.test_batch_num):
            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length, type='test')
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
            for i in range(args.batch_size):
                y_i = one_hot_decode(y)[i]
                output_i = one_hot_decode(output)[i]
                print(y_i)
                print(output_i)
                class_count = [0] * args.n_classes
                for j in range(args.seq_length):
                    class_count[y_i[j]] += 1
                    total[class_count[y_i[j]]] += 1
                    if y_i[j] == output_i[j]:
                        correct[class_count[y_i[j]]] += 1
            print(learning_loss)
        for i in range(args.seq_length):
            if total[i] > 0:
                print("%dst instance: %f" % (i, float(correct[i]) / total[i]))

if __name__ == '__main__':
    main()