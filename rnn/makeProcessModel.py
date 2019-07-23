from __future__ import print_function

import argparse
import os
from six.moves import cPickle

from six import text_type

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default=os.path.join('save', 'review example'), help='model directory to store checkpointed models')
parser.add_argument('-n', type=int, default=500, help='number of characters to sample')
parser.add_argument('--prime', type=text_type, default='START', help='prime text')
parser.add_argument('--sample', type=int, default=1, help='0 to use max at each timestep, 1 to sample at each time step, 2 to sample on spaces')

args = parser.parse_args()

import tensorflow as tf
from rnn.model import Model

def makeProcessModel(args, p_thres):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            model.makeProcessModel(sess, chars, vocab, p_thres=p_thres)

if __name__ == '__main__':

        p_thres = 0.33
        makeProcessModel(args, p_thres)