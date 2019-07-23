import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from pyvis.network import Network

import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import pylab
import re

class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'lstm':
            cell_fn = rnn.LSTMCell
        # elif args.model == 'dnn':
        #     cell_fn =
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell, input_keep_prob=args.input_keep_prob, output_keep_prob=args.output_keep_prob)

            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if training  and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        loss = legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([args.batch_size * args.seq_length])]
        )
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)


    # predict next activities of every activity
    # No sequence information
    def sample(self, sess, chars, vocab):

        with open('data/review_example_activities.txt') as f:
            data = f.read()

        for activity in data.split("\n"):
            G = Network(height=1000, width=1000, directed=True)
            G.add_node(activity, level=0)

            state = sess.run(self.cell.zero_state(1, tf.float32))
            x = np.zeros((1, 1))
            x[0, 0] = vocab[activity]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            ps = probs[0]

            index = 0
            for p in ps:

                if p > 0.01:
                    if chars[index] == "\n":
                        index += 1
                        continue
                    print(activity +"\t\t" + chars[index] + "\t{%f}"%p)
                    G.add_node(chars[index], physics=True, level=1)
                    G.add_edge(activity, chars[index], value=int(p*100),title=int(p*100), physics=True, arrowStrikethrough=False )
                index += 1
            print()

            G.barnes_hut(gravity=-10000)

            # G.show_buttons()
            G.save_graph(os.path.join("results", re.sub('[-=.#/?:$}]', '', activity) +".html"))

    #for sequential input
    def sequence_sample(self, sess, chars, vocab):

        with open('data/review_example_sequence.txt') as f:
            data = f.read()

        state = sess.run(self.cell.zero_state(1, tf.float32))
        x = np.zeros((1, 1))

        for activity in data.split("->"):


            x[0, 0] = vocab[activity]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            ps = probs[0]

            if activity == "time-out 2":
                G = Network(height=1000, width=1000, directed=True)
                G.add_node(activity, level=0)
                index = 0
                for p in ps:

                    if p > 0.01:
                        if chars[index] == "\n":
                            index += 1
                            continue
                        print(activity + "\t\t" + chars[index] + "\t{%f}" % p)
                        G.add_node(chars[index], physics=True, level=1)
                        G.add_edge(activity, chars[index], value=int(p * 100), title=int(p * 100), physics=True,
                                   arrowStrikethrough=False)
                    index += 1
                print()

                G.barnes_hut(gravity=-10000)

                # G.show_buttons()
                G.save_graph(os.path.join("results", re.sub('[-=.#/?:$}]', '', activity) + "_sequence.html"))




# prediction을 통해 가능한 모든 predicted control path를 구함
    def makeProcessModel(self, sess, chars, vocab, activity="START", num=0, p_thres=0):
        state = sess.run(self.cell.zero_state(1, tf.float32))

        self.outFile = open("review_example_predicted_workcase" + "_{:.0%}".format(p_thres) + ".txt","w")

        self.findPath(state, sess, chars, vocab, activity, num, p_thres=p_thres)

        print("done")

    def findPath(self, state, sess, chars, vocab, activity, num, path="", p_thres=0):
        if num > 50:
            return
        elif activity == "END":
            path = path[2:] + "->" + activity + "\n"
            self.outFile.write(path)
            return

        x = np.zeros((1, 1))
        x[0, 0] = vocab[activity]
        feed = {self.input_data: x, self.initial_state: state}
        [probs, state] = sess.run([self.probs, self.final_state], feed)
        ps = probs[0]

        index = 0
        for p in ps:
            if p > p_thres:
                if chars[index] == "\n":
                    continue

                self.findPath(state, sess, chars, vocab, chars[index], num + 1, (path + "->" + activity)[:], p_thres=p_thres)

            index += 1
