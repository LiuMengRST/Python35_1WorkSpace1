from models.tutorials.rnn.ptb import reader
import time
import numpy as np
import tensorflow as tf

class PTBInput(object):

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data)//batch_size)-1)//num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name = name)

class PTBModel(object):

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.batch_size
        size = config.hidden_size
        vocab_size = config.vocab_size

        def lstm_cell():
            return tf.contrib.rnn.BasicLStMCell(size, forget_bias=0.0, stat_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob<1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), oout_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRnnCell([attn_cell() for _ in range(config.num_layers)],state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob<1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs=[]
        state=self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step>0:tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :],state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1),[-1, size])
        softmax_w = tf.get_variable("softmax_w",[size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) +softmax_b
        loss = tf.contrib.legacy_seq2seq.sequenct_loss_by_example([logits], [tf.reshape(input_.targets,[-1])],[tf.ones([batch_size * num_steps],dtype=tf.float32)])
        self._cost=cost=tf.reduce_sum(loss) / batch_size
        self._finam_statl = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        def assign_lr(self, session, lr_value):
            session.run(self._lr_update, feed_dict={self._new_lr:lr_value})

        @property
        def input(self):
            return self._input

        @property
        def initial_state(self):
            return self._initial_state

        @property
        def cost(self):
            return self._cost

        @property
        def final_state(self):
            return self._final_state

        @property
        def lr(self):
            return self._lr

        @property
        def train_op(self):
            return self._train_op

class SamllConfig(object):
    init_scale = 0.1
    learning_rate =1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay=0.5
    batch_size = 20
    vocab_size = 10000

class TestConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000