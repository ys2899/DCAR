from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import pdb

from tensorflow.contrib import legacy_seq2seq
from lib.metrics import masked_mae_loss
from model.dcrnn_cell import DCGRUCell


class DCRNNARModel(object):
    def __init__(self, is_training, batch_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        self._scaler = scaler

        # Train and loss
        self._loss = None
        self._mae = None
        self._train_op = None

        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = 1


        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')
        self.train_inputs = tf.concat((self._inputs, self._labels), axis=1)

        self._targets = tf.slice(self.train_inputs, [0, 0, 0, 0],
                                 [batch_size, horizon + seq_len - 1, num_nodes, 1], name='targets')

        cell_1st_layer = DCGRUCell(rnn_units, adj_mx, first_layer=True, max_diffusion_step=max_diffusion_step,
                                   num_nodes=num_nodes, filter_type=filter_type)

        cell = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                         filter_type=filter_type)

        # We temporarily change the num_proj from output_dim to input_dim.
        cell_with_projection = DCGRUCell(rnn_units, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                                         num_proj=output_dim, filter_type=filter_type)

        decoding_cells = [cell_1st_layer] + [cell] * (num_rnn_layers - 2) + [cell_with_projection]
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells, state_is_tuple=True)
        global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('DCRNN_SEQ'):

            train_inputs = tf.unstack(self.train_inputs, axis=1)

            # We need to tear the train_inputs up.
            def _loop_function(prev, i):
                # To do: the probability of using the previous is increasing when going towards the
                # end of the sequence.
                train_input = train_inputs[i]
                if len(train_input.shape)==3:
                    day_input = tf.slice(train_input, [0, 0, 1], [train_input.shape[0], train_input.shape[1], 1])
                    time_input = tf.slice(train_input, [0, 0, 2], [train_input.shape[0], train_input.shape[1], 1])

                if is_training:
                    if use_curriculum_learning:
                        c = tf.random_uniform((), minval=0.0, maxval=1.0)
                        threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                        if i < seq_len:
                            result = train_input
                        else:
                            result = tf.cond(tf.less(c, threshold), lambda: train_inputs[i], lambda: tf.concat([prev, day_input, time_input], axis=-1))
                    else:
                        result = train_inputs[i]
                else:
                    if i < seq_len:
                        result = train_inputs[i]
                    else:
                        result = tf.concat([prev, day_input, time_input], axis=-1)
                return result

            initial_state = (tf.zeros(shape=(64, 13248)), tf.zeros(shape=(64, 13248)))
            state = initial_state
            outputs = []
            prev = None

            for i, inp in enumerate(train_inputs):

                with tf.variable_scope("loop_function", reuse=True):
                    if prev is not None:
                        inp = _loop_function(prev, i)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                output, state = decoding_cells(inp, state)
                output = tf.reshape(output, (batch_size, num_nodes, output_dim))
                outputs.append(output)
                prev = output

        outputs_dim = 1
        outputs = tf.stack(outputs[:-1], axis=1)
        self._outputs = tf.reshape(outputs, (batch_size, horizon + seq_len - 1, num_nodes, outputs_dim), name='outputs')
        self._merged = tf.summary.merge_all()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def targets(self):
        return self._targets

    @property
    def loss(self):
        return self._loss

    @property
    def mae(self):
        return self._mae

    @property
    def merged(self):
        return self._merged

    @property
    def outputs(self):
        return self._outputs

