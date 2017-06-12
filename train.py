import tensorflow as tf
import config


single_cell = tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE)
multi_cell = tf.contrib.rnn.MultiRNNCell([single_cell] * config.NUM_LAYERS)

encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                  for i in range(config.BUCKETS[-1][0])]
decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                  for i in range(config.BUCKETS[-1][1] + 1)]
decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                 for i in range(config.BUCKETS[-1][1] + 1)]


# tf.contrib.legacy_seq2seq.model_with_buckets()