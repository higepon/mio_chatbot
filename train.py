import tensorflow as tf
import config


def create_seq2seq(encoder_inputs, decoder_inputs, do_decode, cell):
    return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
        encoder_inputs=encoder_inputs,
        decoder_inputs=decoder_inputs,
        cell=cell,
        num_encoder_symbols=config.ENC_VOCAB,
        num_decoder_symbols=config.DEC_VOCAB,
        embedding_size=config.HIDDEN_SIZE,
        output_projection=None, # todo
        feed_previous=do_decode)


def create_model():
    cells = [tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE) for _ in range(config.NUM_LAYERS)]
    multi_cell = tf.contrib.rnn.MultiRNNCell(cells=cells)
    encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                      for i in range(config.BUCKETS[-1][0])]
    decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                      for i in range(config.BUCKETS[-1][1] + 1)]
    decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                     for i in range(config.BUCKETS[-1][1] + 1)]

    x = create_seq2seq(encoder_inputs=encoder_inputs,
                       decoder_inputs=decoder_inputs,
                       do_decode=False,
                       cell=multi_cell)
    print(x)

if __name__ == '__main__':
    create_model()
