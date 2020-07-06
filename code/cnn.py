"""
Convolutional neural network for encoding strings at the character level.

Author: Shruti Rijhwani
Contact: srijhwan@andrew.cmu.edu

Please cite: 
Soft Gazetteers for Low-Resource Named Entity Recognition (ACL 2020)
https://www.aclweb.org/anthology/2020.acl-main.722
"""
import dynet as dy
from collections import defaultdict
import numpy as np


class CNNModule(object):
    def __init__(
        self,
        model,
        char_vocab,
        embed_size=30,
        window_size=3,
        filter_size=30,
        dropout=0.33,
    ):
        self.vocab = char_vocab
        self.model = model
        self.char_embeds = self.model.add_lookup_parameters(
            (len(char_vocab), 1, 1, embed_size),
            init=dy.UniformInitializer(np.sqrt(3.0 / embed_size)),
        )
        self.filter_size = filter_size
        self.W_cnn = self.model.add_parameters(
            (1, window_size, embed_size, filter_size)
        )
        self.b_cnn = self.model.add_parameters((filter_size))
        self.b_cnn.zero()
        self.dropout = dropout

    def encode(self, word, training=False):
        W_cnn = dy.parameter(self.W_cnn)
        b_cnn = dy.parameter(self.b_cnn)

        embs = dy.concatenate([dy.lookup(self.char_embeds, x) for x in word[:45]], d=1)
        if self.dropout > 0 and training:
            embs = dy.dropout(embs, self.dropout)
        cnn_out = dy.conv2d_bias(
            embs, W_cnn, b_cnn, stride=(1, 1), is_valid=False
        )  # maybe change this? diagram shows padding
        max_pool = dy.max_dim(cnn_out, d=1)
        rep = dy.reshape(dy.tanh(max_pool), (self.filter_size,))
        return rep
