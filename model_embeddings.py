#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(
        self, 
        word_embed_size, 
        vocab, 
        char_embed_size=50, 
        dropout_prob=0.3,
        kernel_size=5):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        num_chars = len(vocab.char2id)
        self.pad_value = num_chars

        # neccessary for validation code (nmt_model.save)
        self.word_embed_size = word_embed_size

        self.char_embed = nn.Embedding(
            num_embeddings=num_chars+1,  # for padding character
            embedding_dim=char_embed_size)

        self.cnn = CNN(
            embedding_dim=char_embed_size,
            num_filters=word_embed_size,
            kernel_size=kernel_size,
        )

        self.highway = Highway(word_embed_size)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        ### END YOUR CODE

    def forward(self, inputs, input_lengths=None):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        # (sentence_length, batch_size, max_word_len)
        x = inputs

        # convert to list of words
        # (sentence_length, batch_size, max_word_len) =>
        #     (sentence_length * batch_size, max_word_len)
        x = x.reshape(-1, inputs.shape[-1])

        # (sentence_length * batch_size, max_word_len) lookup on (num_chars, embedding_dim) => 
        #     (sentence_length * batch_size, max_word_len, embedding_dim)
        x = self.char_embed(x)

        # (sentence_length * batch_size, max_word_len, embedding_dim) => 
        #     (sentence_length * batch_size, embedding_dim, max_word_len)
        x = x.transpose(1,2)

        # (sentence_length * batch_size, embedding_dim, max_word_len) conv on
        #     (sentence_length * word_embed_size, embedding_dim, kernel_size) => 
        #     (sentence_length * batch_size, word_embed_size, max_word_len - (kernel_size - 1))
        # and after the max pool we remain with (sentence_length * batch_size, word_embed_size)
        x = self.cnn(x, input_lengths)

        # (sentence_length * batch_size, word_embed_size)
        x = self.highway(x)

        # (sentence_length * batch_size, word_embed_size)
        x = self.dropout(x)

        # reshape into input shape
        # (sentence_length * batch_size, word_embed_size) =>
        #     (sentence_length, batch_size, word_embed_size)
        x = x.view(*inputs.shape[:2], -1)

        return x
        ### END YOUR CODE

