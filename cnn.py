#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import math
import unittest
from typing import Iterable

import torch
import torch.nn as nn

# thanks to https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036
def sequence_mask(
        lengths: Iterable[int], 
        maxlen: int = None,
        dtype: torch.dtype = torch.bool) -> torch.Tensor:
    if maxlen is None:
        maxlen = torch.max(lengths)
    mask = ~(torch.ones((len(lengths), int(maxlen))).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g
    def __init__(self,
        embedding_dim: int,
        num_filters: int,
        kernel_size: int = 5,
        padding: int = 1):

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        
        self.relu = nn.ReLU()

    def forward(self, inputs, input_lengths=None):
        x = inputs

        # (batch_size, max_word_len, embedding_dim) conv on
        #     (num_filters, embedding_dim, kernel_size) => 
        #     (batch_size, num_filters, max_word_len - (kernel_size - 1))
        x = self.conv(x)

        if input_lengths is not None:
            # Remove convolution outputs that were mixed with pad character
            # (batch_size, num_filters, max_word_len - (kernel_size - 1))
            valid_conv_lengths = input_lengths - (self.kernel_size - 1)
            valid_conv_lengths[valid_conv_lengths < 0] = 0
            mask = sequence_mask(lengths=valid_conv_lengths)
            x = x.transpose(1,2)
            x[~mask, :] = 0
            x = x.transpose(1,2)
        
        # (batch_size, num_filters, max_word_len - (kernel_size - 1))
        x = self.relu(x)

        # (batch_size, num_filters, max_word_len - (kernel_size - 1)) =>
        #     (batch_size, num_filters)
        x, _ = torch.max(x, dim=2)

        return x

    ### END YOUR CODE


class CNNTests(unittest.TestCase):
    def test_shape(self):
        for batch_size, e_char, f, k in itertools.product(
            [1, 2], # batch sizes
            [1, 3], # embedding dimensions
            [1, 4], # number of filters
            [1, 5], # kernel sizes
        ):
            for n in [k, k*2]:
                with self.subTest(dict(
                    batch_size=batch_size,
                    e_char=e_char, f=f, k=k, n=n)):

                    c = CNN(e_char, num_filters=f, kernel_size=k)
                    with torch.no_grad():
                        c.conv.weight.copy_(torch.zeros(f, e_char, k))
                        c.conv.bias.copy_(torch.zeros(1))

                    inputs = torch.ones(batch_size, e_char, n)
                    outputs = c(inputs)

                    self.assertEqual(
                        outputs.tolist(),
                        [[0] * f] * batch_size
                    )

    def test_values(self):
        e_char = 3
        f = 2
        k = 5
        n = 10
        padding = 0

        c = CNN(e_char, num_filters=f, padding=padding)
        with torch.no_grad():
            #   2 ... 14  ^
            #  1 ... 13  / e_char
            # 0 ... 12  /
            # -------> k
            # second filter is same as above, but negative values
            c.conv.weight.copy_(torch.Tensor(
                list(range(e_char * k)) +        # first filter - positive values
                [-i for i in range(e_char * k)]  # second filter - negative values
                ).view(f, k, e_char).transpose(1,2))
            c.conv.bias.copy_(torch.ones(1))

        #   27 24 ... 0 ^
        #  28 25 ... 1 / e_char
        # 29 26 ... 2 /
        # ----------> n
        inputs = torch.Tensor(
            list(reversed(range(e_char * n)))).view(1, n, e_char).transpose(1,2)
        outputs = c(inputs)

        self.assertEqual(
            outputs.tolist(),
            [[
                # The first filter gives a sum of (29 - i - (t*3)) * (i)
                # (plus 1 for the bias) over the window at position t.
                # Since t == 0 has the largest values, it will be the value
                # selected by the max-pooling.
                float(sum(
                    (29 - i) * (i)
                    for i in range(e_char * k)
                ) + 1),

                # The second filter calculates sum of (29 - i - (t*3)) * (-i)
                # which is all negative, thus it should give 0 under ReLU
                0.0
            ]]
        )


    def test_values_with_padding(self):
        e_char = 3
        f = 2
        k = 5
        n = 10

        for padding, t_max in [
            (1, 0),
            (2, 1),
            (3, 2),
        ]:
            
            with self.subTest(dict(padding=padding, t_max=t_max)):
                c = CNN(e_char, num_filters=f, padding=padding)
                with torch.no_grad():
                    #   2 ... 14  ^
                    #  1 ... 13  / e_char
                    # 0 ... 12  /
                    # -------> k
                    # second filter is same as above, but negative values
                    c.conv.weight.copy_(torch.Tensor(
                        list(range(e_char * k)) +        # first filter - positive values
                        [-i for i in range(e_char * k)]  # second filter - negative values
                        ).view(f, k, e_char).transpose(1,2))
                    c.conv.bias.copy_(torch.ones(1))

                #   27 24 ... 0 ^
                #  28 25 ... 1 / e_char
                # 29 26 ... 2 /
                # ----------> n
                inputs = torch.Tensor(
                    list(reversed(range(e_char * n)))).view(1, n, e_char).transpose(1,2)
                outputs = c(inputs)

                
                self.assertEqual(
                    outputs.tolist(),
                    [[
                        # This is similar to the calculation above, only now the first
                        # window has the input values shifted by the padding,
                        # so the convolution for the first window (which will still
                        # give the maximal value) is
                        # 0*0 + 0*1 + 0*2 + 29*3 + 28*4 + ... + 20*12 + 19*13 + 18*14
                        # instead of
                        # 29*0 + 28*1 + ... + 17*12 + 16*13 + 15*14
                        float(sum(
                            (29 - i - max(t_max - padding, 0)*3) * (i + 3 * max(padding - t_max, 0))
                            for i in range(e_char * (k - max(padding - t_max, 0)))
                        ) + 1),

                        # The second filter still gives 0 under ReLU
                        0.0
                    ]]
                )

if __name__ == "__main__":
    import itertools
    unittest.main()
