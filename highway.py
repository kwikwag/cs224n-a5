#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import math
import unittest

import torch
import torch.nn as nn

class Highway(nn.Module):
    ### YOUR CODE HERE for part 1f

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.relu = nn.ReLU()
        
        self.sigmoid = nn.Sigmoid()

        self.relu_proj = nn.Linear(input_size, input_size, bias=True)
        self.gate_proj = nn.Linear(input_size, input_size, bias=True)


    def forward(self, inputs):
        gate = self.sigmoid(self.gate_proj(inputs))
        relu = self.relu(self.relu_proj(inputs))
        return gate * relu + (1 - gate) * inputs

    ### END YOUR CODE


class HighwayTests(unittest.TestCase):

    def test_shapes(self):
        n = 3
        h = Highway(n)
        with torch.no_grad():
            h.relu_proj.weight.copy_(torch.zeros(n, n))
            h.gate_proj.weight.copy_(torch.zeros(n, n))
            h.relu_proj.bias.copy_(torch.zeros(n))
            h.gate_proj.bias.copy_(torch.zeros(n))

        inputs = torch.ones(1, n)
        outputs = h(inputs)

        self.assertEqual(
            outputs.tolist(),
            [[0.5] * n]
        )


    def test_batch(self):
        batch_size = 3
        n = 5
        h = Highway(n)
        with torch.no_grad():
            h.relu_proj.weight.copy_(torch.zeros(n, n))
            h.gate_proj.weight.copy_(torch.zeros(n, n))
            h.relu_proj.bias.copy_(torch.zeros(n))
            h.gate_proj.bias.copy_(torch.zeros(n))

        inputs = torch.ones(batch_size, n)
        outputs = h(inputs)

        self.assertEqual(
            outputs.tolist(),
            [[0.5] * n] * batch_size
        )

    def test_values(self):
        n = 3
        h = Highway(n)
        with torch.no_grad():
            # project 1->2, 2->1, 3->3
            h.relu_proj.weight.copy_(torch.Tensor([
                [0,1,0],
                [1,0,0],
                [0,0,1],
            ]))
            # project 1->1, 2->3, 2->3
            h.gate_proj.weight.copy_(torch.Tensor([
                [1,0,0],
                [0,0,1],
                [0,1,0],
            ]))

            h.relu_proj.bias.copy_(torch.Tensor([-1, -2, -3]))

            # W_s (1,2,3) = (1,3,2)
            expected_gate_transform_outputs = torch.Tensor([1, 3, 2])
            x = 1
            
            required_gate_inputs = torch.Tensor([
                -x,
                0,
                x,
            ])
            # bias that zeros W_s @ inputs and then adds the 
            # required_gate_inputs vector
            h.gate_proj.bias.copy_(
                - expected_gate_transform_outputs
                + required_gate_inputs
            )

        inputs = torch.Tensor([
            [1, 2, 3],
        ])
        outputs = h(inputs)

        # W_R (1,2,3) + b_R = (2,1,3) + (-1,-2,-3) = (1,-1,0)
        # under ReLU => (1,0,0)
        expected_relu = torch.Tensor([1, 0, 0])

        # W_s (1,2,3) + b_s = (1,3,2) + (-1,-3,-2) + (-x,0,x) = (-x,0,x)
        # under sigma => (s(-x),0,s(x))
        # also make sure we are calculating sigma correctly
        self.assertEqual(1/2-sigma(-x), sigma(x)-1/2)
        expected_gate = torch.Tensor([sigma(-x), 1/2, sigma(x)])

        expected_outputs = (
            expected_gate * expected_relu + 
            (1 - expected_gate) * inputs
        )

        self.assertEqual(
            outputs.tolist(),
            expected_outputs.tolist()
        )


if __name__ == "__main__":
    from test_utils import sigma
    unittest.main()
