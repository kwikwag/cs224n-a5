from typing import Mapping, Iterable, Callable, Optional

import numpy as np
import torch
from torch import nn


def pad_inputs_to(inputs: Mapping[int, torch.Tensor], max_len: int, padding_value) -> torch.Tensor:
    first_input = inputs[0]
    trailing_dims = first_input.size()[1:]
    out_dims = (len(inputs), max_len) + trailing_dims
    
    out_tensor = first_input.new_full(out_dims, padding_value)
    for i, tensor in enumerate(inputs):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor

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

class Highway(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.relu = nn.ReLU()
        
        self.sigmoid = nn.Sigmoid()

        self.relu_bias    = nn.Parameter(torch.zeros(input_size))
        self.sigmoid_bias = nn.Parameter(torch.zeros(input_size))

    def forward(self, inputs):
        gate = self.sigmoid(inputs + self.sigmoid_bias)
        return gate * self.relu(inputs + self.relu_bias) + (1 - gate) * inputs


class CharacterConvolutionWordEmbedding(nn.Module):
    def __init__(self, 
        num_chars: int,
        max_word_len: int,
        embedding_dim: int,
        num_filters: int,
        kernel_size: int,
        dropout_p: float,
        init_func: Optional[Callable[[torch.Tensor], None]] = None
        ):

        super().__init__()

        self.num_chars = num_chars
        self.max_word_len = max_word_len
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p

        self.pad_value = num_chars

        self.embed = nn.Embedding(
            num_embeddings=num_chars+1,  # for padding character
            embedding_dim=embedding_dim)

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
        )
        
        self.relu = nn.ReLU()
        # TODO : input/output sizes
        # TODO : we can make this not depend on max_word_len
        # self.maxpool = nn.MaxPool1d(kernel_size=max_word_len-(kernel_size-1))

        self.highway = Highway(num_filters)
        
        self.dropout = nn.Dropout(p=dropout_p)

        # TODO : parameters and biases
        # TODO : ?! I think I was wrong here using a bias as long as the window!
        self.conv_bias = nn.Parameter(torch.Tensor(
            num_filters, 
            max_word_len - (kernel_size - 1)
        ))

        if init_func:
            self.conv.bias
            init_func(self.embed.weight)
            init_func(self.conv.weight)
            init_func(self.conv.bias)
            init_func(self.highway.relu_bias)
            init_func(self.highway.sigmoid_bias)
            print(self.conv.bias.shape, self.conv_bias.shape)
            init_func(self.conv_bias)

    
    def get_batch_tensor(self, inputs: Iterable[torch.Tensor]) -> torch.Tensor:
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
                [torch.Tensor(_) for _ in inputs], 
                padding_value=self.pad_value
            ).long().T 
        input_lengths = torch.Tensor([len(_) for _ in inputs])
        # TODO : fix this so it works also with 
        #     input_lengths.max() < self.max_word_len
        assert input_lengths.max() == self.max_word_len

        # x = torch.tensor(x).to('cpu:0').long().unsqueeze(0)
        return padded_inputs, input_lengths
        

    def forward(self, 
        inputs: torch.Tensor, 
        input_lengths: torch.Tensor) -> torch.Tensor:
        # inputs = (batch_size, l)

        # (batch_size, l) => (batch_size, max_word_len)
        # x = pad_inputs_to(
        #     inputs, 
        #     max_len=self.max_word_len, 
        #     padding_value=self.num_chars)
        # x = torch.nn.utils.rnn.pad_sequence(
        #     inputs, 
        #     padding_value=self.pad_value)

        # (batch_size, max_word_len) lookup on (num_chars, embedding_dim) => 
        #     (batch_size, max_word_len, embedding_dim)
        x = self.embed(inputs)

        # (batch_size, max_word_len, embedding_dim) => 
        #     (batch_size, embedding_dim, max_word_len)
        x = x.transpose(1,2)

        # (batch_size, max_word_len, embedding_dim) conv on
        #     (num_filters, embedding_dim, kernel_size) => 
        #     (batch_size, num_filters, max_word_len - (kernel_size - 1))
        x = self.conv(x)

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
        # x = self.maxpool(x)

        # (batch_size, num_filters)
        x = self.highway(x)

        # (batch_size, num_filters)
        x = self.dropout(x)

        return x
        

if __name__ == "__main__":
    model = CharacterConvolutionWordEmbedding(
        num_chars=4,
        max_word_len=8,
        embedding_dim=32,  # a sum_(i in min_len..8) 4^i = 4^8 * (1 + 1/4 + ..)
        num_filters=3,
        kernel_size=3,
        dropout_p=0,
        init_func=lambda tensor: nn.init.constant_(tensor, 1)
    )
    inputs, input_lengths = model.get_batch_tensor([
        [0,1,2],
        [1,1,1],
        [0,1,3,2,1,2],
        [1,3,2,1,2,2,2],
        [0,1,3,2,1,2,2,2]
    ])
    
    outputs = model(inputs, input_lengths)
    print(outputs)

