#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None,
        word_hidden_size=None, same_word_char_hidden=True):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)

        self.char_vocab_len = len(self.target_vocab.char2id)

        self.char_output_projection = nn.Linear(hidden_size, self.char_vocab_len)
        self.decoderCharEmb = nn.Embedding(self.char_vocab_len, char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

        
        self.w_hidden_word_char = None
        self.w_cell_word_char = None
        if not same_word_char_hidden:
            self.w_hidden_word_char = nn.Parameter(torch.tensor(hidden_size, word_hidden_size))
            self.w_cell_word_char = nn.Parameter(torch.tensor(hidden_size, word_hidden_size))
        elif not (word_hidden_size is None or word_hidden_size == hidden_size):
            raise ValueError("Char and word hidden dims should be identical if same_word_char_hidden==True")


    def _make_init_state(self, h_0, c_0):
        # h_0 and c_0 both => (-1, word_hidden_dim)
        if self.w_hidden_word_char is not None:
            h_0 = torch.bmm(h_0, self.w_hidden_word_char) # (-1, char_hidden_dim)
            c_0 = torch.bmm(c_0, self.w_cell_word_char) # (-1, char_hidden_dim)

        return h_0, c_0

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.

        # input => (length, batch_size) == the word to predict, made up of a sequence of character indices
        # assumes input is x_1 .. x_n, where x_1 is START and x_n+1 is END but not included in input
        # (length, batch_size) => (length, batch_size, char_embedding_size)
        x = self.decoderCharEmb(input)  

        # h_0, c_0 => (batch_size, word_hidden_dim) == derived from the output of the word-level network
        if dec_hidden is not None:
            h_0, c_0 = dec_hidden
            h_0, c_0 = self._make_init_state(h_0, c_0)  # h_0, c_0 each => (1, batch_size, char_hidden_size)
            dec_hidden = (h_0, c_0)

        outputs = []
        # both (length, batch_size, char_hidden_size)
        hidden_states, (h_n, c_n) = self.charDecoder(x, dec_hidden)
        # (length, batch_size, char_hidden_size) => (length, batch_size, char_vocab_size)
        decoded_outputs = self.char_output_projection(hidden_states)

        s = decoded_outputs  # (length, batch_size, char_vocab_size)
        assert torch.all(h_n == hidden_states[-1])

        return s, (h_n, c_n)
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        # trim <END> token
        # s => (length-1, batch_size, char_vocab_size)
        # h_n, c_n both => (1, batch_size, char_hidden_size)
        s, (h_n, c_n) = self.forward(char_sequence[:-1], dec_hidden=dec_hidden)

        # match predictions against target, which is one character forward
        # i.e. char_sequence[1:]
        # in addition, sum the NLL values, and ignore the padding index

        # CrossEntropyLoss expects inputs (N, C) and (N)
        # where N is the number of tokens, and C is the number of classes
        s = s.view(-1, self.char_vocab_len)
        target = char_sequence[1:].view(-1)

        return nn.CrossEntropyLoss(
            ignore_index=self.target_vocab.char_pad,
            reduction='sum'
        )(s, target)
        
        ### END YOUR CODE

    def decode_greedy(
        self, 
        initialStates, 
        device, 
        max_length=21,
        premature_end_char_index: int = None):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        # both (1, batch_size, word_hidden_size)
        h_0, c_0 = initialStates

        # Use initialStates to get batch_size.
        batch_size = h_0.size(1)

        # both (1, batch_size, word_hidden_size) => (1, batch_size, hidden_size)
        h_tm1, c_tm1 = self._make_init_state(h_0, c_0)  

        # don't use [[]] * batch_size as this will copy the
        # reference to a single list for batch_size times
        outputs = [[] for _ in range(batch_size)]

        def append_chars_to_outputs(indices, chars):
            for i, c in zip(indices, chars.view(-1)):
                outputs[i].append(int(c))

        current_batch_size = batch_size
        iteration_count = 0
        current_batch_word_indices = torch.tensor(range(batch_size))  # (batch_size,)
        x_tm1 = torch.tensor([self.target_vocab.start_of_word] * batch_size).unsqueeze(0)  # (1, batch_size,)

        while True:
            # s => (1, current_batch_size, vocab_size)
            # h_t, c_t both => (1, current_batch_size, hidden_size)
            s, (h_t, c_t) = self.forward(x_tm1, (h_tm1, c_tm1))
            
            # (1, current_batch_size, vocab_size) TODO : dim=2??
            softmax_outputs = nn.Softmax(dim=2)(s)
            # (1, current_batch_size, vocab_size) => (1, current_batch_size)
            x_t = torch.argmax(softmax_outputs, dim=2)

            # check which indices match the end token
            # (current_batch_size,)
            completed_words = (x_t == self.target_vocab.end_of_word).squeeze(0)

            # only continue with non-completed words
            # (current_batch_size,) => (current_batch_size to-be,)
            current_batch_word_indices = current_batch_word_indices[~completed_words]
            # (1, current_batch_size) => (1, current_batch_size to-be)
            x_t = x_t[:, ~completed_words]
            current_batch_size = current_batch_word_indices.size(0)

            if not current_batch_size:
                break

            append_chars_to_outputs(current_batch_word_indices, x_t)

            if iteration_count >= max_length:
                if premature_end_char_index is not None:
                    append_chars_to_outputs(
                        current_batch_word_indices, 
                        torch.ones(current_batch_size) * premature_end_char_index)
                break

            iteration_count += 1
            x_tm1 = x_t

        # concatenate chars
        outputs = [''.join(self.target_vocab.id2char[c] for c in chars) for chars in outputs]

        return outputs
        ### END YOUR CODE

