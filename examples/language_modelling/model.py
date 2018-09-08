""" Credit goes to https://github.com/salesforce/awd-lstm-lm """
import torch as th
import torch.nn as nn

from dropout import LockedDropout, WeightDrop, embedded_dropout


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
            self,
            rnn_type,
            ntoken,
            ninp,
            nhid,
            nlayers,
            dropout=0.5,
            dropouth=0.5,
            dropouti=0.5,
            dropoute=0.1,
            wdrop=0,
            tie_weights=False,
    ):
        super(RNNModel, self).__init__()

        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type not in {'LSTM', 'GRU'}:
            raise ValueError(f"RNN type {rnn_type} is not supported")

        if rnn_type == "LSTM":
            self.rnns = [
                th.nn.LSTM(
                    ninp if l == 0 else nhid,
                    nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                    1,
                    dropout=0,
                )
                for l in range(nlayers)
            ]
            if wdrop:
                self.rnns = [
                    WeightDrop(rnn, ["weight_hh_l0"], dropout=wdrop)
                    for rnn in self.rnns
                ]

        if rnn_type == "GRU":
            self.rnns = [
                th.nn.GRU(
                    ninp if l == 0 else nhid,
                    nhid if l != nlayers - 1 else ninp,
                    1,
                    dropout=0,
                )
                for l in range(nlayers)
            ]
            if wdrop:
                self.rnns = [
                    WeightDrop(rnn, ["weight_hh_l0"], dropout=wdrop)
                    for rnn in self.rnns
                ]

        self.rnns = th.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(1, 1)

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1

        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):

        emb = embedded_dropout(
            self.encoder, input,
            dropout=self.dropoute if self.training else 0
        )

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)

            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)

        hidden = new_hidden
        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output

        if return_h:
            return result, hidden, raw_outputs, outputs

        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        if self.rnn_type == "LSTM":
            return [
                (
                    weight.new(
                        1,
                        bsz,
                        self.nhid
                        if l != self.nlayers - 1
                        else (self.ninp if self.tie_weights else self.nhid),
                    ).zero_(),
                    weight.new(
                        1,
                        bsz,
                        self.nhid
                        if l != self.nlayers - 1
                        else (self.ninp if self.tie_weights else self.nhid),
                    ).zero_(),
                )
                for l in range(self.nlayers)
            ]

        elif self.rnn_type == "GRU":
            return [
                weight.new(
                    1,
                    bsz,
                    self.nhid
                    if l != self.nlayers - 1
                    else (self.ninp if self.tie_weights else self.nhid),
                ).zero_()
                for l in range(self.nlayers)
            ]
