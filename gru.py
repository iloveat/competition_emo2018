# -*- coding: utf-8 -*-
from mxnet.gluon import nn
from mxnet.gluon import rnn
from mxnet import nd


class BiGRU(nn.Block):
    def __init__(self, hidden_size, num_layers, drop_rate, input_size, **kwargs):
        super(BiGRU, self).__init__(**kwargs)
        self.input_size = input_size
        with self.name_scope():
            self.rnn = rnn.GRU(hidden_size=hidden_size,
                               num_layers=num_layers,
                               input_size=input_size,
                               dropout=drop_rate,
                               bidirectional=True)

    def forward(self, x):
        x = x.reshape(shape=(-1, 1, self.input_size))
        out = self.rnn(x)
        return out


class TuringTTS(nn.Block):
    def __init__(self, num_out=7, num_hidden=256, **kwargs):
        super(TuringTTS, self).__init__(**kwargs)
        assert num_hidden % 2 == 0
        self.num_out = num_out
        self.num_hidden = num_hidden
        with self.name_scope():
            net = self.net = nn.Sequential()
            net.add(nn.Dense(self.num_hidden, activation='tanh'))
            net.add(nn.Dense(self.num_hidden, activation='tanh'))
            net.add(nn.Dense(self.num_hidden, activation='tanh'))
            net.add(BiGRU(self.num_hidden/2, 2, 0.1, self.num_hidden))
            net.add(nn.Flatten())
            self.final_dense = nn.Dense(self.num_out)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
        x = nd.concat(out[out.shape[0]-1], out[0], dim=0)
        x = nd.reshape(x, shape=(-1, x.shape[0]))
        return self.final_dense(x)






