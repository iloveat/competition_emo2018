# -*- coding: utf-8 -*-
import gru as gru
from mxnet import init
from mxnet import nd
import mxnet as mx
import numpy as np


saved_ckp_model = './ckp/emo_gru-000.params'


def user_forward(model, x):
    out = model.net(x)
    out = nd.concat(out[out.shape[0]-1], out[0], dim=0)
    out = nd.reshape(out, shape=(-1, out.shape[0]))
    return out


ctx = mx.gpu()
net = gru.TuringTTS(num_hidden=512)
net.initialize(ctx=ctx, init=init.Xavier())
net.hybridize()
net.load_params(saved_ckp_model, ctx=ctx)


test_data = './data/Sad/021913080.mp4_array'

input_x = np.fromfile(test_data, dtype=np.float32).reshape((-1, 1024))

input_x = nd.array(input_x).as_in_context(ctx)

output = user_forward(model=net, x=input_x)
print output




