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


with open('train_list.txt') as f:
    lines = f.readlines()
    for line in lines:
        input_name = line.replace('\n', '')+'_array'
        # print input_name
        input_x = np.fromfile(input_name, dtype=np.float32).reshape((-1, 1024))
        input_x = nd.array(input_x).as_in_context(ctx)
        hidden = user_forward(model=net, x=input_x)
        output = net.final_dense(hidden)
        hidden_name = input_name[:-5] + '1x1024'
        output_name = input_name[:-5] + '1x7'
        hidden.asnumpy().astype(np.float32).tofile(hidden_name)
        output.asnumpy().astype(np.float32).tofile(output_name)
        print hidden_name
        print output_name


with open('valid_list.txt') as f:
    lines = f.readlines()
    for line in lines:
        input_name = line.replace('\n', '')+'_array'
        # print input_name
        input_x = np.fromfile(input_name, dtype=np.float32).reshape((-1, 1024))
        input_x = nd.array(input_x).as_in_context(ctx)
        hidden = user_forward(model=net, x=input_x)
        output = net.final_dense(hidden)
        hidden_name = input_name[:-5] + '1x1024'
        output_name = input_name[:-5] + '1x7'
        hidden.asnumpy().astype(np.float32).tofile(hidden_name)
        output.asnumpy().astype(np.float32).tofile(output_name)
        print hidden_name
        print output_name







