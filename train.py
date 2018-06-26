# -*- coding: utf-8 -*-
import data_provider as dp
import gru as gru
import datetime
from mxnet import gluon
from mxnet import init
from mxnet import autograd
from mxnet import nd
import mxnet as mx
import numpy as np


def data_iter(data_reader):
    while not data_reader.is_finish():
        train_set_x, train_set_y = data_reader.load_one_sample()
        yield nd.array(train_set_x), nd.array([np.argmax(train_set_y)])


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y).mean().asscalar()


train_x_file_list = []
train_y_file_list = []
with open('train_list.txt', 'r') as ft:
    lines = ft.readlines()
    for line in lines:
        train_x_file_list.append(line.replace('\n', '')+'_array')
        train_y_file_list.append(line.replace('\n', '')+'_label')


valid_x_file_list = []
valid_y_file_list = []
with open('valid_list.txt', 'r') as fv:
    lines = fv.readlines()
    for line in lines:
        valid_x_file_list.append(line.replace('\n', '')+'_array')
        valid_y_file_list.append(line.replace('\n', '')+'_label')


n_ins = 1024
n_outs = 7

train_data_reader = dp.ListDataProvider(x_file_list=train_x_file_list, y_file_list=train_y_file_list,
                                        n_ins=n_ins, n_outs=n_outs, shuffle=True)
valid_data_reader = dp.ListDataProvider(x_file_list=valid_x_file_list, y_file_list=valid_y_file_list,
                                        n_ins=n_ins, n_outs=n_outs, shuffle=False)
train_data_reader.reset()
valid_data_reader.reset()


ctx = mx.gpu()
net = gru.TuringTTS(num_hidden=512)
net.initialize(ctx=ctx, init=init.Xavier())
net.hybridize()


num_epochs = 200
learning_rate = 0.001
save_model_prefix = './ckp/emo'
batch_size = 1


SCELoss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'momentum': 0.9})


prev_time = datetime.datetime.now()
for epoch in range(num_epochs):
    train_data_reader.reset()
    valid_data_reader.reset()

    train_loss = 0.0
    train_accy = 0.0
    for data, label in data_iter(train_data_reader):
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = SCELoss(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_accy += accuracy(output, label)

    valid_loss = 0.0
    valid_accy = 0.0
    for data, label in data_iter(valid_data_reader):
        label = label.as_in_context(ctx)
        output = net(data.as_in_context(ctx))
        loss = SCELoss(output, label)
        valid_loss += nd.mean(loss).asscalar()
        valid_accy += accuracy(output, label)

    epoch_str = ("Epoch %d. Train loss %f, Valid loss %f, Train accu %f, Valid accu %f, "
                 % (epoch, train_loss / train_data_reader.list_size, valid_loss / valid_data_reader.list_size,
                    train_accy / train_data_reader.list_size, valid_accy / valid_data_reader.list_size))

    curr_time = datetime.datetime.now()
    h, remainder = divmod((curr_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)

    print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
    prev_time = curr_time

    net.save_params('%s_gru-%03d.params' % (save_model_prefix, epoch))

    with open('mxnet_gru_train.log', 'a+') as lgf:
        lgf.write(epoch_str + time_str + ', lr ' + str(trainer.learning_rate) + '\n')





