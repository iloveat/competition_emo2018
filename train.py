# -*- coding: utf-8 -*-
import data_provider as dp
import numpy as np
import sru
import time
import cntk


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
learning_rate = 0.001
cntk.device.try_set_default_device(cntk.gpu(0))


train_data_reader = dp.ListDataProvider(x_file_list=train_x_file_list, y_file_list=train_y_file_list,
                                        n_ins=n_ins, n_outs=n_outs, shuffle=True)
valid_data_reader = dp.ListDataProvider(x_file_list=valid_x_file_list, y_file_list=valid_y_file_list,
                                        n_ins=n_ins, n_outs=n_outs, shuffle=False)
train_data_reader.reset()
valid_data_reader.reset()


rnn_model = sru.SruTTS(n_in=n_ins, n_out=n_outs, init_lr=learning_rate, momentum=0.5)
trainer = rnn_model.trainer


num_epochs = 200
save_model_prefix = './ckp/emo_'


for epoch in range(num_epochs):
    log_file = open('cntk_sru_train.log', 'a')
    begin_time = time.time()

    # train
    train_error_list = []
    while not train_data_reader.is_finish():
        xx_train, yy_train = train_data_reader.load_one_partition()
        trainer.train_minibatch({rnn_model.input: xx_train, rnn_model.label: yy_train})
        curr_train_loss = trainer.previous_minibatch_loss_average
        train_error_list.append(curr_train_loss)
    train_data_reader.reset()

    # validation
    valid_loss_list = []
    while not valid_data_reader.is_finish():
        xx_valid, yy_valid = valid_data_reader.load_one_partition()
        curr_valid_loss = trainer.test_minibatch({rnn_model.input: xx_valid, rnn_model.label: yy_valid})
        valid_loss_list.append(curr_valid_loss)
    valid_data_reader.reset()

    # save model
    trainer.save_checkpoint(save_model_prefix + str(epoch))
    curr_valid_loss = np.mean(np.asarray(valid_loss_list))
    curr_train_loss = np.mean(np.asarray(train_error_list))

    # rnn_model.learner.reset_learning_rate(cntk.learning_rate_schedule(learning_rate, cntk.UnitType.sample))

    message = 'Epoch:%04d, Lr:%f, Train:%f, Valid:%f, Time:%f' \
              % (epoch, learning_rate, curr_train_loss, curr_valid_loss, time.time()-begin_time)
    log_file.write(message+'\n')
    log_file.close()
    print message





