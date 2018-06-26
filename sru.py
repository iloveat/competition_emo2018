# -*- coding: utf-8 -*-
import cntk as C


class SruTTS(object):

    def __init__(self, n_in, n_out, init_lr, momentum):

        self.param1 = 256
        self.param2 = 128

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.init_lr = init_lr
        self.momentum = momentum

        self.input = C.sequence.input_variable(shape=(self.n_in))
        self.label = C.sequence.input_variable(shape=(self.n_out))

        self.create_parameters_layers()

        self.output = self.model(self.input)

        self.loss = self.loss_old(self.output, self.label)
        self.eval_err = self.loss_old(self.output, self.label)

        self.lr_s = C.learning_rate_schedule(self.init_lr, C.UnitType.sample)
        self.mom_s = C.momentum_schedule(self.momentum)
        self.learner = C.momentum_sgd(self.output.parameters, lr=self.lr_s, momentum=self.mom_s)
        self.trainer = C.Trainer(self.output, (self.loss, self.eval_err), [self.learner])


    def create_bias(self, number):

        self.list_bias = []

        for i in xrange(number):

            self.list_bias.append(C.parameter(shape=(self.param2), name='bias_' + str(i)))



    def create_parameters_layers(self):

        self.three_dnn = C.layers.Sequential([
            C.layers.Dense(self.param1, activation=C.tanh, name='dnn_three_1'),
            C.layers.Dense(self.param1, activation=C.tanh, name='dnn_three_2'),
            C.layers.Dense(self.param1, activation=C.tanh, name='dnn_three_3')])

        self.final_dnn = C.layers.Dense(self.n_out, name='dnn_final')


        self.dnn_1 = C.layers.Dense(8 * self.param2, bias=False, name='dnn_1')

        self.dnn_2 = C.layers.Dense(8 * self.param2, bias=False, name='dnn_2')

        self.dnn_3 = C.layers.Dense(8 * self.param2, bias=False, name='dnn_3')

        self.dnn_4 = C.layers.Dense(8 * self.param2, bias=False, name='dnn_4')

        self.create_bias(16)

    def bsru_layer(self, sru_1, index):

        f_1_f = C.sigmoid(sru_1[0 * self.param2 : 1 * self.param2] + self.list_bias[0 + index * 4])
        r_1_f = C.sigmoid(sru_1[1 * self.param2 : 2 * self.param2] + self.list_bias[1 + index * 4])
        c_1_f_r = (1 - f_1_f) * sru_1[2 * self.param2: 3 * self.param2]

        dec_c_1_f = C.layers.ForwardDeclaration('f_' + str(index))
        var_c_1_f = C.sequence.delay(dec_c_1_f, initial_state=0, time_step=1)
        nex_c_1_f = var_c_1_f * f_1_f + c_1_f_r
        dec_c_1_f.resolve_to(nex_c_1_f)

        h_1_f = r_1_f * C.tanh(nex_c_1_f) + (1 - r_1_f) * sru_1[3 * self.param2 : 4 * self.param2]

        f_1_b = C.sigmoid(sru_1[4 * self.param2 : 5 * self.param2] + self.list_bias[2 + index * 4])
        r_1_b = C.sigmoid(sru_1[5 * self.param2 : 6 * self.param2] + self.list_bias[3 + index * 4])
        c_1_b_r = (1 - f_1_b) * sru_1[6 * self.param2 : 7 * self.param2]

        dec_c_1_b = C.layers.ForwardDeclaration('b_' + str(index))
        var_c_1_b = C.sequence.delay(dec_c_1_b, time_step=-1)
        nex_c_1_b = var_c_1_b * f_1_b + c_1_b_r
        dec_c_1_b.resolve_to(nex_c_1_b)

        h_1_b = r_1_b * C.tanh(nex_c_1_b) + (1 - r_1_b) * sru_1[7 * self.param2 : 8 * self.param2]

        x = C.splice(h_1_f, h_1_b)

        return x

    def model(self, input):

        x = self.three_dnn(input)

        sru_1 = self.dnn_1(x)
        x = self.bsru_layer(sru_1, 0)

        sru_1 = self.dnn_2(x)
        x = self.bsru_layer(sru_1, 1)

        sru_1 = self.dnn_3(x)
        x = self.bsru_layer(sru_1, 2)

        sru_1 = self.dnn_4(x)
        x = self.bsru_layer(sru_1, 3)

        return self.final_dnn(x)



    def loss_old(self, output, label):

        length = C.sequence.reduce_sum(C.reduce_sum(output) * 0 + 1)

        return C.sequence.reduce_sum(C.reduce_sum(C.square(output - label))) / length
        # return C.sequence.reduce_sum(C.reduce_sum(C.abs(output - label))) / length

