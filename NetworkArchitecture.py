import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import gfile
import imageio
import pickle
import scipy.misc
import sys
from IPython.display import HTML
import imageio
import argparse


def transform(image, resize_height=36, resize_width=64):
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        #         print("c", w.get_shape())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=tftrain,
                                            scope=self.name)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
    if with_w:
        return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
        return tf.matmul(input_, matrix) + bias


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        #         print("w", w.get_shape())
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


class ContextAEReach:
    def __init__(self, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 c_dim=3):
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

    def build(self, image, ablation_type):
        imgshape = image.get_shape().as_list()
        print(imgshape)
        self.output_height, self.output_width = imgshape[-3:-1]
        self.batch_size = imgshape[1]
        featsize = 1024
        srcimg = image[0]
        tgtimg = image[2]
        tgtctx = image[1]

        with tf.variable_scope("conv_context") as scope:
            tgtctx_h0 = lrelu(conv2d(tgtctx, self.df_dim, name='h0_conv'))
            tgtctx_h1 = lrelu(conv2d(tgtctx_h0, self.df_dim * 2, name='h1_conv'))
            tgtctx_h2 = lrelu(conv2d(tgtctx_h1, self.df_dim * 4, name='h2_conv'))
            tgtctx_h3 = lrelu(conv2d(tgtctx_h2, self.df_dim * 8, name='h3_conv'))
            tgtctx_h4 = lrelu(linear(tf.reshape(tgtctx_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            tgtctx_z = linear(tgtctx_h4, featsize, 'hz_lin')

        with tf.variable_scope("conv") as scope:
            srcimg_h0 = lrelu(conv2d(srcimg, self.df_dim, name='h0_conv'))
            srcimg_h1 = lrelu(conv2d(srcimg_h0, self.df_dim * 2, name='h1_conv'))
            srcimg_h2 = lrelu(conv2d(srcimg_h1, self.df_dim * 4, name='h2_conv'))
            srcimg_h3 = lrelu(conv2d(srcimg_h2, self.df_dim * 8, name='h3_conv'))
            print(srcimg_h3.get_shape())
            srcimg_h4 = lrelu(linear(tf.reshape(srcimg_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            srcimg_z = lrelu(linear(srcimg_h4, featsize, 'hz_lin'))

            scope.reuse_variables()

            tgtimg_h0 = lrelu(conv2d(tgtimg, self.df_dim, name='h0_conv'))
            tgtimg_h1 = lrelu(conv2d(tgtimg_h0, self.df_dim * 2, name='h1_conv'))
            tgtimg_h2 = lrelu(conv2d(tgtimg_h1, self.df_dim * 4, name='h2_conv'))
            tgtimg_h3 = lrelu(conv2d(tgtimg_h2, self.df_dim * 8, name='h3_conv'))
            tgtimg_h4 = lrelu(linear(tf.reshape(tgtimg_h3, [self.batch_size, -1]), featsize, 'h4_lin'))
            tgtimg_z = lrelu(linear(tgtimg_h4, featsize, 'hz_lin'))

        with tf.variable_scope("translate") as scope:
            trans_h0 = lrelu(linear(tf.concat([srcimg_z, tgtctx_z], 1), featsize, 'trans_h0'))
            trans_z = linear(trans_h0, featsize, 'trans_z')
            self.translated_z = trans_z

        with tf.variable_scope("deconv") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h / 2), int(s_h / 4), int(s_h / 8), int(s_h / 16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w / 2), int(s_w / 4), int(s_w / 8), int(s_w / 16)

            output_z_ = lrelu(linear(trans_z, self.gf_dim * 8 * s_h16 * s_w16, 'd_h0_lin'))
            output_h0 = tf.reshape(output_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            output_h1 = lrelu(deconv2d(tf.concat([output_h0, tgtctx_h3], 3),
                                       [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='d_h1'))
            output_h2 = lrelu(deconv2d(tf.concat([output_h1, tgtctx_h2], 3),
                                       [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='d_h2'))
            output_h3 = lrelu(deconv2d(tf.concat([output_h2, tgtctx_h1], 3),
                                       [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='d_h3'))
            output_h4 = deconv2d(tf.concat([output_h3, tgtctx_h0], 3),
                                 [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

            scope.reuse_variables()

            truthoutput_z_ = lrelu(linear(tgtimg_z, self.gf_dim * 8 * s_h16 * s_w16, 'd_h0_lin'))
            truthoutput_h0 = tf.reshape(truthoutput_z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            truthoutput_h1 = lrelu(deconv2d(tf.concat([truthoutput_h0, tgtctx_h3], 3),
                                            [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='d_h1'))
            truthoutput_h2 = lrelu(deconv2d(tf.concat([truthoutput_h1, tgtctx_h2], 3),
                                            [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='d_h2'))
            truthoutput_h3 = lrelu(deconv2d(tf.concat([truthoutput_h2, tgtctx_h1], 3),
                                            [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='d_h3'))
            truthoutput_h4 = deconv2d(tf.concat([truthoutput_h3, tgtctx_h0], 3),
                                      [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

        self.simloss = tf.reduce_mean((trans_z - tgtimg_z) ** 2) * 1e3
        mean, var = tf.nn.moments(tgtimg_z, axes=[0])
        print(var.get_shape())
        #         self.simloss /= tf.reduce_mean(var)
        print(tgtimg_z.get_shape())
        self.out = output_h4  # + contextimg#tf.nn.tanh(h4)
        self.out2 = truthoutput_h4
        self.recon1 = tf.nn.l2_loss(tgtimg - self.out)
        self.recon2 = tf.nn.l2_loss(tgtimg - self.out2)
        self.loss = self.recon1 + self.recon2 + self.simloss


if __name__ == "__main__":
    # TODO: add in an argparse

    vdata = np.load("video_data.npy")

    tf.reset_default_graph()
    #idim = (36, 64)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    tftrain = tf.placeholder(tf.bool, name='tftrain')
    batch_size = 100
    idim = (48, 48)
    tfinput = tf.placeholder(tf.float32, (3, batch_size) + idim + (3,), name='x')

    test = ContextAEReach()

    test.build(tfinput, None)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(test.loss)
    sess.run(tf.global_variables_initializer())
    allloss = []
    validloss = []
    itr = 0
    saver = tf.train.Saver()

    n = vdata.shape[1]
    nlen = vdata.shape[0]
    ntrain = int(0.8 * n)
    nvalid = n - ntrain
    validdata = vdata[:, ntrain:]
    traindata = vdata[:, :ntrain]
    while True:
        choicesrc = np.random.choice(ntrain, batch_size)
        choicetgt = np.random.choice(ntrain, batch_size)
        srcdata = traindata[np.arange(0, batch_size) % nlen, choicesrc]
        tgtdata = traindata[np.arange(0, batch_size) % nlen, choicetgt]
        tgtctx = traindata[0, choicetgt]
        batch = [srcdata, tgtctx, tgtdata]
        _, loss, sim, r1, r2 = sess.run([optimizer, test.loss, test.simloss, test.recon1, test.recon2],
                                        {tfinput: batch, learning_rate: 1e-4, tftrain: False, keep_prob: 0.5})
        if itr % 4 == 0:
            print(loss, sim, r1, r2)
            allloss.append(loss)

        if itr % 40 == 0:
            choicesrc = np.random.choice(nvalid, batch_size)
            choicetgt = np.random.choice(nvalid, batch_size)
            srcdata = validdata[np.arange(0, batch_size) % nlen, choicesrc]
            tgtdata = validdata[np.arange(0, batch_size) % nlen, choicetgt]
            tgtctx = validdata[0, choicetgt]
            batch = [srcdata, tgtctx, tgtdata]
            loss, sim, r1, r2 = sess.run([test.loss, test.simloss, test.recon1, test.recon2],
                                         {tfinput: batch, tftrain: False, keep_prob: 1.0})
            print(loss, sim, r1, r2, 'E')
            validloss.append(loss)
            saver.save(sess, 'ablation_' + str(itr))
        itr += 1

