import tensorflow as tf
import numpy as np
import cv2
from image_preprocessing import crop_image


def leaky_relu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def convolution(in_img, out_img, f_size=5, stride=2, name="convolution"):
    # f_size = filter size
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [f_size, f_size, in_img.get_shape()[-1], out_img],
                                  initializer=tf.truncated_normal_initializer(0.02))
        # scope.reuse_variables()
        conv = tf.nn.conv2d(in_img, w, strides=[1, stride, stride, 1], padding="SAME")
        biases = tf.get_variable('biases', [out_img], initializer=tf.constant_initializer(0.0))
        scope.reuse_variables()
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def fully_connected_layer(in_img, out_, scope=None):
    shape = in_img.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], out_], tf.float32,
                 tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [out_],
                               initializer=tf.constant_initializer(0.0))
        return tf.matmul(in_img, matrix) + bias


class CartPoleLearn(object):
    def __init__(self):
        self.expert_average = None
        self.first_expert_encounter = True
        self.bias = 20
        self.N = 0
        self.enc_img = None
        self.expert_img = None

    def encode(self, img):
        imgshape = img.get_shape().as_list()
        self.batch_size = imgshape[1]
        conv1 = leaky_relu(convolution(img, 32, stride=1, name="conv_1"))
        conv2 = leaky_relu(convolution(conv1, 16, name="conv_2"))
        conv3 = leaky_relu(convolution(conv2, 16, stride=1, name="conv_3"))
        conv4 = leaky_relu(convolution(conv3, 8, name="conv_4"))
        fcn1 = leaky_relu(fully_connected_layer(tf.nn.dropout(tf.reshape(conv4, [self.batch_size, -1]), 0.5), 100, 'fcn_1'))
        fcn2 = leaky_relu(fully_connected_layer(tf.nn.dropout(fcn1, 0.5), 100, 'fcn_2'))
        self.enc_img = fcn2
        print('fcn2 shape -- ', fcn2.shape)
        return fcn2





    def build_encoder(self,  expert_imgs, source_imgs):
        # expert_imgs, source_imgs = all_images[0], all_images[1]
        source_shape = source_imgs.get_shape().as_list()
        s_image = source_imgs
        e_image = expert_imgs
       # with tf.variable_scope("conv") as scope:
            # s_img = self.encode(s_image)
            # scope.reuse_variables()
            # e_img = self.encode(e_image)

       #     s_img = s_image
       #     e_img = e_image
       #     self.expert_img = e_image
            # computing the running average
        #    if self.first_expert_encounter:
        #        self.expert_average = e_img
        #        self.N +=1
        #    else:
                # TODO - Imporove this by computing time based average
        #        self.expert_average += tf.subtract(e_img, self.expert_average)/self.N
        #        self.N += 1
        self.loss = .1 #tf.nn.l2_loss(s_img - e_img)
        return self.loss

    def get_reward(self, image):
        # self.enc_img = self.encode(image)
        self.enc_img = image
        self.reward = tf.norm(image) - tf.nn.l2_loss(image - self.expert_imgs)



# image = cv2.imread("temp/test1.PNG")
# img = crop_image(image)
# img = cv2.resize(img, (64, 36))
# img = np.reshape(img, (1, 36, 64, 3))
# img = np.concatenate((img,img))
# img_exp = np.concatenate((img, img, img, img, img))
#
# # img_exp = tf.convert_to_tensor(img_exp, dtype=np.float32)
# # img = tf.convert_to_tensor(img, dtype=np.float32)
# # print(img_exp.get_shape().as_list())
# config = tf.ConfigProto()
#
# with tf.Session(config=config) as sess:
#     img_shape = img.shape[0]
#     batch_size = img_shape
#     img_exp = img_exp[:batch_size]
#     clearn = CartPoleLearn()
#     # features = clearn.encode(img)  # features from the encoder
#     graph_input = tf.placeholder(tf.float32, (batch_size, 36, 64, 3,))
#     graph_input2 = tf.placeholder(tf.float32, (batch_size, 36, 64, 3,))
#
#     graph = clearn.build_encoder(graph_input, graph_input2)
#
#     # loss = sess.run(o)
#     optimizer = tf.train.AdamOptimizer(0.0005).minimize(clearn.loss)
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#
#     # saver.restore(sess, "train_weights/weights")
#     # data = np.load("numpy_vids_36_64/vid0.npy")
#     for i in range(100):
#         batch = [img, img]
#         op, loss = sess.run([optimizer, clearn.loss],{graph_input: img_exp, graph_input2:img_exp})
#         print("loss is "+ str(loss))
#     saver.save(sess, "train_weights/weights")
#     # print(type(out))
#
#     # print(sess.run(tf.report_uninitialized_variables()))
