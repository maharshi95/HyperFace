import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from utils import NetOutput


def hyperface_alexnet(inputs, name='hyperface_alexnet'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
            conv1 = slim.conv2d(inputs, 96, [11, 11], 4, padding='VALID', scope='conv1')
            max1 = slim.max_pool2d(conv1, [3, 3], 2, padding='VALID', scope='max1')

            conv1a = slim.conv2d(max1, 256, [4, 4], 4, padding='VALID', scope='conv1a')

            conv2 = slim.conv2d(max1, 256, [5, 5], 1, scope='conv2')
            max2 = slim.max_pool2d(conv2, [3, 3], 2, padding='VALID', scope='max2')
            conv3 = slim.conv2d(max2, 384, [3, 3], 1, scope='conv3')

            conv3a = slim.conv2d(conv3, 256, [2, 2], 2, padding='VALID', scope='conv3a')

            conv4 = slim.conv2d(conv3, 384, [3, 3], 1, scope='conv4')
            conv5 = slim.conv2d(conv4, 256, [3, 3], 1, scope='conv5')
            pool5 = slim.max_pool2d(conv5, [3, 3], 2, padding='VALID', scope='pool5')

            concat_feat = tf.concat([conv1a, conv3a, pool5], 3)
            conv_all = slim.conv2d(concat_feat, 192, [1, 1], 1, padding='VALID', scope='conv_all')

            shape = int(np.prod(conv_all.get_shape()[1:]))
            fc_full = slim.fully_connected(tf.reshape(conv_all, [-1, shape]), 3072, scope='fc_full')

            fc_detection = slim.fully_connected(fc_full, 512, scope='fc_detection1')
            fc_landmarks = slim.fully_connected(fc_full, 512, scope='fc_landmarks1')
            fc_visibility = slim.fully_connected(fc_full, 512, scope='fc_visibility1')
            fc_pose = slim.fully_connected(fc_full, 512, scope='fc_pose1')
            fc_gender = slim.fully_connected(fc_full, 512, scope='fc_gender1')

            out_detection = slim.fully_connected(fc_detection, 1, scope='detection', activation_fn=None)
            out_landmarks = slim.fully_connected(fc_landmarks, 42, scope='landmarks', activation_fn=None)
            out_visibility = slim.fully_connected(fc_visibility, 21, scope='visibility', activation_fn=None)
            out_pose = slim.fully_connected(fc_pose, 3, scope='pose', activation_fn=None)
            out_gender = slim.fully_connected(fc_gender, 1, scope='gender', activation_fn=None)

        return NetOutput(out_detection, out_landmarks, out_visibility, out_pose, out_gender)


def hyperface_resnet(inputs, name='hyperface_resnet'):
    pass
