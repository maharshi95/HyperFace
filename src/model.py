import numpy as np
import tensorflow as tf

from config import Config
from hyperparams import Hyperparams
from nets import hyperface_alexnet
from utils import NetOutput, NetLosses


class HyperFaceModel(object):
    def __init__(self, H, C, load_model=False, model_load_path=None):
        # type: (Hyperparams, Config, bool, str) -> None

        self.H = H
        self.C = C
        self.model_save_path = C.models_path
        self.best_model_save_path = C.best_models_path
        self.model_load_path = model_load_path
        self.load_model = load_model

        self.writer = {
            'train': tf.summary.FileWriter(C.log_path['train']),
            'test': tf.summary.FileWriter(C.log_path['test']),
        }

    def build(self, session=None):
        if session is None:
            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True
            session = tf.InteractiveSession(config=self.config)

        self.session = session

        self._define_placeholders()
        self._define_network()
        self._define_losses()
        self._define_summaries()

        # self.loss = self.loss_detection
        # self.optimizer = tf.train.AdamOptimizer(1e-7).minimize(self.loss)
        self.optimizer = tf.train.MomentumOptimizer(1e-3, 0.9, use_nesterov=True)
        self.train_op = self.optimizer.minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=4)
        self.best_saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=4)

        if not self.load_model:
            self.session.run(tf.global_variables_initializer())

    def _define_placeholders(self):
        # Image Inputs [B, 227, 227, 3]
        self.inputs = tf.placeholder(tf.float32, [None, self.H.img_height, self.H.img_width, self.H.channel], name='input')

        # Ground Truth Labels
        self.detection_gt = tf.placeholder(tf.float32, [None, 1], name='detection_gt')
        self.landmarks_gt = tf.placeholder(tf.float32, [None, 42], name='landmarks_gt')
        self.visibility_gt = tf.placeholder(tf.float32, [None, 21], name='visibility_gt')
        self.pose_gt = tf.placeholder(tf.float32, [None, 3], name='pose_gt')
        self.gender_gt = tf.placeholder(tf.float32, [None, 1], name='gender_gt')

    def _define_network(self):
        if self.H.model_type == 'alexnet':
            net_output = hyperface_alexnet(self.inputs)  # (out_detection, out_landmarks, out_visibility, out_pose, out_gender)
        elif self.H.model_type == 'resnet':
            net_output = hyperface_alexnet(self.inputs)  # (out_detection, out_landmarks, out_visibility, out_pose, out_gender)
        else:
            raise Exception('%s: Invalid model_type in Hyperparams' % self.H.model_type)
        self.net_output = net_output

    def _define_losses(self):
        net_output = self.net_output
        self.loss_detection = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=net_output.detection, labels=self.detection_gt))

        detection_mask = tf.cast(tf.expand_dims(self.detection_gt, axis=1), tf.float32)
        vis_mask = tf.tile(tf.expand_dims(self.visibility_gt, axis=2), [1, 1, 2])
        visibility_mask = tf.reshape(vis_mask, [tf.shape(vis_mask)[0], -1])

        loss_landmarks = tf.square(net_output.landmarks - self.landmarks_gt)
        self.loss_landmarks = tf.reduce_mean(detection_mask * visibility_mask * loss_landmarks)

        loss_visibility = tf.square(net_output.visibility - self.visibility_gt)
        self.loss_visibility = tf.reduce_mean(detection_mask * loss_visibility)

        loss_pose = tf.square(net_output.pose - self.pose_gt)
        self.loss_pose = tf.reduce_mean(detection_mask * loss_pose)

        gender_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=net_output.gender, labels=self.gender_gt)
        self.loss_gender = tf.reduce_mean(detection_mask * gender_loss)

        self.loss = (
                self.H.weight_detect * self.loss_detection
                + self.H.weight_landmarks * self.loss_landmarks
                + self.H.weight_visibility * self.loss_visibility
                + self.H.weight_pose * self.loss_pose
                + self.H.weight_gender * self.loss_gender
        )

        detection_pred = tf.cast(net_output.detection >= 0.5, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(detection_pred, self.detection_gt), tf.float32))

    def _define_summaries(self):
        self.merged_summary = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.image('images', self.inputs, max_outputs=5),
            tf.summary.histogram('labels', self.detection_gt),
            tf.summary.scalar('det_loss', self.loss_detection),
            tf.summary.scalar('landmarks_loss', self.loss_landmarks),
            tf.summary.scalar('visibility_loss', self.loss_visibility),
            tf.summary.scalar('pose_loss', self.loss_pose),
            tf.summary.scalar('gender_loss', self.loss_gender),
        ])

    def train_step(self, inputs, outputs, global_step=None, log_summary=False):
        # type: (np.ndarray, NetOutput, int, bool) -> None
        losses = self.loss_detection, self.loss_landmarks, self.loss_visibility, self.loss_pose, self.loss_gender, self.loss
        _, merged_summary, loss_values = self.session.run([self.train_op, self.merged_summary, losses], feed_dict={
            self.inputs: inputs,
            self.detection_gt: outputs.detection,
            self.landmarks_gt: outputs.landmarks,
            self.visibility_gt: outputs.visibility,
            self.pose_gt: outputs.pose,
            self.gender_gt: outputs.gender,
        })
        if log_summary:  # Log summary to Tensorboard
            self.writer['train'].add_summary(merged_summary, global_step)

        return NetLosses(*loss_values)

    def val_step(self, inputs, outputs, global_step=None, log_summary=True):
        losses = self.loss_detection, self.loss_landmarks, self.loss_visibility, self.loss_pose, self.loss_gender, self.loss
        merged_summary, loss_values = self.session.run([self.merged_summary, losses], feed_dict={
            self.inputs: inputs,
            self.detection_gt: outputs.detection,
            self.landmarks_gt: outputs.landmarks,
            self.visibility_gt: outputs.visibility,
            self.pose_gt: outputs.pose,
            self.gender_gt: outputs.gender,
        })
        if log_summary:  # Log summary to Tensorboard
            self.writer['test'].add_summary(merged_summary, global_step)

        return NetLosses(*loss_values)

    def save_model(self, global_step):
        self.saver.save(self.session, self.C.models_path, global_step=global_step)

    def save_best_model(self, global_step):
        self.best_saver.save(self.session, self.C.best_models_path, global_step=global_step)