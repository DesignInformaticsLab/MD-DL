import tensorflow as tf
import math
import time
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util

def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    width = 1.
    # CONV
    net = tf_util.conv2d(input_image, int(64*width), [1,7], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, int(64*width), [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, int(64*width), [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, int(128*width), [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    points_feat1 = tf_util.conv2d(net, int(1024*width), [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    # MAX
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point,1], padding='VALID', scope='maxpool1')
    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    print(pc_feat1)
   
    # CONCAT 
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])
    
    # CONV 
    net = tf_util.conv2d(points_feat1_concat, 512, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv6')
    # net = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training, scope='conv6_1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv7')
    # net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training, scope='conv7_1')
    net = tf_util.conv2d(net, 32, [1,1], padding='VALID', stride=[1,1], scope='conv8')
    # net = tf_util.conv2d(net, 32, [1,1], padding='VALID', stride=[1,1], scope='conv8_1')
    net = tf_util.dropout(net, keep_prob=1.0, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 7, [1,1], padding='VALID', stride=[1,1],
                         activation_fn=None, scope='conv9') # xyz position
    net = tf.squeeze(net, [2])

    return net

def get_loss(pred, label):
    """ pred: B,N,13
        label: B,N """
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    loss_dist = tf.sqrt(tf.square(pred[:,:,0] - label[:,:,0])+tf.square(pred[:,:,1] - label[:,:,1])+tf.square(pred[:,:,2] - label[:,:,2]))
    loss_force = tf.sqrt(tf.square(pred[:,:,3] - label[:,:,3])+tf.square(pred[:,:,4] - label[:,:,4])+tf.square(pred[:,:,5] - label[:,:,5]))
    loss_energy = tf.sqrt(tf.square(pred[:,:,6] - label[:,:,6]))
    loss = loss_dist + loss_force + loss_energy
    return tf.reduce_mean(loss_dist), tf.reduce_mean(loss_force), tf.reduce_mean(loss_energy), tf.reduce_mean(loss)

if __name__ == "__main__":
    with tf.Graph().as_default():
        a = tf.placeholder(tf.float32, shape=(32,4096,9))
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={a:np.random.rand(32,4096,9)})
            print(time.time() - start)
