import argparse
import math
# import h5py
import numpy as np
import tensorflow as tf
import socket

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# import provider
import tf_util
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log_larger', help='Log dir [default: log_larger]')
parser.add_argument('--num_point', type=int, default=100, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=5000, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=50, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 7))
            labels_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 7))

            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            # pred += pointclouds_pl[:,:,:3]
            loss_dist, loss_force, loss_energy, loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'loss_dist': loss_dist,
               'loss_force':  loss_force,
               'loss_energy': loss_energy,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        train_acc = []
        test_acc = []
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_acc_i = train_one_epoch(sess, ops, train_writer)
            test_acc_i = eval_one_epoch(sess, ops, test_writer)
            train_acc += [train_acc_i]
            test_acc += [test_acc_i]
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
            np.save('train_acc', train_acc)
        print('training done')

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    # current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label)
    # data = np.load('../new_data_10k/training_data.npy').item()
    data = np.load('../data_1k/training_data.npy').item()
    potential_in = np.expand_dims(data['potential_in'], -1)
    current_data = np.concatenate([data['pos_in'],data['force_in'],potential_in], -1)
    potential_out =np.expand_dims(data['potential_out'], -1)
    current_label = np.concatenate([data['pos_out'],data['force_out'],potential_out], -1)
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    loss_dist_sum = 0
    loss_force_sum = 0
    loss_energy_sum = 0

    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        _, summary, step, loss_val, loss_dist_val, loss_force_val, loss_energy_val, pred_val = sess.run([ops['train_op'],
                                                                                                         ops['merged'],
                                                                                                         ops['step'],
                                                                                                         ops['loss'],
                                                                                                         ops['loss_dist'],
                                                                                                         ops['loss_force'],
                                                                                                         ops['loss_energy'],
                                                                                                         ops['pred']],
                                                                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val * BATCH_SIZE)
        loss_dist_sum += (loss_dist_val * BATCH_SIZE)
        loss_force_sum += (loss_force_val * BATCH_SIZE)
        loss_energy_sum += (loss_energy_val * BATCH_SIZE)

    acc = loss_dist_sum / float(total_seen / NUM_POINT)
    acc_force = loss_force_sum / float(total_seen / NUM_POINT)
    acc_energy = loss_energy_sum / float(total_seen / NUM_POINT)
    log_string('train mean loss: %f %f %f' % (acc, acc_force, acc_energy))
    return acc
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_seen = 0
    loss_sum = 0
    loss_dist_sum = 0
    loss_force_sum = 0
    loss_energy_sum = 0

    log_string('----')
    # current_data = test_data[:,0:NUM_POINT,:]
    # current_label = np.squeeze(test_label)
    # data = np.load('../new_data_10k/testing_data.npy').item()
    data = np.load('../data_1k/testing_data.npy').item()
    potential_in = np.expand_dims(data['potential_in'], -1)
    current_data = np.concatenate([data['pos_in'],data['force_in'],potential_in], -1)
    potential_out =np.expand_dims(data['potential_out'], -1)
    current_label = np.concatenate([data['pos_out'],data['force_out'],potential_out], -1)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, loss_dist_val, loss_force_val, loss_energy_val, pred_val = sess.run([ops['merged'],
                                                                                     ops['step'],
                                                                                     ops['loss'],
                                                                                     ops['loss_dist'],
                                                                                     ops['loss_force'],
                                                                                     ops['loss_energy'],
                                                                                     ops['pred']],
                                                                                    feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        loss_dist_sum += (loss_dist_val*BATCH_SIZE)
        loss_force_sum += (loss_force_val*BATCH_SIZE)
        loss_energy_sum += (loss_energy_val*BATCH_SIZE)

    acc = loss_dist_sum / float(total_seen/NUM_POINT)
    acc_force = loss_force_sum / float(total_seen/NUM_POINT)
    acc_energy = loss_energy_sum / float(total_seen/NUM_POINT)
    log_string('eval mean loss: %f %f %f' % (acc, acc_force, acc_energy))
    return acc

    def visual():
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        pt_idx = 30
        tt_pt = 5
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pred_val[0, pt_idx:pt_idx + tt_pt, 0], pred_val[0, pt_idx:pt_idx + tt_pt, 1],
                pred_val[0, pt_idx:pt_idx + tt_pt, 2], 'k.')
        ax.plot(current_label[start_idx:end_idx][0, pt_idx:pt_idx + tt_pt, 0],
                current_label[start_idx:end_idx][0, pt_idx:pt_idx + tt_pt, 1],
                current_label[start_idx:end_idx][0, pt_idx:pt_idx + tt_pt, 2], 'r.')
        ax.plot(current_data[start_idx:end_idx][0, pt_idx:pt_idx + tt_pt, 0],
                current_data[start_idx:end_idx][0, pt_idx:pt_idx + tt_pt, 1],
                current_data[start_idx:end_idx][0, pt_idx:pt_idx + tt_pt, 2], 'y.')
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)

if __name__ == "__main__":
    train()
    LOG_FOUT.close()


# np.savetxt('case1.txt',pred_val.reshape(50,300)) #testing case prediction, not sure
# np.savetxt('case2.txt',current_data[start_idx:end_idx,:,:3].reshape(50,300)) #initial, high force and energy
# np.savetxt('case3.txt',current_label[start_idx:end_idx].reshape(50,300)) #reference, low force and energy
