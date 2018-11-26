"""
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here: http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is for an autonomous excavator simulating by Vortex Studio

Original author: Patrick Emami and edited by Bart Keulen

Author : Bukun Son
Start date : 2018.10.01
Last edit date : 2018.10.16
"""

import zmq
import time
import numpy as np
import datetime
import tensorflow as tf
import struct
#import matplotlib.pyplot as plt


from actor import ActorNetwork
from critic import CriticNetwork
from replaybuffer import ReplayBuffer
from explorationnoise import ExplorationNoise

# TRAINING PARAMETERS
# Learning rates actor and critic
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001

MAX_EPISODES = 3000         # Maximum number of episodes
MAX_STEPS_EPISODE = 349     # Maximum number of steps per episode
GAMMA = 0.99                # Discount factor
TAU = 0.001                 # Soft target update parameter

OBS_DIM = 5

# Size of replay buffer
BUFFER_SIZE = 1000000
MINIBATCH_SIZE = 50
# Exploration noise variables
NOISE_MEAN = 0
NOISE_VAR = 1
# Ornstein-Uhlenbeck variables

OU_THETA = 0.3
OU_MU = 0.
OU_SIGMA = 0.15
# Exploration duration
EXPLORATION_TIME = 1000

REWARD = []
QMAX =[]
# ================================
#    UTILITY PARAMETERS
# ================================
# Directory for storing gym results
DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/{}/vx_ddpg'.format(DATETIME)
# Directory for storing action network
RANDOM_SEED = 1234

PORT = "5555"

TIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
DATA_DIR = "C:\\Users\INRoL\Desktop\Data\Data1012_01"


# ================================
#    TENSORFLOW SUMMARY OPS
# ================================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar('Reward',  episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar('Qmax Value', episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ================================
#    TRAIN AGENT
# ================================
def train(sess, socket, actor, critic):
    # Set up summary ops
    summary_ops, summary_vars = build_summaries()

    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    f = open("D:\Doosan_simul\\20181117\Results\{}.txt".format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')), 'w')

    new_saver = tf.train.Saver()

    #ckpt = tf.train.get_checkpoint_state('D:\Doosan_simul\\20181115\Saver2')
    #if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    #new_saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(MAX_EPISODES):
        socket.send_string("reset %d" %i)
        s = socket.recv().split()
        for k in range (len(s)):
            s[k] = float(s[k])

        episode_reward = 0
        episode_ave_max_q = 0

        # OU noise, Nt generation
        noise = ExplorationNoise.ou_noise(OU_THETA, OU_MU, OU_SIGMA, MAX_STEPS_EPISODE)
        #noise = ExplorationNoise.exp_decay(noise, MAX_STEPS_EPISODE)

        for j in range(MAX_STEPS_EPISODE):

            # Add exploratory noise according to Ornstein-Uhlenbeck process to action
            # Decay exploration exponentially from 1 to 0 in EXPLORATION_TIME steps
            if i < EXPLORATION_TIME:
                a = actor.predict(np.reshape(s, (1, OBS_DIM))) + noise[j]
            else:
                a = actor.predict(np.reshape(s, (1, OBS_DIM)))

            socket.send_string("step %f %f %f %f" %(a[0][0], a[0][1], a[0][2], j))

            s2 = socket.recv().split()

            #print (s2)
            for k in range(len(s2)-1):
                s2[k] = float(s2[k])
            r = float(s2[len(s2)-3])
            payload = float(s2[len(s2) - 2])
            #terminal = float(s2[len(s2)-1])
            terminal = False

            s2 = s2[:len(s2)-3]
            # Gym 사용시에는 여기서 action 넣어서 one-step 진행 하고 s2(next state), reward, terminal, info 저장
            # DDPG algorithm 에서 execute action and observe reward and observe new state 부분

            #if i%10==0:
                #print (a)



            # Store transition in R
            replay_buffer.add(np.reshape(s, actor.state_dim),
                              np.reshape(a, actor.action_dim), r, terminal,
                              np.reshape(s2, actor.state_dim))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:

                #Sample a random mini-batch of N transitions from R

                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets = Q'
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    # If state is terminal assign reward only
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    # Else assgin reward + net target Q
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                episode_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                a_grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, a_grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            episode_reward += r

            if terminal==1 or j == MAX_STEPS_EPISODE-1:
                #summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: episode_reward[0], summary_vars[1]: episode_ave_max_q})

                #writer.add_summary(summary_str, i)
                #writer.flush()

                print ('Reward: %.2i' % int(episode_reward), ' | Episode', i, '| Qmax: %.4f' % (episode_ave_max_q / float(j)), 'Payload :', payload)
                f.write("%f %f %f %f\n" % (int(episode_reward), i, episode_ave_max_q / float(j), payload))

                REWARD.append(episode_reward)
                QMAX.append(episode_ave_max_q)

                if i%10 == 0 :
                    new_saver.save(sess, 'D:\Doosan_simul\\20181117\Saver/model_181117')

                break

def socket_connet():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:%s" % PORT)
    print("Tensorflow node turned on...")

    return socket

# ================================
#    MAIN
# ================================
def main(_):
    with tf.Session() as sess:

        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        state_dim = 5
        action_dim = 3
        action_bound = 1.0

        # Create actor and critic DNNs
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)
        critic = CriticNetwork(sess, state_dim, action_dim, action_bound, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        # Update actor and critic DNNs

        socket = socket_connet()

        socket.send_string("start")
        #time.sleep(0.5)
        print(socket.recv())


        train(sess, socket, actor, critic)

        saver = tf.train.Saver()
        saver.save(sess,'D:\Doosan_simul\\20181117\Saver/model_181117')

        #plt.figure(1)
        #plt.subplot(121)
        #plt.title('Reward')
        #plt.plot(REWARD)

        #plt.subplot(122)
        #plt.title('Qmax average')
        #plt.plot(QMAX)
        #plt.show()

if __name__ == '__main__':
    tf.app.run()