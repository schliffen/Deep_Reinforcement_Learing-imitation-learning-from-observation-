import sys
sys.path.append("/home/vvglab/jay/gym")
import pickle as pk
import gym
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import time
import random
import cv2
from gym.monitoring import video_recorder
from image_preprocessing import crop_image, crop_image_gray, normalize_img
from ConvNet import *
from REINFORCE import REINFORCE
from sklearn.decomposition import PCA


plt.style.use("seaborn")
g = tf.Graph()

# create environment for cart-pole problem
env = gym.make("CartPole-v1")
gym_state_size = env.observation_space.shape[0]
action_size = env.action_space.n  # we already know there are only 2 actions
done = False


# variables for our state configuration
state_size = 12
agent = REINFORCE(action_size, state_size)
train_started = False
dummy_array = np.zeros(shape=(100, 36, 64, 3))


def reward_function(state):
    global train_started, recorded_frames
    if not train_started:
        return 1, np.zeros(shape=(state_size))
    else:
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 100
            r = random.randint(0, 82)
            graph_input = tf.placeholder(tf.float32, (batch_size, 36, 64, 3,))  # expert vid
            graph_input2 = tf.placeholder(tf.float32, (batch_size, 36, 64, 3,))  # recorded vid
            img_test = tf.placeholder(tf.float32, (1, 36, 64, 3,))
            clearn.build_encoder(graph_input, graph_input2)
            clearn.get_reward(img_test)
            saver = tf.train.Saver()
            saver.restore(sess, "train_weights/weights")
            sess.run(tf.global_variables_initializer())
            r, enc_img = sess.run([clearn.reward, clearn.enc_img],
                        {graph_input:dummy_array , graph_input2: dummy_array, img_test: state})
            reward = r
            # print("reward =", r)
        return reward, enc_img


recorder = video_recorder.VideoRecorder(env, "numpy_vids/vid81.mp4")
time_steps_lived = []
clearn = CartPoleLearn()
c=0

expert_vid = np.load("numpy_vids/vid%d.npy" % 1)


expert_average = np.zeros(shape=state_size)

for e in range(50000):
    env.reset()
    recorder.env = env
    recorded_frames = None
    ts = 1
    cp_state = None
    action = None

    # up to here
    try:
        for tt in range(500):
            if tt == 0:
                recorder.capture_frame()

                # print(recorder.last_frame.shape)
                raw_state = recorder.last_frame
                raw_state_gray = cv2.cvtColor(raw_state, cv2.COLOR_BGR2GRAY)
                agent_features = cv2.goodFeaturesToTrack(raw_state_gray, 12, 0.01, 7)
                midmat = agent_features[:,0]
                #smidmat = sorted(midmat, key=lambda x:x[1])
                #print('sorted midmat', smidmat)
                enc_img = (midmat[:,0] - np.min(midmat[:,0]))  /np.std(midmat[:,0]) # agent_features.ravel()
                cp_state = enc_img
                action = agent.AI_Action(cp_state)
                continue

            recorder.capture_frame()

            # print(recorder.last_frame.shape)
            raw_state = recorder.last_frame
            raw_state_gray = cv2.cvtColor(raw_state, cv2.COLOR_BGR2GRAY)
            expert_features = cv2.goodFeaturesToTrack(expert_vid[tt-1], 12, 0.01, 7)
            midmat2 = expert_features[:,0]
            midmat2 = (midmat2[:, 0] - np.min(midmat2[:, 0]))/ np.std(midmat2[:, 0])



            # reward = np.dot(cp_state, expert_features.ravel())/(np.linalg.norm(cp_state)*np.linalg.norm(expert_features.ravel()))

            _, _, done, _ = env.step(action)

            #print('the agent feature: ', np.sort(enc_img))
            #print('------------------------------------')
            #print('the expert feature: ', np.sort(midmat2))
            #print('------------------------------------')

            reward = 10**(-np.linalg.norm(np.sort(enc_img) - np.sort(midmat2)) + 2) -20


            print("reward = ", reward)
            agent.Store_Transition(cp_state, action, reward)

            n_agent_features = cv2.goodFeaturesToTrack(raw_state_gray, 12, 0.01, 7)
            midmat = n_agent_features[:,0]
            enc_img = (midmat[:,0] - np.min(midmat[:,0])) / np.std(midmat[:,0])
            cp_state = enc_img
            action = agent.AI_Action(cp_state)

            # updating previous time steps
            ts += 1
            if done:
                train_loss = agent.REINFORCE_FC_Train()
                print("episode: {}/{}, score: {}".format(e, 100000, tt))
                print('------------------------------------')
                break
        #   Episode has finished -
    except ValueError:
        print('ses. was not completed')
        continue
    time_steps_lived.append(ts)
    if (e % 500) == 0:
        c += 1
        file_to_save = "time_steps/e%d.p"%c
        pk.dump(time_steps_lived, open(file_to_save, "wb"))
    #     tf.reset_default_graph()
    #     with tf.Session() as sess:
    #         batch_size = 100
    #         start = random.randint(0, 493)
    #         r = random.randint(0, 82)
    #         expert_vid = np.load("numpy_vids_36_64/vid%d.npy" % r)
    #         expert_vid = expert_vid[:batch_size]
    #         frame_size_diff = batch_size -recorded_frames.shape[0]
    #         fill_in_frames = np.zeros(shape=(frame_size_diff, 36, 64, 3))
    #         recorded_frames = np.concatenate((recorded_frames,fill_in_frames))
    #
    #
    #         graph_input = tf.placeholder(tf.float32, (batch_size, 36, 64, 3,))  # expert vid
    #         graph_input2 = tf.placeholder(tf.float32, (batch_size, 36, 64, 3,))  # recorded vid
    #         img_test = tf.placeholder(tf.float32, (1, 36, 64, 3,))
    #         clearn.build_encoder(graph_input, graph_input2)
    #         clearn.get_reward(img_test)
    #         optimizer = tf.train.AdamOptimizer(0.0005).minimize(clearn.loss)
    #
    #         sess.run(tf.global_variables_initializer())
    #         saver = tf.train.Saver()
    #
    #         if train_started:
    #             saver.restore(sess, "train_weights/weights")
    #         for i in range(500):
    #             op, loss, r= sess.run([optimizer, clearn.loss, clearn.reward],
    #                                {graph_input: expert_vid, graph_input2: recorded_frames, img_test: f_state})
    #             if (i % 50) == 0:
    #                 print("loss is " + str(loss), "iteration", i)
    #
    #         saver.save(sess, "train_weights/weights")
    #         train_started = True

recorder.close()


# random agent

def random_agent():
    all_steps = []
    for e in range(100):
        state = env.reset()
        state = np.reshape(state, [1, gym_state_size])
        for tt in range(500):
            # env.render(); time.sleep(0.05)
            action = random.randrange(0,action_size)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, gym_state_size])
            state = next_state

            if done:
                all_steps.append(tt)
                # print("episode: {}/{}, score: {}".format(e, 100, tt))
                break
    return all_steps


def plot_random_agent():
    steps_ = random_agent()
    print(steps_)
    time = range(len(steps_))
    plt.plot(time, steps_)
    plt.title("Living time steps for random agent (Cartpole-V1)")
    plt.ylabel("Time Steps")
    plt.ylim(0, 500)
    plt.xlabel("episode")
    plt.show()

# plot_random_agent()

