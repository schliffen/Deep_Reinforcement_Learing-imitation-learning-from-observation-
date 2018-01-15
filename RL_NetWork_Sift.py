import sys
sys.path.append("/home/vvglab/jay/gym")
import pickle as pk
import gym
import numpy as np
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
state_size = 768
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

expert_vid = np.load("numpy_vids/vid%d.npy" % 81)


expert_average = np.zeros(shape=state_size)
sift = cv2.xfeatures2d.SIFT_create(nfeatures=6)

for e in range(50000):
    env.reset()
    recorder.env = env
    recorded_frames = None
    ts = 1
    cp_state = None
    action = None

    # up to here
    for tt in range(500):
        if tt == 0:
            recorder.capture_frame()

            # print(recorder.last_frame.shape)
            raw_state = recorder.last_frame
            raw_state_gray = cv2.cvtColor(raw_state, cv2.COLOR_BGR2GRAY)
            raw_state_bin = cv2.threshold(raw_state_gray, 200, 255, cv2.THRESH_BINARY)[1]

            kp, cp_des = sift.detectAndCompute(raw_state_bin, None)
            cp_state = cp_des.ravel()[:state_size]
            action = agent.AI_Action(cp_state)
            continue

        recorder.capture_frame()

        # print(recorder.last_frame.shape)
        raw_state = recorder.last_frame
        raw_state_gray = cv2.cvtColor(raw_state, cv2.COLOR_BGR2GRAY)
        raw_state_bin = cv2.threshold(raw_state_gray,200, 255, cv2.THRESH_BINARY)[1]

        expert_bin = cv2.threshold(expert_vid[tt-1],200, 255, cv2.THRESH_BINARY)[1]
        _, exp_des = sift.detectAndCompute(expert_bin, None)
        expert_features = exp_des.ravel()[:state_size]

        _, cp_des = sift.detectAndCompute(raw_state_bin, None)
        next_cp_state = cp_des.ravel()[:state_size]

        _, _, done, _ = env.step(action)
        reward =  1000/(np.linalg.norm(next_cp_state-expert_features)+0.01)
        print("reward = ", reward)
        agent.Store_Transition(cp_state, action, reward)
        cp_state = next_cp_state
        action = agent.AI_Action(cp_state)


        # updating previous time steps
        ts += 1
        if done:
            train_loss = agent.REINFORCE_FC_Train()
            print("episode: {}/{}, score: {}".format(e, 100000, tt))
            break
    #   Episode has finished -
    time_steps_lived.append(ts)
    if (e % 500) == 0:
        c += 1
        file_to_save = "time_steps/e%d.p"%c
        pk.dump(time_steps_lived, open(file_to_save, "wb"))


recorder.close()


# random agent



