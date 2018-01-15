import sys
sys.path.append("/home/bayes/Academic/DeepRL/Project/IMLearnPG")
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
from keras import losses as loss

plt.style.use("seaborn")
g = tf.Graph()

# create environment for cart-pole problem
env = gym.make("CartPole-v1")
gym_state_size = env.observation_space.shape[0]
action_size = env.action_space.n  # we already know there are only 2 actions
done = False


# variables for our state configuration
state_size = 4
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


recorder = video_recorder.VideoRecorder(env, "/home/bayes/Academic/DeepRL/Project/IMLearnPG/numpy_vids/vid81.mp4")
time_steps_lived = []
#clearn = CartPoleLearn()
c=0

expert_vid = np.load("numpy_vids/vid%d.npy" % 81)

pca = PCA(n_components=4)
expert_vid = list(map(lambda x: normalize_img(crop_image_gray(x).ravel()), expert_vid))
expert_vid = pca.fit_transform(expert_vid)


expert_average = np.zeros(shape=state_size)
exp_e_frame = np.zeros(shape=state_size)






for e_frame in expert_vid:
    # e_frame_ = crop_image_gray(e_frame)
    expert_average += e_frame

expert_average = expert_average/len(expert_vid)

for e in range(50000):
    env.reset()
    recorder.env = env
    # gym_state = np.reshape(gym_state, [1, gym_state_size])
    cp_state = None
    action = None
    recorded_frames = None
    ts = 1

    # remove this later
    #r = random.randint(0, 82)
    # expert_vid = np.load("numpy_vids_36_64/vid%d.npy" % 81)
    # expert_vid = np.load("numpy_vids/vid%d.npy" % r)

    # exp_e_frame = np.zeros(shape=state_size)
    # for e_frame in expert_vid:
    #     e_frame_ = crop_image_gray(e_frame)
    #     exp_e_frame += (normalize_img(e_frame_.ravel())-exp_e_frame)/len(expert_vid)
    # expert_average += (exp_e_frame-expert_average)/(e+1)
    reward = 0
    # up to here
    for tt in range(500):
        if tt == 0:
            recorder.capture_frame()

            # print(recorder.last_frame.shape)
            raw_state = recorder.last_frame
            raw_state_gray = cv2.cvtColor(raw_state, cv2.COLOR_BGR2GRAY)
            f_state = crop_image_gray(raw_state_gray)  # state after filtering the image 170 X 400 X 3
            f_state = normalize_img(f_state.ravel())
            f_state = pca.transform(np.reshape(f_state, (1, f_state.size)))
#            f_state = cv2.resize(f_state, (64, 36))  # now should be 36 X 64 X 3
#            f_state = np.reshape(f_state, (1, 36, 64, 3))

            # oc_state = f_state.T[0].T  # state with only one channel (ignore other channels) 36 X 64

            # cp_state = oc_state.ravel()  # cartpole state - 1D vector of size 2304
            cp_state = np.zeros(shape=state_size)
            action = agent.AI_Action(cp_state)
            recorded_frames = f_state
            continue

        recorder.capture_frame()

        # print(recorder.last_frame.shape)
        raw_state = recorder.last_frame
        raw_state_gray = cv2.cvtColor(raw_state, cv2.COLOR_BGR2GRAY)
        f_state = crop_image_gray(raw_state_gray)  # state after filtering the image 170 X 400 (also grayed out)
        f_state = normalize_img(f_state.ravel())
        f_state = pca.transform(np.reshape(f_state, (1, f_state.size)))
        # f_state = cv2.resize(f_state, (64, 36))  # now should be 36 X 64 X 3

        # oc_state = f_state.T[0].T  # state with only one channel (ignore other channels) 36 X 64

        # next_cp_state = oc_state.ravel()  # cartpole state - 1D vector of size 2304
        # next_cp_state = np.reshape(next_cp_state, [1, state_size])
#        recorded_frames = np.concatenate((recorded_frames, f_state))
        _, _, done, _ = env.step(action)

        # TODO - redefine reward function here
        # reward, enc_img = reward_function(f_state)
        # print("reward = ", reward)

        # edits
        enc_img = normalize_img(f_state.ravel())
        #temp_rew = 1 #1/np.sqrt(np.sqrt(np.linalg.norm(enc_img - expert_vid[tt, :])))
        reward = 1/np.linalg.norm(enc_img- normalize_img(expert_vid[tt]))  + .1*tt
        # reward = np.dot(enc_img.T,np.log(np.divide(enc_img,expert_vid[tt,:]))) + np.dot(1-enc_img,np.log(np.divide(1-enc_img,1-expert_vid[tt,:])))
        #reward = 1 if not done else -100#temp_rew


        print("reward = ",reward)

        next_cp_state = enc_img
        agent.Store_Transition(cp_state, action, reward)
        # updating previous time steps
        cp_state = next_cp_state
        action = agent.AI_Action(next_cp_state)
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

