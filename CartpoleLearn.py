# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import json
from gym import wrappers
from REINFORCE import REINFORCE

EPISODES = 3000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.95  # exploration rate
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= (1-self.epsilon):
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
#        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def video_callable(ep_id):
    global render
    return ep_id and render


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    render = False
    # env = wrappers.Monitor(env, 'cart_videos', force=True, video_callable=video_callable)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # agent = DQNAgent(state_size, action_size)
    agent = REINFORCE(action_size, state_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    scores = []
    for e in range(EPISODES):
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        for tt in range(500):
            if render:
                pass
                env.render(); #time.sleep(0.05)
            action = agent.AI_Action(state)
            next_state, reward, done, _ = env.step(action)
            reward = 20 if not done else 5
            # next_state = np.reshape(next_state, [1, state_size])
            # agent.remember(state, action, reward, next_state, done)
            agent.Store_Transition(state, action, reward)
            state = next_state

            if done:
                print(state)
                train_loss = agent.REINFORCE_FC_Train()
                scores.append(tt)
                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, tt))
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    render = True
                else:
                    render = False
                break
        # if len(agent.memory) > batch_size:
        #     agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
    # agent.save("model2.h5")
    # with open("model.json", "w") as outfile:
    #     json.dump(agent.model.to_json(), outfile)



