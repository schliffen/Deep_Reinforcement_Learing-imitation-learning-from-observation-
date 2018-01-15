import gym
from gym import wrappers
import os
import argparse
import numpy as np
from REINFORCE import REINFORCE

ENV_NAME = 'LunarLander-v2' # name of the environment
RENDER = True # show game
RAND_SEED = 0 # random seed for generating psudorandom numbers
LOG_DIR = 'log/' # path for saving the training log
TEST_DIR = 'test/' # path for saving the test log
RECORD_DIR = 'record/' # path for saving the env record
RECORD_FILENAME = ENV_NAME + '-experiment'
EPISODE_MAX = 5000 # maximum number of episodes
API_KEY = '' # API key for submission

env = gym.make(ENV_NAME)
env.seed(RAND_SEED)


def Train_Model(env = env):
    
    # Initialize AI for training
    num_actions = env.action_space.n
    num_features = env.observation_space.shape[0]
    AI_player = REINFORCE(num_actions = num_actions, num_features = num_features)
    

    # AI starts training
    episode = 0
    while episode < EPISODE_MAX:
        # Keep training until episode >= EPISODE_max
        observation = env.reset()
        t = 0
        episode_reward = 0

        while True:  
            # Keep training until termination
            if RENDER:
                env.render() # Turn on visualization
            action = AI_player.AI_Action(observation = observation)
            observation_next, reward, done, info = env.step(action)
            AI_player.Store_Transition(observation = observation, action = action, reward = reward)
            observation = observation_next
            t += 1
            episode_reward += reward

            if done:
                # Train on episode
                train_loss = AI_player.REINFORCE_FC_Train()

                # Print episode information
                print("#" * 50)
                print("Episode: %d" % episode)
                print("Episode reward: %f" % episode_reward)
                print("Episode finished after {} timesteps".format(t+1))
                print("Total time step: %d" % t)                
                
                episode += 1
                AI_player.episode += 1

                break

    # Close environment
    env.close()

def Test_Model(env = env):

    # Initialize AI for test
    num_actions = env.action_space.n
    num_features = env.observation_space.shape[0]
    AI_player = REINFORCE(num_actions = num_actions, num_features = num_features)
    AI_player.REINFORCE_FC_Restore()
    
    # AI starts training
    episode = 0

    while True:
        # Keep testing until hitting 'ctrl + c'
        observation = env.reset()
        t = 0
        episode_reward = 0
        while True:  
            # Keep training until termination
            env.render() # Turn on visualization
            action = AI_player.AI_Action(observation = observation)
            observation, reward, done, info = env.step(action)
            t += 1
            episode_reward += reward

            if done:
                # Print episode information
                print("#" * 50)
                print("Episode: %d" % episode)
                print("Episode reward: %f" % episode_reward)
                print("Episode finished after {} timesteps".format(t+1))
                print("Total time step: %d" % t)
                                  
                episode += 1
                break
                
    # Close environment
    env.close()
    
def Upload():

    # Upload training record
    gym.upload(RECORD_DIR + RECORD_FILENAME, api_key = API_KEY)
    
def main():

    parser = argparse.ArgumentParser(description = 'Designate AI mode')
    parser.add_argument('-m','--mode', help = 'train / test / upload', required = True)
    args = vars(parser.parse_args())

    if args['mode'] == 'train':
        Train_Model(env = env)
    elif args['mode'] == 'test':
        Test_Model(env = env)
    elif args['mode'] == 'upload':
        Upload()  
    else:
        print('Please designate AI mode.')

if __name__ == '__main__':

    main()
