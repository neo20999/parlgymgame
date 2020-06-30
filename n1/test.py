import argparse
import gym
import paddle.fluid as fluid
import numpy as np
import os
import parl
from atari_agent import AtariAgent
from atari_model import AtariModel
from datetime import datetime
from replay_memory import ReplayMemory, Experience
from parl.utils import summary 
from parl.utils import  logger
from tqdm import tqdm
from utils import get_player


test_number = 1000

IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4
FRAME_SKIP = 4
UPDATE_FREQ = 4
GAMMA = 0.99
LEARNING_RATE = 3e-4
def run_evaluate_episode(env, agent):
    obs = env.reset()
    #obs = rgb2gray(obs)
    total_reward = 0
    while True:
        action = agent.predict(obs)
        test_env.render()
        obs, reward, isOver, info = env.step(action)
        total_reward += reward
        if isOver:
            break
    return total_reward

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--game_name', default='Phoenix-v0')
	test_env = get_player('Phoenix-v0',image_size=IMAGE_SIZE,context_len=CONTEXT_LEN)
	save_path = './dqn_model.ckpt'

	act_dim = test_env.action_space.n

	model = AtariModel(act_dim)
	algorithm = parl.algorithms.DQN(model, act_dim=act_dim, gamma=GAMMA)

	agent = AtariAgent(
        algorithm,
        act_dim=act_dim,
        start_lr=LEARNING_RATE,
        total_step=test_number,
        update_freq=UPDATE_FREQ)

	agent.restore(save_path)
	eval_rewards = []
	flag = 0

	while flag < test_number:
		eval_reward = run_evaluate_episode(test_env, agent)

		#eval_rewards.append(eval_reward)
		logger.info(
                "eval_agent done, (steps, eval_reward): ({}, {})".format(
                    flag, eval_reward))
		flag +=1