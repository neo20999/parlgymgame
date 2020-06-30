
# 在原有的基础上增加了rgb2gray灰度处理的函数

import cv2
import gym
from atari_wrapper import FrameStack, MapState, FireResetEnv
import numpy as np

def rgb2gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def get_player(game_name,image_size,train=False,context_len=1):
    env = gym.make(game_name)
    #print(env)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, image_size))
    env = MapState(env, lambda im: rgb2gray(im))
    #print(env.observation_space.shape)
    if not train:
        # in training, context is taken care of in expreplay buffer
        env = FrameStack(env, context_len)
    return env