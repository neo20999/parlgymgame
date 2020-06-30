import paddle.fluid as fluid
import parl
from parl import layers


class AtariModel(parl.Model):
    def __init__(self, act_dim, algo='DQN'):
        self.act_dim = act_dim
        self.conv1 = layers.conv2d(
            num_filters=32, filter_size=5, stride=1, padding=2, act='relu')
        self.conv2 = layers.conv2d(
            num_filters=32, filter_size=5, stride=1, padding=2, act='relu')
        self.conv3 = layers.conv2d(
            num_filters=64, filter_size=4, stride=1, padding=1, act='relu')
        self.conv4 = layers.conv2d(
            num_filters=64, filter_size=3, stride=1, padding=1, act='relu')

        self.algo = algo
        if algo == 'Dueling':
            self.fc1_adv = layers.fc(size=512, act='relu')
            self.fc2_adv = layers.fc(size=act_dim)
            self.fc1_val = layers.fc(size=512, act='relu')
            self.fc2_val = layers.fc(size=1)
        else:
            self.fc1 = layers.fc(size=act_dim)

    def value(self, obs):
        #obs = rgb2gray(obs)
        obs = obs / 255.0
        out = self.conv1(obs)
        out = layers.pool2d(
            input=out, pool_size=2, pool_stride=2, pool_type='max')
        out = self.conv2(out)
        out = layers.pool2d(
            input=out, pool_size=2, pool_stride=2, pool_type='max')
        out = self.conv3(out)
        out = layers.pool2d(
            input=out, pool_size=2, pool_stride=2, pool_type='max')
        out = self.conv4(out)
        out = layers.flatten(out, axis=1)

        if self.algo == 'Dueling':
            As = self.fc2_adv(self.fc1_adv(out))
            V = self.fc2_val(self.fc1_val(out))
            Q = As + (V - layers.reduce_mean(As, dim=1, keep_dim=True))
        else:
            Q = self.fc1(out)
        return Q