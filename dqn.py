import scipy.io as sio
import time
import os

import numpy as np
import gym

from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, BatchNormalization
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

nowtime = time.strftime("%y_%m_%d_%H",time.localtime())
ENV_NAME = 'discrete-action-uav-stable-2d-v0'

if not os.path.exists(ENV_NAME+'-'+nowtime):
    os.mkdir(ENV_NAME+'-'+nowtime)

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n



policy_list = ['maxG', 'minSNR', 'cline']
def Given_policy(env, policy, now):
    dx = env.SPplacex
    dy = env.SPplacey
    selected = np.where(env.G != 0)[0]
    if policy == 'maxG':
        num = np.argmax(env.G)
        aimx, aimy = dx[num] - env.placex, dy[num] - env.placey
    elif policy == 'minSNR':
        num = now
        if env.G[num] == 0:
            tnum = np.argmin(env.SNR[selected] + 10000)
            num = selected[tnum]
        aimx, aimy = dx[num] - env.placex, dy[num] - env.placey
    elif policy == 'random':
        num = now
        if env.G[env.cline] == 0:
            num = np.random.choice(selected)
        aimx, aimy = dx[num] - env.placex, dy[num] - env.placey
    elif policy == 'cline':
        num = env.cline
        if env.G[env.cline] == 0:
            num = np.random.choice(selected)
        aimx, aimy = dx[num] - env.placex, dy[num] - env.placey
    norm = np.sqrt(aimx ** 2 + aimy ** 2)
    aimx = aimx / norm
    aimy = aimy / norm
    if np.abs(env.v[0] + aimx * env.delta * env.amax) > env.Vmax:
        aimx = 0
    if np.abs(env.v[1] + aimy * env.delta * env.amax) > env.Vmax:
        aimy = 0
    aimx = np.around(np.abs(aimx*20) / 5) * np.sign(aimx) * 5
    aimy = np.around(np.abs(aimy*20) / 5) * np.sign(aimy) * 5
    action = env.a_sp.index([aimx, aimy])
    return action, num




# Next, we build a very simple model regardless of the dueling architecture
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(256,
          kernel_regularizer=regularizers.l2(0.1),
          bias_regularizer=regularizers.l2(0.1)))
model.add(Activation('relu'))
model.add(Dense(128,
          kernel_regularizer=regularizers.l2(0.1),
          bias_regularizer=regularizers.l2(0.1)))
model.add(Activation('relu'))
model.add(Dense(128,
          kernel_regularizer=regularizers.l2(0.1),
          bias_regularizer=regularizers.l2(0.1)))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())


memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-6), metrics=['mae'])
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

history = dqn.learning(env, Given_policy, policy_list, nb_steps=5e6, visualize=False, log_interval=1000, verbose=2,
                             nb_max_episode_steps=2000, imitation_leaning_time=1e16, reinforcement_learning_time=0)
sio.savemat(ENV_NAME+'-'+nowtime+'/fit.mat', history.history)
# After training is done, we save the final weights.
dqn.save_weights(ENV_NAME+'-'+nowtime+'/fit-weights.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
history = dqn.test(env, nb_episodes=5, visualize=False,nb_max_episode_steps=2000)
sio.savemat(ENV_NAME+'-'+nowtime+'/test.mat', history.history)