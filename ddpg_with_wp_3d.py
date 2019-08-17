import numpy as np
import gym
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
import scipy.io as sio
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import time

'''
policy part
'''
policy_list = ['maxG','minSNR','random','cline']
def policy(env, policy, now):
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
    return np.array([aimx, aimy, 1, 1]), num
#%% the model part
'''
model part
'''

nowtime = time.strftime("%y_%m_%d_%H",time.localtime())
ENV_NAME = 'uav-downlink-2d-v3'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

nb_actions = env.action_space.shape[0]

# Next, we build a very simple model
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(64)(flattened_observation)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
xa = Dense(3)(x)
x_a = Activation('tanh')(xa)
xp = Dense(1)(x)
x_p = Activation('sigmoid')(xp)
x_out = Concatenate()([x_a, x_p])
actor = Model(inputs=[observation_input], outputs=[x_out])


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=50, nb_steps_warmup_actor=50, random_process=random_process,
                  gamma=1, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1., decay=0.9999), metrics=['mae'])

#%%
'''
the test before warm_up
'''
history = agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=1000)
# sio.savemat('test-before-train-' + ENV_NAME + '-' + nowtime + '.mat', history.history)
before = history.history['episode_reward']
'''
warm_up
'''
history = agent.warm_fit(env, policy, policy_list, nb_steps=3e6, visualize=False, log_interval=1000, verbose=2, nb_max_episode_steps=2000)
sio.savemat('warm-up-' + ENV_NAME + '-' + nowtime + '.mat', history.history)
agent.save_weights('ddpg_{}_weights_after_warm_start.h5f'.format(ENV_NAME), overwrite=True)
'''
the test after warm_up
'''
history = agent.test(env, nb_episodes=20, visualize=True, nb_max_episode_steps=1000)
sio.savemat('test-'+ENV_NAME+'-'+nowtime+'.mat',history.history)
after = history.history['episode_reward']
print('before training ', before)
print('after training ', after)

#%%
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
history = agent.fit(env, nb_steps=5e6, visualize=False, log_interval=1000, verbose=2, nb_max_episode_steps=1000)
sio.savemat('fit-'+ENV_NAME+'-'+nowtime+'.mat', history.history)
# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
history = agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=1000)
sio.savemat('test-final-'+ENV_NAME+'-'+nowtime+'.mat',history.history)
