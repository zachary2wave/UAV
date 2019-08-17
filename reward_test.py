import numpy as np
import gym
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sys
import csv


ENV_NAME = 'uav-downlink-2d-v3'
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
###########################3


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
    return np.array([aimx, aimy, 1]), num


if __name__ == '__main__':
    # records = []
    # recordv = []
    # recorda = []
    # recorddone = []
    # recordcline = []
    # recordrate = []
    # recordreward = []
    # recordG = []
    # recordSP = []
    # recordobservation = []
    records = []
    recorda = []
    recordr = []
    recordd = []
    try:
        for loop in range(500):
            print(loop)
            S = env.reset()
            cline = env.cline
            fig = plt.figure(1)
            plt.ion()
            tarx = []
            tary = []
            recordtemp = []
            acrecord = []
            ac = 0
            while env.done == 0:
                action, cline = policy(env, 'maxG', cline)
                # action = [1,1,1]
                S_, reward, done, info = env.step(np.array(action))
                print('reward:', reward, 'Gleft=', env.G[cline], 'recordtemp', info['temp'])
                records.append(S)
                recorda.append(action)
                recordr.append(reward)
                recordtemp.append(info['temp'])
                ac +=reward
                acrecord.append(ac)
                recordd.append(done)
                # print(loop, 'place =', int(S[12]), int(S[13]), 'speed =', int(S[14]), int(S[15]),
                #        'action =', int(action[0]*30), int(action[1]*30), 'left=', int(np.sum(env.G)))
                # # record.append({"observation":S,"action":action,"reward":reward,"done":done})
                S = S_[:]
                # print('reward=', str(reward), 'left=', np.sum(env.G))
                # print(cline,env.cline)
                '''huatu '''
                # plt.cla()
                # SPx = [str(int(x)) for x in env.SPplacex]
                # SPy = [str(int(x)) for x in env.SPplacey]
                # intG = [str(int(x)) for x in env.G]
                # tarx.append(S[15])
                # tary.append(S[16])
                # plt.scatter(tarx, tary, c='r')
                # SP = plt.scatter(env.SPplacex, env.SPplacey)
                # LIN = plt.plot([env.placex, env.SPplacex[env.cline]], [env.placey, env.SPplacey[env.cline]],'--')
                # plt.text(env.SPplacex[0], env.SPplacey[0], str(0)+ '-G=' + intG[0])
                # plt.text(env.SPplacex[1], env.SPplacey[1], str(1)+ '-G=' + intG[1])
                # plt.text(env.SPplacex[2], env.SPplacey[2], str(2)+ '-G=' + intG[2])
                # plt.text(env.SPplacex[3], env.SPplacey[3], str(3)+ '-G=' + intG[3])
                # plt.text(env.SPplacex[4], env.SPplacey[4], str(4)+ '-G=' + intG[4])
                # plt.xlim(-400, 400)
                # plt.ylim(-400, 400)
                # plt.pause(0.1)
            plt.figure(1)
            plt.plot(np.arange(0, len(acrecord)), acrecord)
            plt.plot(np.arange(0, len(recordtemp)), [x*100000 for x in recordtemp])

            plt.show()
            pass
    except KeyboardInterrupt:
        sio.savemat('warmdata-for-' + ENV_NAME + '.mat',
                    {"observation": records, "action": recorda, "reward": recordr, "done": recordd})
        print('the data has been saved')
