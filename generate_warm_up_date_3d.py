import numpy as np
import gym
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sys
import csv


ENV_NAME = 'uav-downlink-3d-v0'
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
###########################3


def policy(env, policy, now):
    dx = env.SPplacex
    dy = env.SPplacey
    selected = np.where(env.G != 0)[0]
    print(selected)
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
    return np.array([aimx, aimy, 0, 1]), num


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
            while env.done == 0:
                action, cline = policy(env, 'minSNR', cline)
                S_, reward, done, info = env.step(np.array(action))
                records.append(S)
                recorda.append(action)
                recordr.append(reward)
                recordd.append(done)
                # print(loop, 'place =', int(S[12]), int(S[13]), 'speed =', int(S[14]), int(S[15]),
                #        'action =', int(action[0]*30), int(action[1]*30), 'left=', int(np.sum(env.G)))
                # # record.append({"observation":S,"action":action,"reward":reward,"done":done})
                S = S_[:]
                # print('reward=', str(reward), 'left=', np.sum(env.G))
                # print(cline,env.cline)
                plt.cla()
                SPx = [str(int(x)) for x in env.SPplacex]
                SPy = [str(int(x)) for x in env.SPplacey]
                intG = [str(int(x)) for x in env.G]
                tarx.append(S[15])
                tary.append(S[16])
                plt.scatter(tarx, tary, c='r')
                SP = plt.scatter(env.SPplacex, env.SPplacey)
                LIN = plt.plot([env.placex, env.SPplacex[env.cline]], [env.placey, env.SPplacey[env.cline]],'--')
                plt.text(env.SPplacex[0], env.SPplacey[0], str(0)+ '-G=' + intG[0])
                plt.text(env.SPplacex[1], env.SPplacey[1], str(1)+ '-G=' + intG[1])
                plt.text(env.SPplacex[2], env.SPplacey[2], str(2)+ '-G=' + intG[2])
                plt.text(env.SPplacex[3], env.SPplacey[3], str(3)+ '-G=' + intG[3])
                plt.text(env.SPplacex[4], env.SPplacey[4], str(4)+ '-G=' + intG[4])
                plt.xlim(-400, 400)
                plt.ylim(-400, 400)
                plt.pause(0.1)

                # fig = plt.figure(1)
                # plt.ion()
                # ax = Axes3D(fig)
                #
                # # print(tar)
                # ax.scatter3D(tar[:,12], tar[:,13], 100*np.ones_like(tar[:,0]), 'r')
                # ax.scatter3D(env.SPplacex, env.SPplacey, np.zeros_like(env.SPplacex))
                # ax.text(env.placex, env.placey, env.placez,
                #         'loc=' + str([env.placex, env.placey, env.placez]) + '\n'
                #         + 'V=' + str(env.v) + '\n' + 'a=' +str([action[0]*30, action[1]*30])
                #         )
                # ax.plot([env.placex, env.SPplacex[env.cline]], [env.placey, env.SPplacey[env.cline]],
                #         [env.placez, 0], '--')
                # ax.text((env.placex + env.SPplacex[env.cline]) / 2, (env.placey + env.SPplacey[env.cline]) / 2,
                #         (env.placez + 0) / 2, str(int(env.rate[env.cline]) / 1e6))
                # ax.text(env.SPplacex[0], env.SPplacey[0], 0,
                #         'loc=' + SPx[0] + SPy[0] + '\n'
                #         + 'G=' + intG[0] + '\n')
                # ax.text(env.SPplacex[1], env.SPplacey[1], 0,
                #         'loc=' + SPx[1] + SPy[1] + '\n'
                #         + 'G=' + intG[1] + '\n')
                # ax.text(env.SPplacex[2], env.SPplacey[2], 0,
                #         'loc=' + SPx[2] + SPy[2] + '\n'
                #         + 'G=' + intG[2] + '\n')
                # ax.text(env.SPplacex[3], env.SPplacey[3], 0,
                #         'loc=' + SPx[3] + SPy[3] + '\n'
                #         + 'G=' + intG[3] + '\n')
                # ax.text(env.SPplacex[4], env.SPplacey[4], 0,
                #         'loc=' + SPx[4] + SPy[4] + '\n'
                #         + 'G=' + intG[4] + '\n')
                # # for cline in range(env.NUAV):
                # #     ax.text(env.SPplacex[cline], env.SPplacey[cline], 0,
                # #             'loc=' + str(env.SPplacex[cline]) + str(env.SPplacex[cline]) + '\n'
                # #             + 'G=' + str(env.G[cline]) + '\n')
                # ax.set_xlim(-400, 400)
                # ax.set_ylim(-400, 400)
                # ax.set_zlim(0, 150)
                # plt.pause(0.00001)
    # sio.savemat('warmdata-for-' + ENV_NAME + '.mat',
    #             {"observation": records, "action": recorda, "reward": recordr, "done": recordd})
    except KeyboardInterrupt:
        sio.savemat('warmdata-for-' + ENV_NAME + '.mat',
                    {"observation": records, "action": recorda, "reward": recordr, "done": recordd})
        print('the data has been saved')
    #     sio.savemat('warmdata.mat', {'s': records, 'v': recordv, 'a': recorda,
    #                              'SP': [env.SPplacex, env.SPplacey],
    #                              'cline':recordcline, 'G':recordG  })
    #