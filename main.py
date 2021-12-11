import os

import agents
import enviroments
import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np

name = 'GalagaDemonsOfDeath-Nes'

scores_len = 25

agent = agents.DQNAgent(possible_actions=[1, 6, 7, 8], start_memory_len=64000, max_memory_len=4800000, start_epsilon=1,
                        learn_rate=0.003, scores_len=scores_len)

env = enviroments.make_env(name, agent)

last_few_avg = [0]
scores = deque(maxlen=scores_len)
max_score = 0

# agent.model.load_weights(agent.PARENT_FOLDER + '/weights/weights_at_325.hdf5')
# agent.model_target.load_weights(agent.PARENT_FOLDER + '/weights/weights_at_275.hdf5')
# agent.epsilon = 0.0

env.reset()

for i in range(501):
    time_steps = agent.total_time_steps
    time_e = time.time()
    score = enviroments.play_episode(name, env, agent, debug=False)
    scores.append(score)

    if score > max_score:
        max_score = score

    print('\nEpisode: ' + str(i))
    print('Steps: ' + str(agent.total_time_steps - time_steps))
    print('Duration: ' + str(time.time() - time_e))
    print('Score: ' + str(score))
    print('Max Score: ' + str(max_score))
    print('Epsilon: ' + str(agent.epsilon))

    if i % scores_len == 0 and i != 0:
        # clear last scores_len .bk2 "movies"
        for j in range(i - agent.scores_len, i):
            filenamePrefix = 'GalagaDemonsOfDeath-Nes-1Player.Level1-'
            filenameBody = str(j)
            filenameBody = filenameBody.rjust(6, '0')
            filenameSuffix = '.bk2'
            filename = filenamePrefix + filenameBody + filenameSuffix

            # path = os.path.join(agent.MOVIE_FOLDER, filename)
            os.remove(agent.PARENT_FOLDER + '/movies/' + filename)

        last_few_avg.append(sum(scores) / len(scores))
        plt.ylim([0, 10000])
        plt.plot(np.arange(0, i + 1, agent.scores_len), last_few_avg)
        plt.title("Average Score Per " + str(agent.scores_len) + " Episodes (Exploration Only)")
        plt.ylabel("Average Score")
        plt.xlabel("Episodes")
        plt.savefig(agent.PARENT_FOLDER + '/plots/' + 'last_' + str(agent.scores_len) + '_avg_plt_at_' + str(i))

        # if last_few_avg[-1] <= last_few_avg[-2] + 100:
        #    agent.epsilon += 0.05

        #    if agent.epsilon > 1:
        #        agent.epsilon = 0.95

        #    if agent.learn_rate > agent.MIN_LEARN_RATE:
        #        agent.learn_rate -= .0001

        weightsFilename = 'weights_at_' + str(i) + '.hdf5'

        agent.model.save_weights(agent.PARENT_FOLDER + '/weights/' + weightsFilename)
        print('\nWeights saved!')
