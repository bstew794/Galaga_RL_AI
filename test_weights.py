import os

import agents
import enviroments
import matplotlib.pyplot as plt
import time
import numpy as np

name = 'GalagaDemonsOfDeath-Nes'

scores_len = 25

agent = agents.DQNAgent(possible_actions=[1, 6, 7, 8], start_memory_len=64000, max_memory_len=4800000, start_epsilon=0,
                        learn_rate=0.0025, scores_len=scores_len)

env = enviroments.make_env(name, agent)

scores = [0]
max_score = 0
i = 0
directory = agent.PARENT_FOLDER + '/weights'

env.reset()

for file in os.scandir(directory):
    agent.model.load_weights(file)
    agent.model_target.load_weights(file)

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

    i += 1

plt.ylim([0, 10000])
plt.plot(np.arange(0, 300, 25), scores)
plt.title("Score Using Saved Weights per " + str(agent.scores_len) + " Episodes (Exploitation Only)")
plt.ylabel("Score")
plt.xlabel("Number of Training Episodes")
plt.savefig(agent.PARENT_FOLDER + '/test_plots/' + 'scores_plt')
