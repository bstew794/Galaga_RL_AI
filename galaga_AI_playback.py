import retro
import time

directory = './Progress_Reports/DQN_v2/movies'
filename = 'GalagaDemonsOfDeath-Nes-1Player.Level1-000003.bk2'

movie = retro.Movie(directory + '/' + filename)
movie.step()

env = retro.make(game=movie.get_game(), state=None, use_restricted_actions=retro.Actions.ALL, players=movie.players)
env.initial_state = movie.get_state()
env.reset()

while movie.step():
    keys = []

    for p in range(movie.players):
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, p))

    _obs, _rew, _done, _info = env.step(keys)

    env.render()
    time.sleep(0.01)

env.close()
