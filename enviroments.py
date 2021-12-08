import retro
import preprocess_frame as ppf
import numpy as np


def init_new_game(name, env, agent):
    initial_state = env.reset()
    starting_frame = ppf.resize_frame(initial_state, agent.STATE_DIM_1, agent.STATE_DIM_2)

    dummy_action = 1
    dummy_reward = 0
    dummy_done = False

    for i in range(3):
        agent.memory.add_xp(starting_frame, dummy_reward, dummy_action, dummy_done)


def make_env(name, agent):
    env = retro.make(name, state='1Player.Level1', record=(agent.PARENT_FOLDER + '/movies'))
    return env


def take_step(name, env, agent, score, debug):
    input_action = np.zeros(len(env.buttons))
    input_action[agent.memory.actions[-1]] = 1
    next_frame, next_frame_reward, next_done_flag, info = env.step(input_action)

    if info['lives'] < agent.lives:
        agent.lives -= 1
        agent.idle_steps = 0
    elif info['lives'] > agent.lives:
        agent.lives += 1
    else:
        if next_frame_reward == 0:
            agent.idle_steps += 1
        else:
            agent.idle_steps = 0

    next_frame = ppf.resize_frame(next_frame, agent.STATE_DIM_1, agent.STATE_DIM_2)
    new_state = [agent.memory.frames[-3], agent.memory.frames[-2], agent.memory.frames[-1], next_frame]
    new_state = np.moveaxis(new_state, 0, 2) / 255
    new_state = np.expand_dims(new_state, 0)

    next_action = agent.get_action(new_state)

    if agent.idle_steps >= agent.MAX_IDLE_STEPS:
        next_done_flag = True

    if next_done_flag:
        agent.memory.add_xp(next_frame, next_frame_reward, next_action, next_done_flag)
        return info['score'], True

    agent.memory.add_xp(next_frame, next_frame_reward, next_action, next_done_flag)

    if debug:
        env.render()

    return info['score'], False


def play_episode(name, env, agent, debug=False):
    init_new_game(name, env, agent)
    agent.t = 0
    agent.idle_steps = 0
    agent.lives = 2
    score = 0

    while True and agent.idle_steps < agent.MAX_IDLE_STEPS:
        score, done = take_step(name, env, agent, score, debug)

        if done:
            break

        agent.t += 1
        agent.total_time_steps += 1

    if len(agent.memory.frames) > agent.start_memory_len:
        agent.learn(debug)

    return score