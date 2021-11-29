from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from agent_memory import Memory
import numpy as np
import random


class DQNAgent:
    def __init__(self, possible_actions, start_memory_len, max_memory_len, start_epsilon, learn_rate, scores_len=10,
                 start_lives=2, debug=False):

        # Dimensions we reduce the input image down to to simplify training
        self.STATE_DIM_1 = 112
        self.STATE_DIM_2 = 120
        self.STATE_DIM_3 = 4

        self.t = 0  # current time step
        self.MAX_STEPS = 4000  # Max length of an episode

        self.memory = Memory(max_memory_len)
        self.possible_actions = possible_actions
        self.epsilon = start_epsilon
        self.epsilon_decay = 50
        self.epsilon_min = 0.01
        self.gamma = .9
        self.learn_rate = learn_rate
        # self.MIN_LEARN_RATE = .0001
        self.scores_len = scores_len
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.lives = start_lives
        self.start_memory_len = start_memory_len
        self.learns = 0
        self.batch_size = 16
        self.total_time_steps = 0

        self.PARENT_FOLDER = './DQN'

    def _build_model(self):
        inputs = Input((self.STATE_DIM_1, self.STATE_DIM_2, self.STATE_DIM_3))
        conv1 = Conv2D(filters=32, kernel_size=8, strides=2, data_format="channels_last", activation='swish',
                       kernel_initializer=tf.keras.initializers.GlorotUniform())(inputs)
        batchNorm1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=64, kernel_size=4, strides=2, data_format="channels_last", activation='swish',
                       kernel_initializer=tf.keras.initializers.GlorotUniform())(batchNorm1)
        batchNorm2 = BatchNormalization()(conv2)
        conv3 = Conv2D(filters=64, kernel_size=3, strides=2, data_format="channels_last", activation='swish',
                       kernel_initializer=tf.keras.initializers.GlorotUniform())(batchNorm2)
        batchNorm3 = BatchNormalization()(conv3)
        flatten_conv = Flatten()(batchNorm3)
        fc1 = Dense(512, activation='swish', kernel_initializer=tf.keras.initializers.GlorotUniform())(flatten_conv)
        q_values = Dense(len(self.possible_actions), activation=None)(fc1)
        optimizer = Adam(self.learn_rate)
        model = tf.keras.Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()

        print('\nAgent Initialized\n')

        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions, 1)[0]

        a_index = np.argmax(self.model.predict(state))

        return self.possible_actions[a_index]

    def _index_valid(self, index):
        if self.memory.done_flags[index - 3] or self.memory.done_flags[index - 2] or self.memory.done_flags[
            index - 1] or \
                self.memory.done_flags[index]:

            return False

        else:
            return True

    def learn(self, debug=False):
        states = []
        next_states = []
        actions_taken = []
        next_rewards = []
        next_done_flags = []

        while len(states) < self.batch_size:
            index = np.random.randint(4, len(self.memory.frames) - 1)

            if self._index_valid(index):
                state = [self.memory.frames[index - 3], self.memory.frames[index - 2], self.memory.frames[index - 1],
                         self.memory.frames[index]]
                state = np.moveaxis(state, 0, 2) / 255
                next_state = [self.memory.frames[index - 2], self.memory.frames[index - 1], self.memory.frames[index],
                              self.memory.frames[index + 1]]
                next_state = np.moveaxis(next_state, 0, 2) / 255

                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index + 1])
                next_done_flags.append(self.memory.done_flags[index + 1])

        labels = self.model.predict(np.array(states))
        next_state_values = self.model_target.predict(np.array(next_states))

        for i in range(self.batch_size):
            action_index = self.possible_actions.index(actions_taken[i])
            labels[i][action_index] = next_rewards[i] + (not next_done_flags[i]) * self.gamma * max(
                next_state_values[i])

        self.model.fit(np.array(states), labels, batch_size=self.batch_size, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon / self.epsilon_decay

        self.learns += 1

        if self.learns % 10 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')


class DDQNAgent:
    def __init__(self, possible_actions, start_memory_len, max_memory_len, start_epsilon, learn_rate, scores_len=10,
                 start_lives=2, debug=False):

        # Dimensions we reduce the input image down to to simplify training
        self.STATE_DIM_1 = 112
        self.STATE_DIM_2 = 120
        self.STATE_DIM_3 = 4

        self.t = 0  # current time step
        self.MAX_STEPS = 4000  # Max length of an episode

        self.memory = Memory(max_memory_len)
        self.possible_actions = possible_actions
        self.epsilon = start_epsilon
        self.epsilon_decay = 50
        self.epsilon_min = .01
        self.gamma = 0.9
        self.learn_rate = learn_rate
        # self.MIN_LEARN_RATE = .0001
        self.scores_len = scores_len
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.lives = start_lives
        self.start_memory_len = start_memory_len
        self.learns = 0
        self.batch_size = 16
        self.total_time_steps = 0

        self.PARENT_FOLDER = './DDQN'

    def _build_model(self):
        inputs = Input((self.STATE_DIM_1, self.STATE_DIM_2, self.STATE_DIM_3))
        conv1 = Conv2D(filters=32, kernel_size=8, strides=2, data_format="channels_last", activation='swish',
                       kernel_initializer=tf.keras.initializers.GlorotUniform())(inputs)
        batchNorm1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=64, kernel_size=4, strides=2, data_format="channels_last", activation='swish',
                       kernel_initializer=tf.keras.initializers.GlorotUniform())(batchNorm1)
        batchNorm2 = BatchNormalization()(conv2)
        conv3 = Conv2D(filters=64, kernel_size=3, strides=2, data_format="channels_last", activation='swish',
                       kernel_initializer=tf.keras.initializers.GlorotUniform())(batchNorm2)
        batchNorm3 = BatchNormalization()(conv3)
        flatten_conv = Flatten()(batchNorm3)
        fc1 = Dense(512, activation='swish', kernel_initializer=tf.keras.initializers.GlorotUniform())(flatten_conv)
        q_values = Dense(len(self.possible_actions), activation=None)(fc1)
        optimizer = Adam(self.learn_rate)
        model = tf.keras.Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()

        print('\nAgent Initialized\n')

        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions, 1)[0]

        a_index = np.argmax(self.model.predict(state))

        return self.possible_actions[a_index]

    def _index_valid(self, index):
        if self.memory.done_flags[index - 3] or self.memory.done_flags[index - 2] or self.memory.done_flags[
            index - 1] or \
                self.memory.done_flags[index]:

            return False

        else:
            return True

    def learn(self, debug=False):
        states = []
        next_states = []
        actions_taken = []
        next_rewards = []
        next_done_flags = []

        while len(states) < self.batch_size:
            index = np.random.randint(4, len(self.memory.frames) - 1)

            if self._index_valid(index):
                state = [self.memory.frames[index - 3], self.memory.frames[index - 2], self.memory.frames[index - 1],
                         self.memory.frames[index]]
                state = np.moveaxis(state, 0, 2) / 255
                next_state = [self.memory.frames[index - 2], self.memory.frames[index - 1], self.memory.frames[index],
                              self.memory.frames[index + 1]]
                next_state = np.moveaxis(next_state, 0, 2) / 255

                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index + 1])
                next_done_flags.append(self.memory.done_flags[index + 1])

        labels = self.model.predict(np.array(states))
        next_state_values = self.model.predict(np.array(next_states))
        next_state_values_t = self.model_target.predict(np.array(next_states))

        for i in range(self.batch_size):
            action_index = self.possible_actions.index(actions_taken[i])
            labels[i][action_index] = next_rewards[i] + (not next_done_flags[i]) * self.gamma * \
                next_state_values_t[i][np.argmax(next_state_values[i])]

        self.model.fit(np.array(states), labels, batch_size=self.batch_size, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon / self.epsilon_decay

        self.learns += 1

        if self.learns % 10 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')
