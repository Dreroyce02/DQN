import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import ImageGrab
import game_func as gf
import random
from collections import deque
import time

bbox = (504,170,1101,246)
epsilon = 1 #decrease to 0
epsilon_dec = .9
min_epsilon = 0
gamma = .9
max_mem = 50000
min_mem = 1000
minibatch_size = 64
update_every = 5

def choose_action(action_index, p):
    if action_index==0:
        p.jump()
    elif action_index==1:
        p.duck()

def get_state():
    return np.array(ImageGrab.grab(bbox=bbox).convert('L').getdata())

class DQN_Agent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_mem = deque(maxlen=max_mem)
        self.counter = 0
        self.player = gf.Player()

    def create_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(256, (3,3), input_shape=(597,76,1), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Conv2D(256, (3,3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64,activation='relu'))
        model.add(keras.layers.Dense(3,  activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def update_replay_mem(self, transition):
        self.replay_mem.append(transition)

    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, *state.shape)/255)[0]

    def train(self):
        global epsilon
        if len(self.replay_mem)<min_mem:
            return None
        print(epsilon)
        minibatch = random.sample(self.replay_mem, minibatch_size)
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        Y = []
        for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):
            if reward!=-10:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + gamma*max_future_q
                self.counter += 1
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            Y.append(current_qs)
        self.model.fit(np.array(X)/255, np.array(Y), batch_size=minibatch_size, verbose=0, shuffle=False)
        if self.counter > update_every:
            self.target_model.set_weights(self.model.get_weights())
            self.counter = 0

agent = DQN_Agent()
_  = 0        
while True:
    current_state = get_state()
    if np.random.random()>epsilon:
        action = np.argmax(agent.get_qs(current_state))
    else:
        action = random.randint(0, 2)
    choose_action(action, agent.player)
    new_state = get_state()
    reward = gf.reward_func()
    agent.update_replay_mem((current_state, action, reward, new_state))
    agent.train()       
    if epsilon > min_epsilon and _ == 999:
        _ = 0
        epsilon *= epsilon_dec
        epsilon = max(min_epsilon, epsilon)
    _ += 1
