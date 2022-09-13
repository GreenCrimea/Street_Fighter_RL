import retro
import time
import numpy as np
from gym.spaces import MultiBinary, Box 
from gym import Env 
import cv2


#python -m retro.import .   <<<   IMPORT ROMS

#setup environment - preprocess game data
class StreetFighter(Env):

    def __init__(self):
        super().__init__()

        #specify action and observation space
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)

        #start game
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)


    def step(self, action):
        
        #grab original ENV step
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)

        #create frame delta
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        #fix reward val (score_delta + enemy_health_delta - health_delta)
        reward = ((info['score'] - self.score) / 20) + abs(info['enemy_health'] - self.enemy_health) - abs(info['health'] - self.health)
        self.score = info['score']
        self.enemy_health = info['enemy_health']
        self.health = info['health']

        return frame_delta, reward, done, info



    def render(self):
        self.game.render()


    def reset(self):

        obs = self.game.reset()
        obs = self.preprocess(obs)

        #empty movement delta
        self.previous_frame = obs

        #empty score, health, enemy_health delta
        self.score = 0
        self.enemy_health = 0
        self.health = 0

        return obs


    def preprocess(self, observation):

        #grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

        #resize
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)

        #add color channel
        channel = np.reshape(resize, (84, 84, 1))

        return channel


    def close(self):
        self.game.close()

env = StreetFighter()  

#game loop
'''
obs = env.reset()
done = False
for game in range(1):
    while not done:
        if done:
            obs = env.reset()
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        #time.sleep(0.015)
        if reward > 0:
            print(reward)
        if reward < 0:
            print(reward)
'''

