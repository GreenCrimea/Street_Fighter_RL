import os
import retro
import numpy as np
from gym.spaces import MultiBinary, Box 
from gym import Env 
import cv2
import optuna
from stable_baselines3 import PPO 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback


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

#env = StreetFighter()  

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

LOG_DIR = './logs/'
OPT_DIR = './opt/'

#return tested hyper parameters
def optimize_ppo(trial):
    return {    
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1e-5),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    }

#train and return mean reward
def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial)

        #create environment
        env = StreetFighter()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

        #create model
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=100000)

        #evaluate model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=25)
        env.close()

        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_BEST'.format(trial.number))
        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:
        return -1000

study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=100, n_jobs=1)

print(study.best_params)
'''
best_params = 

CHECKPOINT_DIR = './train/'

#SAVE MODEL
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

callback = TrainAndLoggingCallback(check_freq=50000, save_path=CHECKPOINT_DIR)


#TRAIN MODEL
episodes = 10

#create environment
env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

#create model
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **best_params)

#load weights
model.load(os.path.join(OPT_DIR, 'trial__BEST'))

for e in range(episodes):
    model.learn(total_timesteps=500000, callback=callback)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=25)
    print('================')
    print(f'EPISODE {e} DONE\n')
    print(f'mean reward: {mean_reward}\n')
    '''
