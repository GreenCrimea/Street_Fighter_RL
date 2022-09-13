import retro
import time

#python -m retro.import .   <<<   IMPORT ROMS

#open Street Fighter ROM
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')

#game loop
obs = env.reset()
done = False
for game in range(1):
    while not done:
        if done:
            obs = env.reset()
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        print(reward)
        