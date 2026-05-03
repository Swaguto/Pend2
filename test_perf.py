from RL_train import PendulumEnv
from stable_baselines3 import SAC
import numpy as np

def test_model():
    env = PendulumEnv(render=False)
    model = SAC.load('checkpoints/best/best_model')
    
    rewards = []
    for ep in range(5):
        obs, _ = env.reset(seed=ep)
        cum_rew = 0
        steps = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            cum_rew += reward
            steps += 1
            done = terminated or truncated
            
        print(f"Episode {ep}: Steps={steps}, Reward={cum_rew:.2f}, Terminated={terminated}")
        rewards.append(cum_rew)

    print("Average reward:", np.mean(rewards))

if __name__ == "__main__":
    test_model()
