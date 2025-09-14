from stable_baselines3.common.callbacks import BaseCallback

# Learn and Load the best saved model
class DebugCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            infos = self.locals.get('infos', [{}])
            actions = self.locals.get('actions', [])
            rewards = self.locals.get('rewards', [])
            
            if infos and 'valid_actions' in infos[0]:
                valid_actions = infos[0]['valid_actions']
                print(f"Step {self.n_calls}: Action={actions[0] if actions else 'N/A'}, Reward={rewards[0] if rewards else 'N/A'}")
                print(f"Valid actions count: {len(valid_actions)}")
        return True
        
def test_output(env, eval_env):

    valid_actions = env.unwrapped.get_valid_actions()

    print("Observation space keys:", list(env.observation_space.spaces.keys()))
    print("Board shape:", env.observation_space['board'].shape)
    print("Numeric shape:", env.observation_space['numeric'].shape)

    print("Training config:", env.unwrapped.config)
    print("Evaluation config:", eval_env.unwrapped.config)

    # Check for TimeLimit wrapper
    print("TimeLimit max steps:", getattr(env, '_max_episode_steps', 'None'))
    print("Eval TimeLimit max steps:", getattr(eval_env, '_max_episode_steps', 'None'))

    # Test what happens when we reach time limit
    test_obs = eval_env.reset()
    for i in range(15):  # Go beyond suspected time limit
        action = eval_env.action_space.sample()
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            done = True
        else:
            done = False
        print(f"Step {i}: Reward={reward}, Done={done}, Info={info}")
        if done:
            break