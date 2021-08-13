import gym


def preview(env):
    print(env.action_space)
    print(env.observation_space)

    observation = env.reset()
    for t in range(100):
        env.render()
        print(f'Observation: {observation}')
        action = env.action_space.sample()
        print(f'Action: {action}')
        observation, reward, done, info = env.step(action)
        print(f'Reward: {reward}')
        if done:
            break
    env.close()