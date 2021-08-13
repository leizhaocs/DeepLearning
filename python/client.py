import gym

env = gym.make("CartPole-v0")

observation = env.reset()

for _ in range(10000):
    env.render()
    mage = Image.fromarray(screen)
    image = image.resize((40, 90))
    image.show()


    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()

env.close()
