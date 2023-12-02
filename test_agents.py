import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers

def get_actor_critic_model(input_shape, actor_out_shape, hidden_size):
    inputs = layers.Input(shape=input_shape)
    hidden = layers.Dense(hidden_size, activation='relu')(inputs)
    actor = layers.Dense(actor_out_shape, activation='softmax', name='actor')(hidden)
    critic = layers.Dense(1, activation='linear', name='critic')(hidden)

    return keras.Model(inputs=inputs, outputs=[actor,critic])

env = gym.make("CartPole-v1", render_mode='human') 
# env = gym.make("LunarLander-v2")

states_shape = env.observation_space.shape[0]
n_actions = env.action_space.n

actor_critic = get_actor_critic_model(input_shape=states_shape,
                                     actor_out_shape=n_actions,
                                     hidden_size=32
)
actor_critic.load_weights("checkpoints/ac_cartpole.h5")

max_test_steps = 200
step_count = 0
done = 0

state, _ = env.reset()
while not done:
    env.render()
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)
    actions_prob, _ = actor_critic(state)
    action = np.random.choice(n_actions, p=np.squeeze(actions_prob))
    state, reward, done, _, _ = env.step(action)
    step_count += 1
    if step_count > max_test_steps:
        break
env.close()