
# coding: utf-8

# In[1]:


import tensorflow as tf
import gym
import numpy as np


# In[2]:


def test_environment(iterations):
    env = gym.make("MountainCar-v0")   
    
    print(env.action_space)
    print(env.observation_space)
    
    env.reset()
    
    for _ in range(iterations):
        env.render()
        env.step(env.action_space.sample()) # take a random action
    
    env.close()


# In[3]:


test_environment(10)


# In[4]:


env = gym.make("MountainCar-v0")
env = env.unwrapped
env.seed(1)


# In[5]:


dims = env.observation_space.shape

observation_size = dims[0]
action_size = env.action_space.n

max_episodes = 10000
learning_rate = 0.01  ##probar diferentes usando tensorflow
gamma = 0.95 #discount rate


# In[6]:


def calculate_discounted_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards -mean) / (std)
    
    return discounted_episode_rewards


# In[7]:


class PolicyGradient():
    
    def __init__(self):
        self.observations = tf.placeholder(shape=(None,observation_size),dtype = tf.float32, name = "observations")
        self.actions = tf.placeholder(shape = (None, action_size), dtype = tf.float32, name = "actions")
        self.discounted_rewards_ = tf.placeholder(shape = (None,), dtype=tf.float32, name = "discounted_episodes_rewards")

        #usamos esta variable para visualizar en tensorboard
        self.mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

        #3 fully connected layers
        self.fc1 = tf.layers.dense(inputs=self.observations, units=10,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),name="Fc1")

        self.fc2 = tf.layers.dense(inputs=self.fc1, units= action_size, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(), name = "Fc2")

        self.output = tf.layers.dense(inputs=self.fc2, units = action_size, activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name = "output")

        self.action_distribution = tf.nn.softmax(self.output, name= "action_distribution")

        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.output, labels = self.actions, name = "sotfmax")
        self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_rewards_, name = "loss") 
        self.train_opt = tf.train.AdamOptimizer(learning_rate,name="optimizer").minimize(self.loss)


# In[8]:


tf.reset_default_graph()

policy = PolicyGradient()


# In[ ]:


# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/pg/1")

## Losses
tf.summary.scalar("Loss", policy.loss)
## Reward mean
tf.summary.scalar("Reward_mean", policy.mean_reward_)

write_op = tf.summary.merge_all()


# In[ ]:


allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
#episode data after full simulation before training
episode_states, episode_actions, episode_rewards = [],[],[]

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    writer.add_graph(sess.graph)
    for episode in range(max_episodes):
        episodes_rewards_sum = 0
        state = env.reset()
        env.render()
    
        print(state)
        print(state.shape)
        #we simulate the episode until its finished
        while True:
            #conseguimos las probabilidades de las acciones segun la politica
            action_prob_distribution = sess.run(policy.action_distribution, feed_dict = {policy.observations: state.reshape([1,2])})
            action = np.random.choice(range(action_prob_distribution.shape[1]), p = action_prob_distribution.ravel())
            
            #realizamos esas acciones y guardamos la informacion del estado, accion y recompensa en sus respectivos arrays
            new_state, reward, done, info = env.step(action)
            episode_states.append(state)
            
            action_ = np.zeros(action_size)
            action_[action] = 1
            episode_actions.append(action_)
            
            episode_rewards.append(reward)
            
            #when we finished the episode we calculate all the reward and calculate the score function
            if done:
                print(state.shape)
                episode_rewards_sum = np.sum(episode_rewards)
                allRewards.append(episode_rewards_sum)
                total_rewards = np.sum(allRewards)
                
                mean_reward = np.divide(total_rewards,episode+1)
                maximumRewardRecorded = np.amax(allRewards)
                
                print("======================================")
                print('Episode: ',episode)
                print('Episode reward sum:  ', episode_rewards_sum)
                print('Mean reward: ', mean_reward)
                print('Maximum reward: ',maximumRewardRecorded)
                
                discounted_episode_rewards = calculate_discounted_rewards(episode_rewards)
                
                #feedforward, gradient calculation and backprop
                # Feedforward, gradient and backpropagation
                loss_, _ = sess.run([policy.loss, policy.train_opt], feed_dict={policy.observations: np.vstack(np.array(episode_states)),
                                                                 policy.actions: np.vstack(np.array(episode_actions)),
                                                                 policy.discounted_rewards_: discounted_episode_rewards})
                
 
                                                                 
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={policy.observations: np.vstack(np.array(episode_states)),
                                                        policy.actions: np.vstack(np.array(episode_actions)),
                                                        policy.discounted_rewards_: discounted_episode_rewards,
                                                        policy.mean_reward_: mean_reward})
                
               
                writer.add_summary(summary, episode)
                writer.flush()
                
                # Reset the transition stores
                episode_states, episode_actions, episode_rewards = [],[],[]
                
                break
            
            state = new_state

