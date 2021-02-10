from gym.envs.toy_text import frozen_lake
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from gym import envs
from numpy import save, load
import random
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time

matplotlib.style.use('ggplot')


def value_iteration(env, max_iterations=100000, lmbda=0.9):
  stateValue = [0 for i in range(env.nS)]
  value_diff= []
  time_iters=[]
  newStateValue = stateValue.copy()
  iter=0
  delta=1e-04
  conv_e3=0
  start =time.time()
  for i in range(max_iterations):
    iter=i
    for state in range(env.nS):
      action_values = []
      for action in range(env.nA):
        state_value = 0
        for j in range(len(env.P[state][action])):
          prob, next_state, reward, done = env.P[state][action][j]
          state_action_value = prob * (reward + lmbda*stateValue[next_state])
          state_value += state_action_value
        action_values.append(state_value)      #the value of each action
        best_action = np.argmax(np.asarray(action_values))   # choose the action which gives the maximum value
        newStateValue[state] = action_values[best_action]  #update the value of the state
    value_diff.append(abs(sum(stateValue)-sum(newStateValue)))
    iter_end = time.time()-start
    time_iters.append(iter_end)
    #if i > 1000:
    if (abs(sum(stateValue) - sum(newStateValue)) < 1e-04):
        #print(abs(sum(stateValue) - sum(newStateValue)))# if there is negligible difference break the loop
        break
    else:
        #if(abs(sum(stateValue) - sum(newStateValue)) < 1e-03):
           #print("converrged for 1e-03 at {i}".format(i=i))
        stateValue = newStateValue.copy()
  finaliters = range(0, len(value_diff))
  plt.plot(finaliters, value_diff, label="Difference in values")
  # plt.plot(83, value_diff[83], 'g*',label ="convergence point for delta=0.001")#20x20
  # plt.plot(53, value_diff[53], 'g*', label="convergence point for delta=0.0001")#20x20
  #plt.plot(55, value_diff[55], 'g*',label ="convergence point for delta=0.001")#8x8
  #plt.plot(74, value_diff[74], 'g*', label="convergence point for delta=0.0001")#8x8
  # plt.plot(40, value_diff[40], 'g*', label="convergence point for delta=0.001")#4x4
  # plt.plot(57, value_diff[57], 'g*', label="convergence point for delta=0.0001")#4x4
  plt.plot(finaliters, time_iters, label="Time taken over iterations")
  plt.xlabel("Iterations")
  plt.ylabel("Value Difference")
  plt.title("Convergence of Value Iteration- 16x16, discount=1.0")
  #plt.title("Convergence of Value Iteration- 8x8, discount=1.0")
  #plt.title("Convergence of Value Iteration- 4x4, discount=0.9")
  plt.legend()
  plt.savefig("ValueIteration_frozenlake16x16.png")
  #plt.savefig("ValueIteration_frozenlake8x8.png")
  #plt.savefig("ValueIteration_frozenlake4x4.png")
  plt.clf()
  print("Value Iteration converged in: {i} iterations".format(i=iter))
  return stateValue

def policy_iteration(env, max_iterations=1000, lmbda=0.9):
    #Random Policy
    policy = {}
    time_iters = []
    stateValue = [0 for i in range(env.nS)]
    tol = 1e-04
    for state in range(env.nS):
        policy[state] = np.random.choice(range(env.nA))
    iter=0
    inneriters=[]
    start = time.time()
    for i in range(max_iterations):
        ctr = 0
        iter = i
        #policy evaluation step
        while True:
            ctr+= 1
            delta = 0
            for s in range(env.nS):
                a = policy[s]
                state_value = 0
                for j in range(len(env.P[s][a])):
                   prob, next_state, reward, done = env.P[s][a][j]
                   state_action_value = prob * (reward + lmbda * stateValue[next_state])
                   state_value += state_action_value
                delta = max(delta, np.abs(state_value - stateValue[s]))
                stateValue[s] = state_value
            if(delta < tol):
                break
            # else:
            #     if(delta<1e-03):
            #         print("converged for 0.001 {iter}".format(iter=iter))
        inneriters.append(ctr)
        #policy improvement step
        is_converged = True
        for s in range(env.nS):
            old_a = policy[s]
            new_a = None
            best_val = float('-inf')
            for action in range(env.nA):
                state_value = 0
                for i in range(len(env.P[s][action])):
                    prob, next_state, reward, done = env.P[s][action][i]
                    state_action_value = prob * (reward + lmbda * stateValue[next_state])
                    state_value += state_action_value
                if(state_value > best_val):
                    best_val= state_value
                    new_a = action
            policy[s] = new_a
            if(new_a != old_a):
                is_converged = False
            policy[s] = new_a
        iter_end = time.time() - start
        time_iters.append(iter_end)
        if(is_converged):
            break
    finaliters = range(0,len(inneriters))
    #print(time_iters)
    print("Policy Iteration converged in: {n} iterations".format(n=iter))
    fig, ax1 = plt.subplots()
    ax1.plot(finaliters, inneriters)
    #ax1.plot(1, inneriters[1], 'g*', label="convergence point for delta=0.001")#16x16
    #ax1.plot(13, inneriters[13], 'g*', label="convergence point for delta=0.0001")#16x16
    # ax1.plot(1, inneriters[1], 'g*', label="convergence point for delta=0.001")#8x8
    # ax1.plot(8, inneriters[8], 'g*', label="convergence point for delta=0.0001")#8x8
    # ax1.plot(1, inneriters[1], 'g*', label="convergence point for delta=0.001")#4x4
    # ax1.plot(5, inneriters[5], 'g*', label="convergence point for delta=0.0001")#4x4
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Inner value iterations")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time in s")
    ax2.plot(finaliters, time_iters, color='tab:blue',label="Time taken over iterations")

    plt.title("Convergence of policy iteration-16x16, discount=1.0")
    #plt.title("Convergence of policy iteration-8x8, discount=1.0")
    #plt.title("Convergence of policy iteration-4x4, discount=1.0")
    plt.legend(loc="best")
    fig.tight_layout()
    plt.savefig("PolicyIteration_frozenlake16x16.png")
    #plt.savefig("PolicyIteration_frozenlake8x8.png")
    #plt.savefig("PolicyIteration_frozenlake4x4.png")
    plt.clf()
    return policy

def q_learning(env, max_iters, gamma=0.9, alpha=0.4,epsilon=0.1,dyna=0):
    #print(env.P)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    saved_state = None
    last_action = None
    episodes = []
    time_iters = []
    for i in range(max_iters):
        episodes.append(i)
    # episode_lengths = np.zeros(max_iters)
    # episode_rewards = np.zeros(max_iters)
    modelTC = np.full((env.observation_space.n, env.action_space.n, env.observation_space.n), 0.0000001)
    modelT = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    modelR = np.zeros((env.observation_space.n, env.action_space.n))
    episode_lengths = []
    episode_rewards = []
    start = time.time()
    for i in range(max_iters):
        print(i)
        #state = np.random.choice(range(env.nS))
        #if(saved_state is not None):
         #   env.sim.set_state(saved_state)
          #  state = saved_state
        #state = env.reset(saved_state, last_action)
        state = env.reset()
        epochs, penalities, reward, =0,0,0
        done = False
        ctr=0
        while not done:
            ctr +=1
            if(random.uniform(0,1) < epsilon):
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state,:])
            saved_state = state
            last_action = action
            next_state, reward, done, info = env.step(action)
            #episode_rewards[i] =reward
            #episode_lengths[i] = ctr
            #print(next_state)
            if(done and reward==0):
                reward=-1
            if (done):
                episode_lengths.append(ctr)
                episode_rewards.append(reward)
            #if(reward>0):  print(reward)
            old_val = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_val = (1-alpha)* old_val + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_val
            #####Dyna
            modelTC[state, action, next_state] += 1
            modelT[state, action, next_state] = modelTC[state, action, next_state] / np.sum(
                modelTC[state, action])
            modelR[state, action] = (1 - alpha) * modelR[state, action] + (alpha * reward)
            for i in range(dyna):
                news = np.random.choice(range(env.nS))
                newa = np.random.choice(range(env.nA))
                news_prime = np.argmax(modelT[news, newa])
                r_prime = modelR[news, newa]
                prev_i = (1 - alpha) * (q_table[news, newa])
                bestaction_i = np.argmax(q_table[news_prime])
                curr_i = alpha * (r_prime + gamma * q_table[news_prime, bestaction_i])
                q_table[news, newa] = prev_i + curr_i
            epochs += 1
            state = next_state
        #epsilon = epsilon*0.99
        iter_end = time.time() - start
        time_iters.append(iter_end)
    var_eps = np.absolute(np.diff(episode_lengths))
    length_eps = range(0,len(var_eps))
    # fig, ax1 = plt.subplots()
    # ax1.plot(finaliters, inneriters)
    # # ax1.plot(1, inneriters[1], 'g*', label="convergence point for delta=0.001")#16x16
    # # ax1.plot(13, inneriters[13], 'g*', label="convergence point for delta=0.0001")#16x16
    # # ax1.plot(1, inneriters[1], 'g*', label="convergence point for delta=0.001")#8x8
    # # ax1.plot(8, inneriters[8], 'g*', label="convergence point for delta=0.0001")#8x8
    # # ax1.plot(1, inneriters[1], 'g*', label="convergence point for delta=0.001")#4x4
    # # ax1.plot(5, inneriters[5], 'g*', label="convergence point for delta=0.0001")#4x4
    # ax1.set_xlabel("Iterations")
    # ax1.set_ylabel("Inner value iterations")
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("Time in s")
    # ax2.plot(finaliters, time_iters, color='tab:blue', label="Time taken over iterations")
    fig, ax1 = plt.subplots()
    ax1.plot(length_eps,var_eps, label ="Episode Lengths")
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time in s")
    alliters = range(0, max_iters)
    ax2.plot(alliters,time_iters,color='tab:blue', label="Time taken over iterations")
    ax2.legend()
    #plt.plot(episodes, episode_lengths,linestyle="dotted",marker=".")
    #plt.title('Episode Length over Time-4x4(Epsilon=0.5,Decayrate=0.01)')
    #plt.title('Episode Length over Time-8x8(Epsilon=0.1)')
    plt.title('Episode Length over Time-16x16(Epsilon=0.1,gamma=1.0)')
    fig.tight_layout()
    #plt.savefig('Episodelengths_frozenlake4x4(0.1,Decayrate-0.01).png')
    #plt.savefig('Episodelengths_frozenlake8x8(0.1Nodecay).png')
    plt.savefig('Episodelengths_frozenlake16x16(0.1Nodecay).png')
    plt.clf()
   # plt.plot(episodes, episode_rewards, linestyle="dotted",marker=".")
    plt.plot(episodes, episode_rewards)
    plt.axis([0,max_iters,-5,5])
    plt.xlabel('Episode')
    plt.ylabel('Episode Rewards')
    #plt.title('Episode Rewards over Time-4x4(Epsilon=0.5)')
    #plt.savefig('EpisodeRewards_frozenlake4x4(0.1No decay).png')
    plt.title('Episode Rewards over Time-8x8')
    plt.savefig('EpisodeRewards_frozenlake8x8.png')
    #plt.title('Episode Rewards over Time-16x16')
    #plt.savefig('EpisodeRewards_frozenlake16x16(0.1Nodecay).png')
    plt.clf()
    return q_table

def get_policy(env,stateValue, lmbda=0.9):
  policy = [0 for i in range(env.nS)]
  for state in range(env.nS):
    action_values = []
    for action in range(env.nA):
      action_value = 0
      for i in range(len(env.P[state][action])):
        prob, next_state, r, _ = env.P[state][action][i]
        action_value += prob * (r + lmbda * stateValue[next_state])
      action_values.append(action_value)
    best_action = np.argmax(np.asarray(action_values))
    policy[state] = best_action
  return policy

def get_policy_Q(env, q_table):
    policy = [0 for i in range(env.nS)]
    for state in range(env.nS):
        policy[state] = np.argmax(q_table[state])
    return policy

def get_score(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        #print(episode)
        observation = env.reset()
        steps = 0
        while True:

            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward == 1:
                steps_list.append(steps)
                break
            elif done and reward == 0:
                misses += 1
                break
    print('Average of {:.0f} steps taken to reach goal'.format(np.mean(steps_list)))
    print('Fell in hole {:.2f} % of the times'.format((misses / episodes) * 100))
    print('----------------------------------------------')

def print_policy(policy, n):
    index = 0
    for row in range(n):
        for col in range(index, index+n):
            if(policy[index]== 0): print("<", end=" ")
            elif(policy[index]== 1): print("v", end=" ")
            elif(policy[index] == 2): print(">", end=" ")
            elif(policy[index] == 3): print("^", end=" ")
            index += 1
        print("\n")

if __name__=="__main__":
    #env = frozen_lake.FrozenLakeEnv(map_name="4x4")
    #print(env.P)
    #env = frozen_lake.FrozenLakeEnv(map_name="8x8")
    #env.render()
    # env_p_o = frozen_lake.generate_random_map(size=16)
    # print(env_p_o)
    #save('env.npy', env_p)
    env_p=['SFFFFFHFFFHFFFFH', 'FFFFFFFHFFFFFFFH', 'FFFFFFHHFFHFFHFF', 'FFHFFFFFFFFFHFFF', 'FFHFHFFFFFFHHFFF', 'FHFHFFHHFFFFFFFF', 'HFFFFFFFFFFFFFHF', 'FFHFFFFFFHFHFFHF', 'FFHFFFFFFFFFFFFF', 'FHFFFFFHFFFHFFFF', 'FHFFFFFFHFFHFHFH', 'FFFHFFFFFFFFHFFF', 'FFFHFHFFFHFFFFFF', 'FFHHFFHFFFHFFFFH', 'FFHFHFFHFFHHFFFF', 'FFFFFFFHFFFFFFFG']
    #env_p = load('env.npy')
    #print(env_p)
    env = frozen_lake.FrozenLakeEnv(desc=env_p, map_name="16x16")
    env.render()
    vals = value_iteration(env, max_iterations=10000,lmbda=1.0)
    policy = get_policy(env, vals)
    # # # # print('-------------------Value Iteration---------------------------')
    # print_policy(policy,16)
    get_score(env, policy, episodes=10000)
    policy = policy_iteration(env, max_iterations=10000,lmbda=1.0)
    # # # print('-------------------Policy Iteration---------------------------')
    #print_policy(policy,16)
    get_score(env, policy, episodes=10000)
    q_table = q_learning(env, max_iters=500000, alpha=0.8,epsilon=0.1,gamma=1.0)#20x20
    # q_table = q_learning(env, max_iters=100000, alpha=0.8, epsilon=0.1,dyna=5)#8x8
    # q_table = q_learning(env, max_iters=2000, gamma=0.9, alpha=0.8, epsilon=0.6)#4x4
    # #print(q_table)
    policy = get_policy_Q(env, q_table)
    # print('-------------------Q-learning---------------------------')
    print_policy(policy,16)
    get_score(env, policy, episodes=1000)