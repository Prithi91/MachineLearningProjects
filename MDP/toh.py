import gym
import gym_hanoi
import time
from matplotlib import pyplot as plt
import itertools
import numpy as np
from numpy import save, load
import random

def value_iteration(env, states, states_map, actions, actions_map ,T,max_iterations=100000, lmbda=0.9):
  stateValue = [0 for i in range(len(states))]
  value_diff= []
  time_iters=[]
  newStateValue = stateValue.copy()
  iter=0
  delta=1e-4
  conv_e3=0
  start =time.time()
  for i in range(max_iterations):
    iter=i
    for state in states:
        action_values = []
        for a in actions:
            state_value = 0
            s0,s1,s2 = state
            reward = T[s0][s1][s2][a]
            env.current_state = state
            if(reward==float('-inf')):
                next_state = state
                reward = -1
            else:
                if(len(env.disks_on_peg(actions_map[a][0])) is 0):
                    reward=-1
                    next_state = state
                else:
                    if (state == (2, 2, 2)):
                        reward = 100
                        next_state = state
                    else:
                        next_state = list(state)
                        disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                        next_state[disk_to_move]= actions_map[a][1]
                        next_state = tuple(next_state)
                        #print(next_state)
                        if(next_state)==(2,2,2):
                            #print("Goal")
                            reward=100
            state_action_value = reward + lmbda*stateValue[states_map[next_state]]
            state_value += state_action_value
            action_values.append(state_value)
            best_action = np.argmax(np.asarray(action_values))
            newStateValue[states_map[state]] = action_values[best_action]
    value_diff.append(abs(sum(stateValue) - sum(newStateValue)))
    iter_end = time.time() - start
    time_iters.append(iter_end)
    if (abs(sum(stateValue) - sum(newStateValue)) < 200):
        print(abs(sum(stateValue) - sum(newStateValue)))# if there is negligible difference break the loop
        break
    else:
        #if(abs(sum(stateValue) - sum(newStateValue)) < 1e-03):
         #   print("converrged for 1e-03 at {i}".format(i=i))
        stateValue = newStateValue.copy()
  finaliters = range(0, len(value_diff))
  fig, ax1 = plt.subplots()
  ax1.plot(finaliters, value_diff, color='red', label="Difference in values")
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Value Difference")
  ax1.legend()
  ax2 = ax1.twinx()
  ax2.set_ylabel("Time in s")
  ax2.plot(finaliters, time_iters, color='tab:blue', label="Time taken over iterations")
  ax2.legend()
  plt.title("Convergence of Value Iteration- TOH(3 disks), delta=300")
  fig.tight_layout()
  plt.savefig("ValueIteration_TOH_3.png")
  plt.clf()
  print("Value Iteration converged in: {i} iterations".format(i=iter))
  return stateValue

def value_iteration_4(env, states, states_map, actions, actions_map ,T,max_iterations=100000, lmbda=0.9):
  stateValue = [0 for i in range(len(states))]
  value_diff= []
  time_iters=[]
  newStateValue = stateValue.copy()
  iter=0
  delta=1e-4
  conv_e3=0
  start =time.time()
  for i in range(max_iterations):
    iter=i
    for state in states:
        action_values = []
        for a in actions:
            state_value = 0
            s0,s1,s2,s3 = state
            reward = T[s0][s1][s2][s3][a]
            env.current_state = state
            if(reward==float('-inf')):
                next_state = state
                reward = -1
            else:
                if(len(env.disks_on_peg(actions_map[a][0])) is 0):
                    reward=-1
                    next_state = state
                else:
                    if (state == (2, 2, 2,2)):
                        reward = 100
                        next_state = state
                    else:
                        next_state = list(state)
                        disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                        next_state[disk_to_move]= actions_map[a][1]
                        next_state = tuple(next_state)
                        #print(next_state)
                        if(next_state)==(2,2,2,2):
                            #print("Goal")
                            reward=100
            state_action_value = reward + lmbda*stateValue[states_map[next_state]]
            state_value += state_action_value
            action_values.append(state_value)
            best_action = np.argmax(np.asarray(action_values))
            newStateValue[states_map[state]] = action_values[best_action]
    value_diff.append(abs(sum(stateValue) - sum(newStateValue)))
    iter_end = time.time() - start
    time_iters.append(iter_end)
    if (abs(sum(stateValue) - sum(newStateValue)) < 300):   # if there is negligible difference break the loop
        break
    else:
        stateValue = newStateValue.copy()
  finaliters = range(0, len(value_diff))
  fig, ax1 = plt.subplots()
  ax1.plot(finaliters, value_diff,color='red',label="Difference in values")
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Value Difference")
  ax1.legend()
  ax2 = ax1.twinx()
  ax2.set_ylabel("Time in s")
  ax2.plot(finaliters, time_iters,color='tab:blue', label="Time taken over iterations")
  ax2.legend()
  plt.title("Convergence of Value Iteration- TOH(4 disks), delta=300")
  fig.tight_layout()
  # plt.legend()
  plt.savefig("ValueIteration_TOH_4.png")
  plt.clf()
  print("Value Iteration converged in: {i} iterations".format(i=iter))
  return stateValue

def value_iteration_6(env, states, states_map, actions, actions_map ,T,max_iterations=100000, lmbda=0.9):
  stateValue = [0 for i in range(len(states))]
  value_diff= []
  time_iters=[]
  newStateValue = stateValue.copy()
  iter=0
  delta=1e-4
  conv_e3=0
  start =time.time()
  for i in range(max_iterations):
    #print(i)
    iter=i
    for state in states:
        action_values = []
        for a in actions:
            state_value = 0
            s0,s1,s2,s3,s4,s5 = state
            reward = T[s0][s1][s2][s3][s4][s5][a]
            env.current_state = state
            if(reward==float('-inf')):
                next_state = state
                reward = -1
            else:
                if(len(env.disks_on_peg(actions_map[a][0])) is 0):
                    reward=-1
                    next_state = state
                else:
                    if (state == (2, 2, 2,2,2,2)):
                        reward = 100
                        next_state = state
                    else:
                        next_state = list(state)
                        disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                        next_state[disk_to_move]= actions_map[a][1]
                        next_state = tuple(next_state)
                        #print(next_state)
                        if(next_state)==(2,2,2,2,2,2):
                            #print("Goal")
                            reward=100
            state_action_value = reward + lmbda*stateValue[states_map[next_state]]
            state_value += state_action_value
            action_values.append(state_value)
            best_action = np.argmax(np.asarray(action_values))
            newStateValue[states_map[state]] = action_values[best_action]
    value_diff.append(abs(sum(stateValue) - sum(newStateValue)))
    iter_end = time.time() - start
    time_iters.append(iter_end)
    if (abs(sum(stateValue) - sum(newStateValue)) < 10):   # if there is negligible difference break the loop
        break
    else:
        stateValue = newStateValue.copy()
  finaliters = range(0, len(value_diff))
  fig, ax1 = plt.subplots()
  ax1.plot(finaliters, value_diff,color='red', label="Difference in values")
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Value Difference")
  ax1.legend()
  ax2 = ax1.twinx()
  ax2.set_ylabel("Time in s")
  ax2.plot(finaliters, time_iters, color='tab:blue', label="Time taken over iterations")
  ax2.legend()
  plt.title("Convergence of Value Iteration- TOH(6 disks),delta=50")
  fig.tight_layout()
  # plt.legend()
  plt.savefig("ValueIteration_TOH_6.png")
  plt.clf()
  print("Value Iteration converged in: {i} iterations".format(i=iter))
  return stateValue

def policy_iteration(env, states, states_map, actions, actions_map,T, max_iterations=1000, lmbda=0.9):
    policy = {}
    policy_an={}
    time_iters = []
    stateValue = [0 for i in range(len(states))]
    tol = 1
    #tol=300
    for state in range(len(states)):
        randoma = np.random.choice(actions)
        policy_an[state] = randoma
        policy[state] = actions_map[randoma]
    iter = 0
    inneriters = []
    start = time.time()
    for i in range(max_iterations):
        ctr = 0
        iter = i
        #policy evaluation step
        while True:
            ctr+= 1
            delta = 0
            for s in states:
                env.current_state = s
                a = policy_an[states_map[s]]
                state_value = 0
                s0, s1, s2 = s
                reward = T[s0][s1][s2][a]
                if (reward == float('-inf')):
                    next_state = s
                    reward = -1
                else:
                    if (len(env.disks_on_peg(actions_map[a][0])) is 0):
                        reward = -1
                        next_state = s
                    else:
                        if (s == (2, 2, 2)):
                            reward = 100
                            next_state = s
                        else:
                            next_state = list(s)
                            disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                            next_state[disk_to_move] = actions_map[a][1]
                            next_state = tuple(next_state)
                            # print(next_state)
                            if (next_state) == (2, 2, 2):
                                # print("Goal")
                                reward = 100
                state_action_value = reward + lmbda * stateValue[states_map[next_state]]
                state_value += state_action_value
                delta = max(delta, np.abs(state_value - stateValue[states_map[s]]))
                stateValue[states_map[s]] = state_value
            if(delta < tol):
                break
            #else:
                #if(delta<1e-03):
                    #print("converged for 0.001 {iter}".format(iter=iter))
        inneriters.append(ctr)
        #policy improvement step
        is_converged = True
        for s in states:
            env.current_state = s
            old_a = policy_an[states_map[s]]

            new_a = None
            best_val = float('-inf')
            for a in actions:
                state_value = 0
                s0, s1, s2 = s
                reward = T[s0][s1][s2][a]
                if (reward == float('-inf')):
                    next_state = s
                    reward = -1
                else:
                    if (len(env.disks_on_peg(actions_map[a][0])) is 0):
                        reward = -1
                        next_state = s
                    else:
                        if (s == (2, 2, 2)):
                            reward = 100
                            next_state = s
                        else:
                            next_state = list(s)
                            disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                            next_state[disk_to_move] = actions_map[a][1]
                            next_state = tuple(next_state)
                            # print(next_state)
                            if (next_state) == (2, 2, 2):
                                # print("Goal")
                                reward = 100
                state_action_value = reward + lmbda * stateValue[states_map[next_state]]
                state_value += state_action_value
                if(state_value > best_val):
                     best_val= state_value
                     new_a = a
            policy[states_map[s]] = actions_map[new_a]
            policy_an[states_map[s]] = new_a
            if(new_a != old_a):
                is_converged = False
        iter_end = time.time() - start
        time_iters.append(iter_end)
        if(is_converged):
            break
    finaliters = range(0,len(inneriters))
    print("Policy Iteration converged in: {n} iterations".format(n=iter))
    fig, ax1 = plt.subplots()
    ax1.plot(finaliters, inneriters,color='red',label="Length of Inner Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Inner value iterations")
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time in s")
    ax2.plot(finaliters, time_iters, color='tab:blue', label="Time taken over iterations")
    ax2.legend()
    plt.title("Convergence of policy iteration-TOH")
    fig.tight_layout()
    plt.savefig("PolicyIteration_TOH.png")
    plt.clf()
    return policy

def policy_iteration_4(env, states, states_map, actions, actions_map,T, max_iterations=1000, lmbda=0.9):
    policy = {}
    policy_an={}
    time_iters = []
    stateValue = [0 for i in range(len(states))]
    tol = 1
    for state in range(len(states)):
        randoma = np.random.choice(actions)
        policy_an[state] = randoma
        policy[state] = actions_map[randoma]
    iter = 0
    inneriters = []
    start = time.time()
    for i in range(max_iterations):
        ctr = 0
        iter = i
        #policy evaluation step
        while True:
            ctr+= 1
            delta = 0
            for s in states:
                env.current_state = s
                a = policy_an[states_map[s]]
                state_value = 0
                s0, s1, s2,s3 = s
                reward = T[s0][s1][s2][s3][a]
                if (reward == float('-inf')):
                    next_state = s
                    reward = -1
                else:
                    if (len(env.disks_on_peg(actions_map[a][0])) is 0):
                        reward = -1
                        next_state = s
                    else:
                        if (s == (2, 2, 2,2)):
                            reward = 100
                            next_state = s
                        else:
                            next_state = list(s)
                            disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                            next_state[disk_to_move] = actions_map[a][1]
                            next_state = tuple(next_state)
                            # print(next_state)
                            if (next_state) == (2, 2, 2,2):
                                # print("Goal")
                                reward = 100
                state_action_value = reward + lmbda * stateValue[states_map[next_state]]
                state_value += state_action_value
                delta = max(delta, np.abs(state_value - stateValue[states_map[s]]))
                stateValue[states_map[s]] = state_value
            if(delta < tol):
                break
            #else:
                #if(delta<1e-03):
                    #print("converged for 0.001 {iter}".format(iter=iter))
        inneriters.append(ctr)
        #policy improvement step
        is_converged = True
        for s in states:
            env.current_state = s
            old_a = policy_an[states_map[s]]
            new_a = None
            best_val = float('-inf')
            for a in actions:
                state_value = 0
                s0, s1, s2,s3 = s
                reward = T[s0][s1][s2][s3][a]
                if (reward == float('-inf')):
                    next_state = s
                    reward = -1
                else:
                    if (len(env.disks_on_peg(actions_map[a][0])) is 0):
                        reward = -1
                        next_state = s
                    else:
                        if (s == (2, 2, 2,2)):
                            reward = 100
                            next_state = s
                        else:
                            next_state = list(s)
                            disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                            next_state[disk_to_move] = actions_map[a][1]
                            next_state = tuple(next_state)
                            # print(next_state)
                            if (next_state) == (2, 2, 2,2):
                                # print("Goal")
                                reward = 100
                state_action_value = reward + lmbda * stateValue[states_map[next_state]]
                state_value += state_action_value
                if(state_value > best_val):
                     best_val= state_value
                     new_a = a
            policy[states_map[s]] = actions_map[new_a]
            policy_an[states_map[s]] = new_a
            if(new_a != old_a):
                is_converged = False
        iter_end = time.time() - start
        time_iters.append(iter_end)
        if(is_converged):
            break
    finaliters = range(0,len(inneriters))
    print("Policy Iteration converged in: {n} iterations".format(n=iter))
    fig, ax1 = plt.subplots()
    ax1.plot(finaliters, inneriters, color='red', label="Length of Inner Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Inner value iterations")
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time in s")
    ax2.plot(finaliters, time_iters, color='tab:blue', label="Time taken over iterations")
    ax2.legend()
    plt.title("Convergence of policy iteration-TOH_4")
    fig.tight_layout()
    plt.savefig("PolicyIteration_TOH_4.png")
    plt.clf()
    return policy

def policy_iteration_6(env, states, states_map, actions, actions_map,T, max_iterations=1000, lmbda=0.9):
    policy = {}
    policy_an={}
    time_iters = []
    stateValue = [0 for i in range(len(states))]
    tol = 1
    for state in range(len(states)):
        randoma = np.random.choice(actions)
        policy_an[state] = randoma
        policy[state] = actions_map[randoma]
    iter = 0
    inneriters = []
    start = time.time()
    for i in range(max_iterations):
        ctr = 0
        iter = i
        #policy evaluation step
        while True:
            ctr+= 1
            delta = 0
            for s in states:
                env.current_state = s
                a = policy_an[states_map[s]]
                state_value = 0
                s0, s1, s2,s3,s4,s5 = s
                reward = T[s0][s1][s2][s3][s4][s5][a]
                if (reward == float('-inf')):
                    next_state = s
                    reward = -1
                else:
                    if (len(env.disks_on_peg(actions_map[a][0])) is 0):
                        reward = -1
                        next_state = s
                    else:
                        if (s == (2, 2, 2,2,2,2)):
                            reward = 100
                            next_state = s
                        else:
                            next_state = list(s)
                            disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                            next_state[disk_to_move] = actions_map[a][1]
                            next_state = tuple(next_state)
                            # print(next_state)
                            if (next_state) == (2, 2, 2,2,2,2):
                                # print("Goal")
                                reward = 100
                state_action_value = reward + lmbda * stateValue[states_map[next_state]]
                state_value += state_action_value
                delta = max(delta, np.abs(state_value - stateValue[states_map[s]]))
                stateValue[states_map[s]] = state_value
            if(delta < tol):
                break
            #else:
                #if(delta<1e-03):
                    #print("converged for 0.001 {iter}".format(iter=iter))
        inneriters.append(ctr)
        #policy improvement step
        is_converged = True
        for s in states:
            env.current_state = s
            old_a = policy_an[states_map[s]]
            new_a = None
            best_val = float('-inf')
            for a in actions:
                state_value = 0
                s0, s1, s2,s3,s4,s5 = s
                reward = T[s0][s1][s2][s3][s4][s5][a]
                if (reward == float('-inf')):
                    next_state = s
                    reward = -1
                else:
                    if (len(env.disks_on_peg(actions_map[a][0])) is 0):
                        reward = -1
                        next_state = s
                    else:
                        if (s == (2, 2, 2,2,2,2)):
                            reward = 100
                            next_state = s
                        else:
                            next_state = list(s)
                            disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                            next_state[disk_to_move] = actions_map[a][1]
                            next_state = tuple(next_state)
                            # print(next_state)
                            if (next_state) == (2, 2, 2,2,2,2):
                                # print("Goal")
                                reward = 100
                state_action_value = reward + lmbda * stateValue[states_map[next_state]]
                state_value += state_action_value
                if(state_value > best_val):
                     best_val= state_value
                     new_a = a
            policy[states_map[s]] = actions_map[new_a]
            policy_an[states_map[s]] = new_a
            if(new_a != old_a):
                is_converged = False
        iter_end = time.time() - start
        time_iters.append(iter_end)
        if(is_converged):
            break
    finaliters = range(0,len(inneriters))
    print("Policy Iteration converged in: {n} iterations".format(n=iter))
    fig, ax1 = plt.subplots()
    ax1.plot(finaliters, inneriters, color='red', label="Length of Inner Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Inner value iterations")
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time in s")
    ax2.plot(finaliters, time_iters, color='tab:blue', label="Time taken over iterations")
    ax2.legend()
    plt.title("Convergence of policy iteration-TOH_6")
    fig.tight_layout()
    plt.savefig("PolicyIteration_TOH_6.png")
    plt.clf()
    return policy

def q_learning(env, states, states_map, actions, actions_map,max_iters, gamma=0.9, alpha=0.6,epsilon=0.1,dyna=0):
    q_table = np.zeros([len(states), len(actions)])
    episodes = []
    time_iters = []
    for i in range(max_iters):
        episodes.append(i)
    modelTC = np.full((len(states), len(actions), len(states)), 0.0000001)
    modelT = np.zeros((len(states), len(actions), len(states)))
    modelR = np.zeros((len(states), len(actions)))
    episode_lengths = []
    episode_rewards = []
    start = time.time()
    for i in range(max_iters):
        print(i)
        state = env.reset()
        epochs, penalities, reward, =0,0,0
        done = False
        ctr=0
        while not done:
            ctr +=1
            if(random.uniform(0,1) < epsilon):
                action = np.random.choice(actions)
            else:
                action = np.argmax(q_table[states_map[state],:])
            newstate, reward, done, info = env.step(action)
            if(reward==0): reward=-100
            if(reward==-1):reward=-200
            #print(reward)
            if (done):
                episode_lengths.append(ctr)
                episode_rewards.append(reward)
            old_val = q_table[states_map[state], action]
            next_max = np.max(q_table[states_map[newstate]])
            new_val = (1-alpha)* old_val + alpha * (reward + gamma * next_max)
            q_table[states_map[state], action] = new_val
            #####Dyna
            modelTC[state, action, newstate] += 1
            modelT[state, action, newstate] = modelTC[state, action, newstate] / np.sum(
                modelTC[state, action])
            modelR[state, action] = (1 - alpha) * modelR[state, action] + (alpha * reward)
            for i in range(dyna):
                news = np.random.choice(range(len(states)))
                newa = np.random.choice(range(len(actions)))
                news_prime = np.argmax(modelT[news, newa])
                r_prime = modelR[news, newa]
                prev_i = (1 - alpha) * (q_table[news, newa])
                bestaction_i = np.argmax(q_table[news_prime])
                curr_i = alpha * (r_prime + gamma * q_table[news_prime, bestaction_i])
                q_table[news, newa] = prev_i + curr_i
            state = newstate
            epochs += 1
        epsilon = epsilon*0.99
        iter_end = time.time() - start
        time_iters.append(iter_end)
    var_eps = np.diff(episode_lengths)
    length_eps = range(0, len(var_eps))
    fig, ax1 = plt.subplots()
    ax1.plot(length_eps, var_eps, color='red',label="Episode Lengths")
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time in s")
    alliters = range(0, max_iters)
    ax2.plot(alliters, time_iters, color='tab:blue', label="Time taken over iterations")
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    #plt.title('Episode Length over Time-6 disks(epsilon=0.1, Decayrate=0.99)')
    #plt.title('Episode Length over Time-4 disks(constant epsilon at 0.1,alpha=0.9)')
    plt.title('Episode Length over Time-4 disks(epsilon=0.1,dyna=20,alpha=0.4)')
    fig.tight_layout()
    #plt.savefig('Episodelengths_TOH_6(alpha decay).png')
    plt.savefig('Episodelengths_TOH_6(dyna=3).png')
    #plt.savefig('Episodelengths_TOH_4(No decay).png')
    plt.clf()
    # plt.plot(episodes, episode_rewards, linestyle="dotted", marker=".")
    # plt.xlabel('Episode')
    # plt.ylabel('Episode Rewards')
    # plt.title('Episode Rewards over Time-TOH(4 disks)')
    # #plt.savefig('EpisodeRewards_frozenlakeTOH.png')
    # plt.savefig('EpisodeRewards_frozenlakeTOH_4.png')
    #plt.clf()
    return q_table


def get_policy(env,states, states_map, actions, actions_map ,T,stateValue, lmbda=0.9):
  policy = [0 for i in range(len(states))]
  for state in states:
    action_values = []
    for a in actions:
      action_value = 0
      s0, s1, s2 = state
      reward = T[s0][s1][s2][a]
      env.current_state = state
      if (reward == float('-inf')):
          next_state = state
          reward = -1
      else:
          if (len(env.disks_on_peg(actions_map[a][0])) is 0):
              reward = -1
              next_state = state
          else:
              if (state == (2, 2, 2)):
                  reward = 100
                  next_state = state
              else:
                  next_state = list(state)
                  disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                  next_state[disk_to_move] = actions_map[a][1]
                  next_state = tuple(next_state)
                  # print(next_state)
                  if (next_state) == (2, 2, 2):
                      #print("Goal")
                      reward = 100
      action_value += reward + lmbda * stateValue[states_map[next_state]]
      action_values.append(action_value)
    best_action = np.argmax(np.asarray(action_values))
    policy[states_map[state]] = actions_map[best_action]
  return policy

def get_policy_4(env,states, states_map, actions, actions_map ,T,stateValue, lmbda=0.9):
  policy = [0 for i in range(len(states))]
  for state in states:
    action_values = []
    for a in actions:
      action_value = 0
      s0, s1, s2,s3 = state
      reward = T[s0][s1][s2][s3][a]
      env.current_state = state
      if (reward == float('-inf')):
          next_state = state
          reward = -1
      else:
          if (len(env.disks_on_peg(actions_map[a][0])) is 0):
              reward = -1
              next_state = state
          else:
              if (state == (2, 2, 2,2)):
                  reward = 100
                  next_state = state
              else:
                  next_state = list(state)
                  disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                  next_state[disk_to_move] = actions_map[a][1]
                  next_state = tuple(next_state)
                  # print(next_state)
                  if (next_state) == (2, 2, 2,2):
                      #print("Goal")
                      reward = 100
      action_value += reward + lmbda * stateValue[states_map[next_state]]
      action_values.append(action_value)
    best_action = np.argmax(np.asarray(action_values))
    policy[states_map[state]] = actions_map[best_action]
  return policy

def get_policy_6(env,states, states_map, actions, actions_map ,T,stateValue, lmbda=0.9):
  policy = [0 for i in range(len(states))]
  for state in states:
    action_values = []
    for a in actions:
      action_value = 0
      s0, s1, s2,s3,s4,s5 = state
      reward = T[s0][s1][s2][s3][s4][s5][a]
      env.current_state = state
      if (reward == float('-inf')):
          next_state = state
          reward = -1
      else:
          if (len(env.disks_on_peg(actions_map[a][0])) is 0):
              reward = -1
              next_state = state
          else:
              if (state == (2, 2, 2,2,2,2)):
                  reward = 100
                  next_state = state
              else:
                  next_state = list(state)
                  disk_to_move = min(env.disks_on_peg(actions_map[a][0]))
                  next_state[disk_to_move] = actions_map[a][1]
                  next_state = tuple(next_state)
                  # print(next_state)
                  if (next_state) == (2, 2, 2,2,2,2):
                      #print("Goal")
                      reward = 100
      action_value += reward + lmbda * stateValue[states_map[next_state]]
      action_values.append(action_value)
    best_action = np.argmax(np.asarray(action_values))
    policy[states_map[state]] = actions_map[best_action]
  return policy

def get_policy_Q(states, q_table, actions_map):
    policy = [0 for i in range(len(states))]
    for state in range(len(states)):
        policy[state] = actions_map[np.argmax(q_table[state])]
    return policy

def evaluate_policy(policy, env,states_map):
    curr_state = env.reset()
    actions_map = {(0, 1):0,(0, 2):1, (1, 0):2,
                   (1, 2):3, (2, 0):4, (2, 1):5}
    reward_total=0
    done=False
    f = open('solution_6_QLdyna.txt',"a")
    s_str=""
    for s in curr_state:
        s_str += str(s)
    f.write(s_str)
    while not done:
        best_move = actions_map[policy[states_map[curr_state]]]
        newstate,reward,done,info = env.step(best_move)
        s_str=""
        for s in newstate:
            s_str += str(s)
        f.write("\n")
        f.write(s_str)
        curr_state = newstate
        reward_total+=reward
    print(reward_total)

def printstates(states,n):
    # cnt=-1
    # for index in range(0,28):
    #     cnt+= 1
    #     for i in states:
    #         print(states[i])
    index = 0
    f = open('policy_4.txt',"a")
    for row in range(0,10):
        for col in range(index, index + n):
            s_str=""
            for s in states[index]:
                s_str += str(s)
            f.write(" ")
            f.write(s_str)
            index += 1
        f.write("\n")
    f.close()

if __name__=="__main__":
    env = gym.make("Hanoi-v0")
    #env.set_env_parameters(num_disks=3)
    #env.set_env_parameters(num_disks=4)
    env.set_env_parameters(num_disks=6)
    env.reset()
    actions=range(0,6)
    actions_map = {0: (0, 1), 1: (0, 2), 2: (1, 0),
                   3: (1, 2), 4: (2, 0), 5: (2, 1)}
    #s_new, reward, done, info = env.step(1)
    #T = env.get_movability_map(fill=True)
    T = load('T.npy')
    #save('T.npy', T)
    # f = open('mov_map_6.txt',"a")
    # f.write(T)
    # f.close()
    #print(T)
    #id_list = 3 * [0] + 3 * [1] + 3 * [2]
    #id_list = 4 * [0] + 4 * [1] + 4 * [2]
    id_list = 6 * [0] + 6 * [1] + 6 * [2]
    #print(id_list)
    #states = list(set(list(itertools.permutations(id_list, 3))))
    #states = list(set(list(itertools.permutations(id_list, 4))))
    states = list(set(list(itertools.permutations(id_list, 6))))
    states = sorted(states, key=lambda x:x[0])
    states = sorted(states,key=lambda  x:x[1])
    states = sorted(states,key=lambda  x:x[2])
    states = sorted(states,key=lambda  x:x[3])
    states = sorted(states,key=lambda  x:x[4])
    states = sorted(states,key=lambda  x:x[5])
    states_map={}
    for s in range(len(states)):
         states_map[states[s]]=s
    # printstates(states,27)
    values = value_iteration_6(env, states, states_map, actions, actions_map, T)
    policy = get_policy_6(env, states, states_map, actions, actions_map, T, values)
    # printstates(policy,27)
    evaluate_policy(policy,env,states_map)
    #printstates(states,10)
    #values = value_iteration_4(env,states,states_map,actions,actions_map,T)
    #policy = get_policy_4(env, states, states_map, actions, actions_map, T, values)
    # #printstates(policy,10)
    #evaluate_policy(policy,env,states_map)
    #values = value_iteration(env,states,states_map,actions,actions_map,T)
    # policy = get_policy(env, states, states_map, actions, actions_map, T, values)
    # print(policy)
    # evaluate_policy(policy,env,states_map)
    # policy = policy_iteration(env, states, states_map, actions, actions_map,T)
    # print(policy)
    # evaluate_policy(policy,env,states_map)
    # policy = policy_iteration_4(env, states, states_map, actions, actions_map, T)
    # print(policy)
    # evaluate_policy(policy,env,states_map)
    policy = policy_iteration_6(env, states, states_map, actions, actions_map, T)
    # print(policy)
    evaluate_policy(policy,env,states_map)
    qvalues = q_learning(env, states,states_map, actions, actions_map,500,alpha=0.4,dyna=20)#3-d
    #qvalues = q_learning(env, states,states_map, actions, actions_map,200,alpha=0.9)#no decay
    #qvalues = q_learning(env, states,states_map, actions, actions_map,150,alpha=0.9,epsilon=0.1)#no decay
    policy = get_policy_Q(states,qvalues,actions_map)
    evaluate_policy(policy,env,states_map)




