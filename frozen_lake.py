import numpy as np
import gym
import time


def play_episodes(enviorment, n_episodes, policy, random=False):
    # intialize wins and total reward
    wins = 0
    total_reward = 0

    # loop over number of episodes to play
    for episode in range(n_episodes):

        # flag to check if the game is finished
        terminated = False

        # reset the enviorment every time when playing a new episode
        state = enviorment.reset()

        while not terminated:

            # check if the random flag is not true then follow the given policy other wise take random action
            if random:
                action = enviorment.action_space.sample()
            else:
                action = policy[state]

            # take the next step
            next_state, reward, terminated, info = enviorment.step(action)

            enviorment.render()

            # accumalate total reward
            total_reward += reward

            # change the state
            state = next_state

            # if game is over with positive reward then add 1.0 in wins
            if terminated and reward == 1.0:
                wins += 1

    # calculate average reward
    average_reward = total_reward / n_episodes

    return wins, total_reward, average_reward


def policy_eval(env, policy, V, discount_factor):
    policy_value = np.zeros(env.nS)
    for state, action in enumerate(policy):
        for probablity, next_state, reward, info in env.P[state][action]:
            policy_value[state] += probablity * (reward + (discount_factor * V[next_state]))

    return policy_value


def one_step_lookahead(env, state, V, discount_factor=0.99):
    # initialize vector of action values
    action_values = np.zeros(env.nA)

    # loop over the actions we can take in an enviorment
    for action in range(env.nA):
        # loop over the P_sa distribution.
        for probablity, next_state, reward, info in env.P[state][action]:
            # if we are in state s and take action a. then sum over all the possible states we can land into.
            action_values[action] += probablity * (reward + (discount_factor * V[next_state]))

    return action_values


def update_policy(env, policy, V, discount_factor):

    for state in range(env.nS):
        # for a given state compute state-action value.
        action_values = one_step_lookahead(env, state, V, discount_factor)

        # choose the action which maximizez the state-action value.
        policy[state] = np.argmax(action_values)

    return policy


def policy_iteration(env, discount_factor=0.999, max_iteration=1000):
    # intialize the state-Value function
    V = np.zeros(env.nS)

    # intialize a random policy
    policy = np.random.randint(0, 4, env.nS)
    policy_prev = np.copy(policy)

    for i in range(max_iteration):

        # evaluate given policy
        V = policy_eval(env, policy, V, discount_factor)

        # improve policy
        policy = update_policy(env, policy, V, discount_factor)

        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            if (np.all(np.equal(policy, policy_prev))):
                print('policy converged at iteration %d' % (i + 1))
                break
            policy_prev = np.copy(policy)

    return V, policy


if __name__ == '__main__':
    enviorment2 = gym.make('FrozenLake-v0')
    tic = time.time()
    opt_V2, opt_policy2 = policy_iteration(enviorment2.env, discount_factor=0.999, max_iteration=10000)
    toc = time.time()
    elapsed_time = (toc - tic) * 1000
    # action mapping for display the final result
    action_mapping = {
        3: '\u2191',  # UP
        2: '\u2192',  # RIGHT
        1: '\u2193',  # DOWN
        0: '\u2190'  # LEFT
    }
    print(f"Time to converge: {elapsed_time: 0.3} ms")
    print('Optimal Value function: ')
    print(opt_V2.reshape((4, 4)))
    print('Final Policy: ')
    print(opt_policy2)
    print(' '.join([action_mapping[(action)] for action in opt_policy2]))

    ##############

    n_episode = 10
    wins, total_reward, avg_reward = play_episodes(enviorment2, n_episode, opt_policy2, random=False)

    print(f'Total wins with Policy iteration: {wins}')
    print(f"Average rewards with Policy iteration: {avg_reward}")
