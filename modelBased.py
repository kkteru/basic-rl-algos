import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Solving MDP by DP')

parser.add_argument('--n', type=int, default=5,
                    help='The dimension of the grid')
parser.add_argument('--mod_pol_iter', type=int, default=3,
                    help='Number iterations for modified policy iteration')
parser.add_argument('--discount', type=float, default=0.9,
                    help='The discount rate for future rewards')
parser.add_argument('--exp_rate', type=float, default=0.9,
                    help='The discount rate for future rewards')

params = parser.parse_args()

n_states = params.n ** 2
n_actions = 4


def inject_exploration(policy):
    return np.ones((n_actions, n_states)) * (1 - params.exp_rate) / n_actions + policy * params.exp_rate


def get_neighbors(n, k):
    i = k // n
    j = k % n
    a = np.arange(n * n).reshape(n, n)

    right = a[min(n - 1, max(i, 0)), min(n - 1, max(j + 1, 0))]
    down = a[min(n - 1, max(i + 1, 0)), min(n - 1, max(j, 0))]
    left = a[min(n - 1, max(i, 0)), min(n - 1, max(j - 1, 0))]
    up = a[min(n - 1, max(i - 1, 0)), min(n - 1, max(j, 0))]

    return [right, down, left, up]


def policy_eval(policy, reward, trans, init_value, n_iter):

    # Value of each cell in the grid for a given policy
    # i.e. the expected reward that can be obtaineed by starting in this cell and following the given policy.
    value = init_value

    for _ in range(n_iter):
        # THE CRUX OF THIS FILE
        value = np.sum(policy * (reward + params.discount * (np.matmul(trans, value).squeeze())), axis=0)

    return value


def policy_improv(value):
    policy = np.zeros((n_actions, n_states))
    for i in range(n_states):
        neighbors = get_neighbors(params.n, i)
        policy[np.argmax([value[j] for j in neighbors]), i] = 1

    return policy


def policy_interation(reward, trans, n_iter=500):

    # Initial policy
    policy = np.ones((n_actions, n_states)) * (1 / n_actions)
    value = np.zeros(n_states)

    policies = []
    for _ in range(100):
        value = policy_eval(policy, reward, trans, value, n_iter)
        policy = policy_improv(value)
        policies.append(policy)
    return policies


# Transition matrix; deterministic for our environment
trans = np.zeros((n_actions, n_states, n_states))
for k in range(n_states):
    neighbors = get_neighbors(params.n, k)
    trans[0, k, neighbors[0]] = 1
    trans[1, k, neighbors[1]] = 1
    trans[2, k, neighbors[2]] = 1
    trans[3, k, neighbors[3]] = 1


a = np.arange(n_states).reshape(params.n, params.n)

# Rewards obtained after leaving a cell
reward = np.zeros((4, n_states))
reward[:, a[0, 0]] = 1
reward[:, a[0, -1]] = 10

policy_iter_policies = list(map(inject_exploration, policy_interation(reward, trans)))
mod_policy_iter_policies = list(map(inject_exploration, policy_interation(reward, trans, n_iter=params.mod_pol_iter)))
value_iter_policies = list(map(inject_exploration, policy_interation(reward, trans, n_iter=1)))

init_value = np.zeros(n_states)
policy_iter_values = np.array(list(map(policy_eval, policy_iter_policies, [reward] * len(policy_iter_policies), [trans] * len(policy_iter_policies), [init_value] * len(policy_iter_policies), [1000] * len(policy_iter_policies))))
mod_policy_iter_values = np.array(list(map(policy_eval, mod_policy_iter_policies, [reward] * len(mod_policy_iter_policies), [trans] * len(mod_policy_iter_policies), [init_value] * len(mod_policy_iter_policies), [1000] * len(mod_policy_iter_policies))))
value_iter_values = np.array(list(map(policy_eval, value_iter_policies, [reward] * len(value_iter_policies), [trans] * len(value_iter_policies), [init_value] * len(value_iter_policies), [1000] * len(value_iter_policies))))

policies = {
    'grid_dim': params.n,
    'policy_iter': policy_iter_values,
    'mod_policy_iter_%d' % params.mod_pol_iter: mod_policy_iter_values,
    'value_iter': value_iter_values
}

pkl.dump(policies, open('policies%dp%d.pkl' % (params.n, params.exp_rate * 10), 'wb'))


fig = plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(policy_iter_values[0:10, a[-1, 0]])
plt.plot(mod_policy_iter_values[0:10, a[-1, 0]])
plt.plot(value_iter_values[0:10, a[-1, 0]])
plt.title('Left bottom state value over the first 10 iterations')
plt.legend(['Policy iter', 'Modified policy iter', 'Value iter'])

plt.subplot(122)
plt.plot(policy_iter_values[0:10, a[-1, -1]])
plt.plot(mod_policy_iter_values[0:10, a[-1, -1]])
plt.plot(value_iter_values[0:10, a[-1, -1]])
plt.title('Right bottom state value over the first 10 iterations')
plt.legend(['Policy iter', 'Modified policy iter', 'Value iter'])
plt.savefig('n%dp%d.png' % (params.n, params.exp_rate * 10), dpi=fig.dpi)
