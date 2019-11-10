import numpy as np
from itertools import product
from scipy.stats import poisson


POISSON_UPPER_BOUND = 11


def state_value_compute(state, action, states_value, gamma=0.9):
    state_value = 0.0
    state_value -= 2 * abs(action)
    for (lent1, lent2) in product(range(POISSON_UPPER_BOUND), range(POISSON_UPPER_BOUND)):
        lent_prob = poisson.pmf(lent1, 3) * poisson.pmf(lent2, 4)
        num1 = min(state[0] - action, 20)
        num2 = min(state[1] + action, 20)
        real_lent1 = min(num1, lent1)
        real_lent2 = min(num2, lent2)
        for (return1, return2) in product(range(POISSON_UPPER_BOUND), range(POISSON_UPPER_BOUND)):
            num1_ = min(num1 - real_lent1 + return1, 20)
            num2_ = min(num2 - real_lent2 + return2, 20)
            return_prob = poisson.pmf(return1, 3) * poisson.pmf(return2, 2)
            total_prob = lent_prob * return_prob
            state_value += total_prob * ((real_lent1+real_lent2)*10+gamma*states_value[num1_, num2_])
    return state_value


if __name__ == "__main__":
    value = np.zeros((21, 21))
    policy = np.zeros(value.shape, dtype=np.int)
    actions = np.arange(-5, 6)
    
    while True:

        # states_eval
        while True:
            value_ = np.zeros_like(value)
            for (i, j) in product(range(21), range(21)):
                state = (i, j)
                action = policy[i, j]
                value_[i, j] = state_value_compute(state, action, value)
            max_value_change = abs(value_ - value).max()
            print('max value change {}'.format(max_value_change))
            if max_value_change < 1e-4:
                break
            value = value_

        policy_stable = True
        for (i, j) in product(range(21), range(21)):
            old_action = actions[i, j]
            state = (i, j)
            q_sa = []
            for action in actions:
                if 0 <= action <= i or 0 <= -action <= j:
                    q_sa.append(state_value_compute(state, action, value))
                else:
                    q_sa.append(-np.inf)
                new_action = actions[np.argmax(q_sa)]
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:
                    policy_stable = False
        if policy_stable:
            print(value, policy)
            break            
