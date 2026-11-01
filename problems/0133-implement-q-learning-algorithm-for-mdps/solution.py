import numpy as np

def q_learning(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes):

    Q = np.array([[0.0 for _ in range(num_actions)] for _ in range(num_states)])
    for episode in range(num_episodes):
        state = np.random.choice([i for i in range(num_states) if i not in terminal_states])
        while True:
            rnd = np.random.rand()
            if rnd<epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[state, :])
            next_state = np.random.choice(num_states, p=P[state, action])
            if next_state in terminal_states:
                r = R[state,action]
            else:
                r = R[state,action] + gamma*np.max(Q[next_state, :])
            Q[state, action] = Q[state, action] + alpha*(r -  Q[state, action])
            if next_state in terminal_states:
                break
            state = next_state

    return Q 

