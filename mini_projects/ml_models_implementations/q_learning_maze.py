import numpy as np

n = 12
m = 12

actions = np.array([i for i in range(n)])
states = np.array([i for i in range(m)])

reward = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1000, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])

q_values = np.zeros((m, n))

n_iterations = 1000
gamma = 0.75
alpha = 0.9
for i in range(n_iterations):

    s = np.random.choice(states)
    possible_actions = [index for index, value in enumerate(reward[s]) if value != 0]
    a = np.random.choice(possible_actions)

    td = reward[s][a] + gamma * q_values[a, np.argmax(q_values[a])] -  q_values[s][a]
    q_values[s][a] += alpha * td

s = 4
location_to_state = { 0: 'A',
 1:'B',
 2:'C',
 3:'D',
 4: 'E',
 5:'F',
 6:'G',
 7:'H',
 8:'I',
 9:'J',
 10:'K',
 11:'L'}
while s != 6:
    print(location_to_state[s])

    s = np.argmax(q_values[s])
