import numpy as np

gamma = 0.75
alpha = 0.9

# Defining the states
location_to_state = {'A': 0,
 'B': 1,
 'C': 2,
 'D': 3,
 'E': 4,
 'F': 5,
 'G': 6,
 'H': 7,
 'I': 8,
 'J': 9,
 'K': 10,
 'L': 11}

state_to_location = {state:location for location, state in location_to_state.items()}

# Defining the actions
n = 12
states = [i for i in range(n)]
actions = [i for i in range(n)]



# Defining the rewards
reward = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
 [1,0,1,0,0,1,0,0,0,0,0,0],
 [0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,0,0,0,0],
 [0,0,0,0,0,0,0,0,1,0,0,0],
 [0,1,0,0,0,0,0,0,0,1,0,0],
 [0,0,1,0,0,0,1,1,0,0,0,0],
 [0,0,0,1,0,0,1,0,0,0,0,1],
 [0,0,0,0,1,0,0,0,0,1,0,0],
 [0,0,0,0,0,1,0,0,1,0,1,0],
 [0,0,0,0,0,0,0,0,0,1,0,1],
 [0,0,0,0,0,0,0,1,0,0,1,0]])


# Making a function that returns the shortest route from a starting to ending location
def route(starting_location, ending_location):
    R_new = np.copy(reward)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000

    q_values = np.zeros(R_new.shape)
    
    # Find q-value values
    n_iterations = 1000
    for i in range(n_iterations):
        s = np.random.choice(states)
        possible_actions = [index for index, value in enumerate(reward[s]) if value != 0]
        a = np.random.choice(possible_actions)

        td = R_new[s][a] + gamma * q_values[a, np.argmax(q_values[a])] -  q_values[s][a]
        q_values[s][a] += alpha * td

    # make route
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(q_values[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location

    return route

print(route('E', 'G'))
