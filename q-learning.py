import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

target_name = 'Waypoint'

game_map = np.load('array_map.npy')

with open('coordinates.json') as infile:
    coordinates = json.load(infile)


start = [coordinates['spawn_point']['y'], coordinates['spawn_point']['x']]
target = [coordinates['targets'][target_name]['y'], coordinates['targets'][target_name]['x']]

# Possible moves (N, Ne, E, Se, S, Sw, W, Nw)
moves = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

# Create coordinates for all walkable positions.
possible_states = np.transpose(np.nonzero(game_map)).tolist()

# Create coordinates for blocked positions.
blocked_states = np.transpose(np.where(game_map == 0)).tolist()

# Create or load Q-table for State-Action pairs using possible states and actions.
if os.path.exists(f'q_table_{target_name}.npy'):
    q_table = np.load(f'q_table_{target_name}.npy')
else:
    q_table = np.zeros([len(possible_states), len(moves)])

print(q_table.shape)

# Hyperparameters
alpha = 0.5
gamma = 0.75
epsilon = 0.1

for i in range(100000):
    coord_state = start
    state = possible_states.index(coord_state)

    done = False

    print(f'Epoch {i}')

    start_time = time.time()

    route = [coord_state]

    while not done:
        # Explore mode, change epsilon higher if want to use Q-values sooner
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(len(moves))
        # Use learned values
        else:
            action = np.argmax(q_table[state])

        # Set next coordinate state
        coord_next_state = [coord_state[0] - moves[action][0], coord_state[1] - moves[action][1]]

        # If location blocked, keep last state
        if coord_next_state in blocked_states:
            coord_next_state = coord_state
            reward = -1

        # Done, congrats!
        elif coord_next_state == target:
            reward = 10
            done = True

        # Give -1 reward every timestep if not end
        else:
            reward = -1

        # Set next state searching the index of next coordinates
        next_state = possible_states.index(coord_next_state)

        # Update Q-values
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        # Set states for next iteration
        state = next_state
        coord_state = coord_next_state

        route.append(coord_state)

    epoch_time = time.time() - start_time
    print(f'Epoch time: {epoch_time}')
    print(f'Steps: {len(route)}')
    np.save(f'q_table_{target_name}.npy', q_table)
