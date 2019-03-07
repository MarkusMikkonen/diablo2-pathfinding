import json
import time

import matplotlib.pyplot as plt
import numpy as np

# Edit target name here
target_name = 'Charsi'

# Load game map, q_table and coordinates
game_map = np.load('array_map.npy')
q_table = np.load(f'q_table_{target_name}.npy')
with open('coordinates.json') as infile:
    coordinates = json.load(infile)

# Clean this up, load start and target coordinates
start = [coordinates['spawn_point']['y'], coordinates['spawn_point']['x']]
target = [coordinates['targets'][target_name]['y'], coordinates['targets'][target_name]['x']]

# Possible moves (N, Ne, E, Se, S, Sw, W, Nw)
moves = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

# Create coordinates for all walkable positions.
possible_states = np.transpose(np.nonzero(game_map)).tolist()

# Create coordinates for blocked positions.
blocked_states = np.transpose(np.where(game_map == 0)).tolist()

# Run 100 iterations using optimal policy.
for i in range(100):
    coord_state = start
    state = possible_states.index(coord_state)
    route = [coord_state]

    wall, penalty = 0, 0

    done = False

    print(f'Epoch {i}')

    start_time = time.time()

    while not done:
        # Select action through max value from q-table
        action = np.argmax(q_table[state])

        # Set next coordinate state
        coord_next_state = [coord_state[0] - moves[action][0], coord_state[1] - moves[action][1]]

        # If location blocked, keep last state
        if coord_next_state in blocked_states:
            coord_next_state = coord_state
            reward = -10
            wall += 1

        # Done, congrats!
        elif coord_next_state == target:
            reward = 100
            done = True

        # Give -1 reward every timestep if not end
        else:
            reward = -1
            penalty += 1

        # Set states for next iteration
        next_state = possible_states.index(coord_next_state)

        state = next_state
        coord_state = coord_next_state

        route.append(coord_state)

    # Print some statistics
    epoch_time = time.time() - start_time
    print(f'Epoch time: {epoch_time}')
    print(f'Hitted wall {wall} times, penalties {penalty}')
    print(f'Route used: {route}\n')

# Save last route to image
route = np.array(route)
plt.imshow(game_map, cmap='gray')
plt.plot(route[:, 1], route[:, 0], 'ro')
plt.figsave()
