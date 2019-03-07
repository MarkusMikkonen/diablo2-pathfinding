import numpy as np

import cv2


def img_map_to_array():
    '''Read image and convert to array.'''
    game_map = cv2.imread('map.png', 0)
    game_map[game_map > 0] = 1
    return game_map


game_map = img_map_to_array()

np.save('array_map.npy', game_map)
