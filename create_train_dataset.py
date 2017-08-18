import pickle
import numpy as np
import random
import copy

train_size = 10
image_side = 20

array_form = (train_size, image_side, image_side)
train_data = np.zeros(array_form)

image_form = (image_side, image_side)
empty_image_array = np.zeros(image_form)


def get_coordinate(start, end):
    return int(random.uniform(start, end))


def add_deviation(coordinate):
    return get_coordinate(coordinate - 1, coordinate + 1)


def add_deviations(coordinates):
    new_coordinates = copy.deepcopy(coordinates)
    for point in new_coordinates:
        for coordinate in point:
            coordinate_index = point.index(coordinate)
            point[coordinate_index] = add_deviation(coordinate)

    return new_coordinates


def set_points_on_image(image_array, points_coordinates):
    image_array_with_points = copy.deepcopy(image_array)
    for point in points_coordinates:
        image_array_with_points[point[0], point[1]] = 1
    return image_array_with_points


# square place in the center of image without rotations
perfect_coordinates = [[4, 4], [4, 14], [14, 4], [14, 14]]  # coordinates of 4 points


for i in range(train_size):
    new_coordinates = add_deviations(perfect_coordinates)
    image_with_points = set_points_on_image(empty_image_array, new_coordinates)
    train_data[i] = image_with_points


perfect_coordinates = [[4, 4], [4, 14], [14, 4], [14, 14]]  # coordinates of 4 points

# TODO rotate in cycle: in range 0°-360° with step of 4 for all 3 axes
