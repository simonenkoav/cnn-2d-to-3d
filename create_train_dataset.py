import pickle
import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
from enum import Enum

focal_length = 2.41421356

train_size = 10  # count of deviations for one position
image_side = 20

array_form = (train_size, image_side, image_side)
train_data = np.zeros(array_form)

image_form = (image_side, image_side)
empty_image_array = np.zeros(image_form)


class Axes(Enum):
    X = 0
    Y = 1
    Z = 2


def get_coordinate(start, end):
    return int(random.uniform(start, end))


def add_deviation(coordinate):
    return get_coordinate(coordinate - 2, coordinate + 2)


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


def get_sin_cos(degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return c, s


def make_rotation_matrix_about_z_axis(degrees):
    c, s = get_sin_cos(degrees)
    R = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, -s, 0,
                                                        s, c, 0,
                                                        0, 0, 1))
    return R


def make_rotation_matrix_about_x_axis(degrees):
    c, s = get_sin_cos(degrees)
    R = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(1, 0, 0,
                                                        0, c, -s,
                                                        0, s, c))
    return R


def make_rotation_matrix_about_y_axis(degrees):
    c, s = get_sin_cos(degrees)
    R = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, 0, s,
                                                        0, 1, 0,
                                                        -s, 0, c))
    return R


def add_transpose(R):
    RT = np.zeros((R.shape[0], R.shape[1] + 1))
    RT[:, :-1] = R
    RT[0][-1] = 0
    RT[1][-1] = 0
    RT[2][-1] = 5
    return RT


def make_rt_matrix(axis, degrees):
    if axis == Axes.X:
        R = make_rotation_matrix_about_x_axis(degrees)
        return add_transpose(R)
    elif axis == Axes.Y:
        R = make_rotation_matrix_about_y_axis(degrees)
        return add_transpose(R)
    elif axis == Axes.Z:
        R = make_rotation_matrix_about_z_axis(degrees)
        return add_transpose(R)


# coordinates should be passed in form [[point0_x, point0_y, point0_z], [point1_x, point1_y, point1_z], ...]
def transform_to_homogeneous(coordinates):
    new_coordinates = np.zeros((coordinates.shape[0], coordinates.shape[1] + 1))
    new_coordinates[:, :-1] = coordinates
    new_coordinates[0][-1] = 1
    new_coordinates[1][-1] = 1
    new_coordinates[2][-1] = 1
    new_coordinates[3][-1] = 1
    return new_coordinates


def generate_rotation_matrices():
    rotations = np.zeros((int(360 / 20), 3, 3))  # shape is (count of rotation matrices, size of single rotation matrix)
    i = 0
    index = 0
    while i < 360:
        rotations[index] = make_rotation_matrix_about_z_axis(i)
        index += 1
        i += 20
    return rotations


def get_projection(point):
    x = (focal_length / point[-1]) * point[0]
    y = (focal_length / point[-1]) * point[1]
    return x, y


def get_normal_coordinates(in_vector_coordinates):
    new_coordinates = copy.deepcopy(in_vector_coordinates)
    for point in new_coordinates:
        point0 = point[0]
        point1 = point[1]
        point[0] = (point1 - 9) / 5
        point[1] = (9 - point0) / 5

    return np.array(new_coordinates)


def get_2d(all_cordinates):
    coordinates_2d = np.array(all_cordinates[:-1, :]).transpose()
    return coordinates_2d


# square place in the center of image without rotations
perfect_coordinates = [[4, 14, 0], [4, 4, 0], [14, 4, 0], [14, 14, 0]]  # coordinates of 4 points


for i in range(train_size):
    new_coordinates = add_deviations(perfect_coordinates)
    image_with_points = set_points_on_image(empty_image_array, new_coordinates)
    train_data[i] = image_with_points


perfect_coordinates = [[4, 14, 0], [4, 4, 0], [14, 4, 0], [14, 14, 0]]  # coordinates of 4 points

# TODO rotate in cycle: in range 0°-360° with step of 20 for all 3 axes -> 58320 (10*18*18*18) correct images of square


RT = make_rt_matrix(Axes.X, 45)
print("RT = " + str(RT))
normal_coordinates = get_normal_coordinates(perfect_coordinates)
print("normal_coordinates = " + str(normal_coordinates))
normal_coordinates = transform_to_homogeneous(normal_coordinates)
normal_coordinates = np.transpose(normal_coordinates)
print("normal_coordinates (homogeneous) = " + str(normal_coordinates))
rotated = np.dot(RT, normal_coordinates)
rotated = np.transpose(rotated)
rotated = np.array(rotated)  # if i won't do it, when iterating each element would be presented in the form [[1 2 3]]
# i don't know why this shit happens

print("rotated = " + str(rotated))

for point in rotated:
    point[0], point[1] = get_projection(point)

print("rotated (projections) = " + str(rotated))
print("2d = " + str(get_2d(np.transpose(rotated))))