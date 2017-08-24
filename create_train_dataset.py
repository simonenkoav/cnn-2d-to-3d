import utils.file_utils as file_utils
import numpy as np
import random
import copy
from enum import IntEnum

focal_length = 2.41421356

image_side = 20

count_of_object_points = 4

angle_step = 20  # we woukd take angles in the range 0-360° with the shep=angle_step

# square placed in the center of image without rotations
perfect_coordinates = np.array([[4, 14, 0], [4, 4, 0], [14, 4, 0], [14, 14, 0]])  # coordinates of 4 points

count_of_deformations = 10  # count of deviations for one position

# counts are counts of rotation angles about each axis
x_count = y_count = 360 / angle_step - 1
z_count = 360 / angle_step
count_of_train_data = int(count_of_deformations * x_count * y_count * z_count)
train_data = np.zeros((count_of_train_data, count_of_object_points, 2))
train_labels = np.zeros((count_of_train_data, count_of_object_points, 3))

data_filename = "TRAIN_DATA"
labels_filename = "TRAIN_LABELS"


class Axes(IntEnum):
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
        for i in range(len(point) - 1):  # len(point) - 1 because we don't take into account z coordinates
            point[i] = add_deviation(point[i])

    return new_coordinates


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


def choose_rotation_matrix(axis, degrees):
    if axis == Axes.X:
        return make_rotation_matrix_about_x_axis(degrees)
    elif axis == Axes.Y:
        return make_rotation_matrix_about_y_axis(degrees)
    elif axis == Axes.Z:
        return make_rotation_matrix_about_z_axis(degrees)


# axes__to_degrees is the dictionary of three axes and corresponding degrees
def make_rt_matrix(axes__to_degrees):
    assert len(axes__to_degrees) == len(Axes)

    R_x = make_rotation_matrix_about_x_axis(axes__to_degrees[Axes.X])
    R_y = make_rotation_matrix_about_y_axis(axes__to_degrees[Axes.Y])
    R_z = make_rotation_matrix_about_z_axis(axes__to_degrees[Axes.Z])

    R = np.dot(R_z, R_y)
    R = np.dot(R, R_x)

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


def generate_rotation_angles():
    x_count = y_count = int(360 / angle_step - 1)
    z_count = int(360 / angle_step)
    xy_angles = list(range(0, 360, angle_step))
    xy_angles.remove(180)
    z_angles = list(range(0, 360, angle_step))
    count = x_count * y_count * z_count
    angles = np.zeros((count, 3))
    for i_z in range(z_count):
        for i_y in range(y_count):
            for i_x in range(x_count):
                index = i_x + x_count * i_y + x_count * y_count * i_z
                angles[index] = [xy_angles[i_x], xy_angles[i_y], z_angles[i_z]]

    return angles


def get_projection(point):
    x = point[0]
    y = point[1]
    if point[-1] != 0:
        x = (focal_length / point[-1]) * x
        y = (focal_length / point[-1]) * y
    return x, y


def get_normal_coordinates(in_vector_coordinates):
    new_coordinates = np.array(copy.deepcopy(in_vector_coordinates), dtype='float32')
    for point in new_coordinates:
        point0 = point[0]
        point1 = point[1]
        point[0] = (point1 - 9.0) / 5.0
        point[1] = (9.0 - point0) / 5.0
    return new_coordinates


def get_2d(all_coordinates):
    coordinates_2d = np.array(all_coordinates[:-1, :]).transpose()
    return coordinates_2d


def get_3d(homogeneous_coordinates):
    new_coordinates = np.array(homogeneous_coordinates[:, :-1])
    return new_coordinates


def save_data(xy, label_3d_points, index):
    train_data[index] = xy
    train_labels[index] = label_3d_points


if __name__ == "__main__":
    # shape -- (count of deformations, count of object points, count of coordinates of each point -- x, y, z, 1 --
    # because homogeneous)
    deformed = np.zeros((count_of_deformations, count_of_object_points, 3 + 1))

    for i in range(count_of_deformations):
        deformed_coordinates = add_deviations(perfect_coordinates)
        normal_coordinates = get_normal_coordinates(deformed_coordinates)
        normal_coordinates = transform_to_homogeneous(normal_coordinates)
        deformed[i] = normal_coordinates

    angles_list = generate_rotation_angles()

    for i_coordinates in range(len(deformed)):
        coordinates = deformed[i_coordinates]
        for i_angles in range(len(angles_list)):
            angles = angles_list[i_angles]
            RT = make_rt_matrix({Axes.X: angles[Axes.X],
                                 Axes.Y: angles[Axes.Y],
                                 Axes.Z: angles[Axes.Z]})
            rotated = np.dot(RT, coordinates)
            rotated = np.transpose(rotated)
            # if i won't do the shit on the next line,
            # it would be presented in the form [[1 2 3]] when iterating each element.
            # i don't know why this shit happens
            rotated = np.array(rotated)

            for point in rotated:
                point[0], point[1] = get_projection(point)

            only_xy = get_2d(np.transpose(rotated))
            only_xyz = get_3d(coordinates)

            index_of_train_data = i_coordinates * len(angles_list) + i_angles
            save_data(only_xy, only_xyz, index_of_train_data)

    file_utils.save_data_to_file(train_data, train_labels, data_filename, labels_filename)


