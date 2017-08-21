import pickle
import numpy as np
import random
import copy
import math

train_size = 10  # count of deviations for one position
image_side = 20

array_form = (train_size, image_side, image_side)
train_data = np.zeros(array_form)

image_form = (image_side, image_side)
empty_image_array = np.zeros(image_form)


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


def make_rotation_matrix_about_z_axis(degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, -s, 0,
                                                        s, c, 0,
                                                        0, 0, 1))
    return R


def make_rotation_matrix_about_x_axis(degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(1, 0, 0,
                                                        0, s, c,
                                                        0, c, s))
    return R


def rotate(coordinates, rotation_matrix):
    new_coordinates = copy.deepcopy(np.array(coordinates))

    for i in range(new_coordinates.shape[0]):
        point = np.array(new_coordinates[i])
        xyz = np.array(point)
        new_coordinates[i] = np.dot(rotation_matrix, xyz)

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


def make_camera_matrix(initial_fx_in_degrees, image_width, image_height):
    aspect = image_width / image_height;

    initial_fx_in_radians = np.radians(initial_fx_in_degrees);
    fy = 1.0 / math.atan(initial_fx_in_radians / 2.0);
    fx = fy / aspect;

    cx = image_width / 2.0;
    cy = image_height / 2.0;

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return camera_matrix


def get_normal_coordinates(in_vector_coordinates):
    new_coordinates = copy.deepcopy(in_vector_coordinates)
    for point in new_coordinates:
        for i in range(0, 2):
            point[i] = (point[i] - 9) / 5

    return new_coordinates


# square place in the center of image without rotations
perfect_coordinates = [[4, 4, 0], [4, 14, 0], [14, 4, 0], [14, 14, 0]]  # coordinates of 4 points


for i in range(train_size):
    new_coordinates = add_deviations(perfect_coordinates)
    image_with_points = set_points_on_image(empty_image_array, new_coordinates)
    train_data[i] = image_with_points


perfect_coordinates = [[4, 4, 0], [4, 14, 0], [14, 4, 0], [14, 14, 0]]  # coordinates of 4 points

print("perfect_coords = " + str(perfect_coordinates))
r = make_rotation_matrix_about_x_axis(45)
res = rotate(perfect_coordinates, r)

# res = [[2, 2, 2], [2, 7, 2], [14, 4, 0], [14, 14, 0]]
cam_matrix = make_camera_matrix(45, 2.0, 2.0)
print("res = " + str(res))
print("cam_matrix = " + str(cam_matrix))
res = np.transpose(res)
res = np.dot(cam_matrix, res)
print("test rotate about x (45°) = " + str(np.transpose(res)))

r = make_rotation_matrix_about_x_axis(60)
res = rotate(perfect_coordinates, r)
res = np.transpose(res)
res = np.dot(cam_matrix, res)
print("test rotate about x (60°)= " + str(np.transpose(res)))
# TODO rotate in cycle: in range 0°-360° with step of 20 for all 3 axes -> 58320 (10*18*18*18) correct images of square

rotations = generate_rotation_matrices()  # around Z axis

rotation_matrix = make_rotation_matrix_about_z_axis(90)
rotate(perfect_coordinates, rotation_matrix)

# ANOTHER ATTEMPT: (u v) = CAM_MATRIX * RT * XYZ_coordinates

print("\n\n-----")
rotation_M = make_rotation_matrix_about_x_axis(45)
rotation_M = np.transpose(rotation_M)

RT = np.zeros((rotation_M.shape[0], rotation_M.shape[1] + 1))
RT[:, :-1] = rotation_M
RT[-1][-1] = -2

print("cam_M = " + str(cam_matrix))
print("RT = " + str(RT))

res1 = np.dot(cam_matrix, RT)
print("res1 = " + str(res1))

normal_coordinates = np.array(get_normal_coordinates(perfect_coordinates))
print("normal_coordinates = " + str(normal_coordinates))
# new_coords = np.transpose(new_coords)
points_with_one = np.ones((normal_coordinates.shape[0], normal_coordinates.shape[1] + 1))
points_with_one[:, :-1] = normal_coordinates
points_with_one = np.transpose(points_with_one )
print("points_with_one = " + str(points_with_one))

res2 = np.dot(res1, points_with_one)
print("res2 = " + str(np.transpose(res2)))



