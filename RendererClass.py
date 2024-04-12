from time import time

from PIL import Image
from numba import njit
import numpy as np
import math

import ObjModelClass
import colors



def algorithm_bresenham(image: np.ndarray, x0, y0, x1, y1, color) -> None:
    """
    :param x0:
    :param y0:
    :param x1:
    :param y1:
    :param color:
    :param image: numpy ndarray
    """
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        derror += dy
        if derror > 0.5:
            derror -= 2 * (x1 - x0)
            y += y_update
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color


def determine_baricentric(x: int, x0: float, x1: float, x2: float, y: int, y0: float, y1: float,
                          y2: float, divider: float) -> np.array:
    '''

    :param x:
    :param x1:
    :param x2:
    :param y:
    :param y1:
    :param y2:
    :return: np.array: lambda0, lambda1, lambda2
    '''

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / divider
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / divider
    lambda2 = 1.0 - lambda0 - lambda1

    return (lambda0, lambda1, lambda2)




def calculate_normal_to_triangle(x0: float, x1: float, x2: float, y0: float, y1: float,
                          y2: float, z0: float, z1: float, z2: float) -> list:

    '''
    vec1 = np.array([point0.x - point1.x, point0.y - point1.y, point0.z - point1.z])
    vec2 = np.array([point0.x - point0.x, point0.y - point0.y, point0.z - point0.z])
    '''

    i = (y1 - y0) * (z1 - z2) - (y1 - y2) * (z1 - z0)
    j = -((x1 - x0) * (z1 - z2) - (x1 - x2) * (z1 - z0))
    k = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)

    result = [i, j, k] #np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 - y0, z1 - z0])
    return result



def calculate_cos_to_triangle(x0: float, x1: float, x2: float, y0: float, y1: float,
                          y2: float, z0: float, z1: float, z2: float, light_direction_vector=[0.0, 0.0, 1.]) -> float:
    normal_coordinates = calculate_normal_to_triangle(x0, x1, x2, y0, y1, y2, z0, z1, z2)

    norma_normal_to_triangle = np.linalg.norm(normal_coordinates)
    normalized_vector = normal_coordinates / norma_normal_to_triangle

    norma_light = np.linalg.norm(light_direction_vector)
    normalized_light = light_direction_vector / norma_light

    result = np.dot(normalized_vector, normalized_light)

    return result

def calculate_new_point_position(model, point1, point2, point3, R):

    point1_matrix = np.array([
        point1.x,
        point1.y,
        point1.z
    ])

    point2_matrix = np.array([
        point2.x,
        point2.y,
        point2.z
    ])

    point3_matrix = np.array([
        point3.x,
        point3.y,
        point3.z
    ])

    # print(R)

    # print(point1_matrix)

    result1 = R @ point1_matrix
    result2 = R @ point2_matrix
    result3 = R @ point3_matrix
    # print(result1)

    point1.x = result1[0]
    point1.y = result1[1]
    point1.z = result1[2]

    point2.x = result2[0]
    point2.y = result2[1]
    point2.z = result2[2]

    point3.x = result3[0]
    point3.y = result3[1]
    point3.z = result3[2]


    return point1, point2, point3


def draw_with_light(scale: float, x0: float, x1: float, x2: float, y0: float, y1: float,
                          y2: float, z0: float, z1: float, z2: float, image, z_buffer, color, i):
    light_cos = calculate_cos_to_triangle(x0, x1, x2, y0, y1, y2, z0, z1, z2)
    if light_cos > 0:
        return
    offset_y = 200
    offset_x = 500
    p_x0 = (x0 / z0) * scale + offset_x
    p_x1 = (x1 / z1) * scale + offset_x
    p_x2 = (x2 / z2) * scale + offset_x
    p_y0 = (y0 / z0) * scale + offset_y
    p_y1 = (y1 / z1) * scale + offset_y
    p_y2 = (y2 / z2) * scale + offset_y

    x_min = int(min(p_x0, p_x1, p_x2) - 1)
    y_min = int(min(p_y0, p_y1, p_y2) - 1)
    x_max = int(max(p_x0, p_x1, p_x2) + 1)
    y_max = int(max(p_y0, p_y1, p_y2) + 1)

    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0

    if x_max > image.shape[1]:
        x_max = image.shape[1] - 1
    if y_max > image.shape[0]:
        y_max = image.shape[0] - 1

    color_with_light = [abs(item * light_cos) for item in color]

    divider = ((p_x0 - p_x2) * (p_y1 - p_y2) - (p_x1 - p_x2) * (p_y0 - p_y2))
    if (divider==0):
        return
    z = 0
    for y in np.arange(y_min-1, y_max+1):
        for x in np.arange(x_min-1, x_max+1):
            baricentrics = determine_baricentric(x, p_x0, p_x1, p_x2, y, p_y0, p_y1, p_y2, divider)
            if baricentrics[0]>=0 and baricentrics[1]>=0 and baricentrics[2]>=0:
                z = (baricentrics[0] * z0 + baricentrics[1] * z1 + baricentrics[2] * z2)
                if z < z_buffer[y, x]:
                    image[y, x] = color_with_light
                    z_prev = z_buffer[y, x]
                    z_buffer[y, x] = z

def draw_with_rotation_by_index(image: Image, color: list[int], model: ObjModelClass.ObjModel, z_buffer: np.ndarray,
                                R: np.array, polygon_number: int):
    point1, point2, point3 = model.get_points_by_index(polygon_number)
    point1, point2, point3 = calculate_new_point_position(model, point1, point2, point3, R)
    offset_z = model.offset_z * 2
    x0 = point1.x
    x1 = point2.x
    x2 = point3.x
    y0 = point1.y
    y1 = point2.y
    y2 = point3.y
    z0 = point1.z + offset_z
    z1 = point2.z + offset_z
    z2 = point3.z + offset_z
    draw_with_light(model.scale, x0, x1, x2, y0, y1, y2, z0, z1, z2, image, z_buffer, color)


def calculate_matrix_r(rotate_x, rotate_y, rotate_z) -> np.array:
    # rotate_x, rotate_y, rotate_z = math.radians(rotate_x), math.radians(rotate_y), math.radians(rotate_z)
    sin_x, cos_x = np.sin(rotate_x), np.cos(rotate_x)
    sin_y, cos_y = np.sin(rotate_y), np.cos(rotate_y)
    sin_z, cos_z = np.sin(rotate_z), np.cos(rotate_z)
    matrix_x = np.array([
        [1, 0, 0],
        [0, cos_x, sin_x],
        [0, -sin_x, cos_x],
    ])

    matrix_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y],
    ])

    matrix_z = np.array([
        [cos_z, sin_z, 0],
        [-sin_z, cos_z, 0],
        [0, 0, 1]
    ])
    R = matrix_x @ matrix_y @ matrix_z
    return R


def draw_with_rotation(image: Image, color: list[int], model: ObjModelClass.ObjModel, z_buffer: np.ndarray,
                       rotate_x: int, rotate_y: int, rotate_z: int):

    R = calculate_matrix_r(rotate_x, rotate_y, rotate_z)
    total_faces = len(model.faces)
    print("total faces: ", total_faces)
    for i in range(1, len(model.faces) + 1):
        start_time = time()

        draw_with_rotation_by_index(image, color, model, z_buffer, R, i)

        end_time = time()
        print(f"Итерация {i} / {total_faces}, время {end_time - start_time}")

def draw_triangle_projective_transformation(image: np.ndarray, color: list[int], model: ObjModelClass.ObjModel,
                                            polygon_number: int,
                                            zbuffer: np.ndarray):
    point1, point2, point3 = model.get_points_by_index(polygon_number)

    offset_z = model.offset_z
    x0 = point1.x
    x1 = point2.x
    x2 = point3.x
    y0 = point1.y
    y1 = point2.y
    y2 = point3.y
    z0 = point1.z + offset_z
    z1 = point2.z + offset_z
    z2 = point3.z + offset_z

    draw_with_light(model.scale, x0, x1, x2, y0, y1, y2, z0, z1, z2, image, zbuffer, color, polygon_number)



def draw_model_projective_transformation(image: Image, color: list[int], model: ObjModelClass.ObjModel,
                                         zbuffer: np.ndarray):
    """
    Отрисовка модели циклом по полигонам
    :param image:
    :param model:
    :return:
    """
    total_faces = len(model.faces)
    print("total faces: ", total_faces)
    for i in range(1, total_faces + 1):
        start_time = time()

        draw_triangle_projective_transformation(image, color, model, i, zbuffer)

        end_time = time()
        print(f"Итерация {i} / {total_faces}, время {end_time-start_time}")

