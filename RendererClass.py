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

    result = [i, j, k]  # np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 - y0, z1 - z0])
    return result


def calculate_cos_to_triangle(x0: float, x1: float, x2: float, y0: float, y1: float,
                              y2: float, z0: float, z1: float, z2: float,
                              light_direction_vector=[0.0, 0.0, 1.]) -> float:
    normal_coordinates = calculate_normal_to_triangle(x0, x1, x2, y0, y1, y2, z0, z1, z2)

    norma_normal_to_triangle = np.linalg.norm(normal_coordinates)
    normalized_vector = normal_coordinates / norma_normal_to_triangle

    norma_light = np.linalg.norm(light_direction_vector)
    normalized_light = light_direction_vector / norma_light

    result = np.dot(normalized_vector, normalized_light)

    return result


def calculate_new_point_position(model, x0, x1, x2, y0, y1, y2,z0, z1, z2, R):
    point1_matrix = np.array([
        x0,
        y0,
        z0
    ])

    point2_matrix = np.array([
        x1,
        y1,
        z1
    ])

    point3_matrix = np.array([
        x2,
        y2,
        z2
    ])

    # print(R)

    # print(point1_matrix)

    result1 = R @ point1_matrix
    result2 = R @ point2_matrix
    result3 = R @ point3_matrix
    # print(result1)

    x0 = result1[0]
    y0 = result1[1]
    z0 = result1[2]

    x1 = result2[0]
    y1 = result2[1]
    z1 = result2[2]

    x2 = result3[0]
    y2 = result3[1]
    z2 = result3[2]

    return x0, x1, x2, y0, y1, y2,z0, z1, z2


def draw_with_light(x0: float, x1: float, x2: float, y0: float, y1: float,
                    y2: float, z0: float, z1: float, z2: float, image, z_buffer, color, intensity):
    light_cos = calculate_cos_to_triangle(x0, x1, x2, y0, y1, y2, z0, z1, z2)
    if light_cos > 0:
        return

    x_min = int(min(x0, x1, x2) - 1)
    y_min = int(min(y0, y1, y2) - 1)
    x_max = int(max(x0, x1, x2) + 1)
    y_max = int(max(y0, y1, y2) + 1)

    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0

    if x_max > image.shape[1]:
        x_max = image.shape[1] - 1
    if y_max > image.shape[0]:
        y_max = image.shape[0] - 1

    divider = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    if divider == 0:
        return
    for y in np.arange(y_min - 1, y_max + 1):
        for x in np.arange(x_min - 1, x_max + 1):
            I0, I1, I2 = determine_baricentric(x, x0, x1, x2, y, y0, y1, y2, divider)
            if I0 >= 0 and I1 >= 0 and I2 >= 0:
                z = (I0 * z0 + I1 * z1 + I2 * z2)
                if z < z_buffer[y, x]:
                    pixel_intensity = -(intensity[0] * I0 + intensity[1] * I1 + intensity[2] * I2)
                    if pixel_intensity < 0:
                        pixel_intensity = 0
                    image[y, x] = color * pixel_intensity
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

    draw_with_light(model.scale, x0, x1, x2, y0, y1, y2, z0, z1, z2, image, zbuffer, color)


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
        print(f"Итерация {i} / {total_faces}, время {end_time - start_time}")


# def draw_iteration_guro_shading(image: np.ndarray, color: list[int], model: ObjModelClass.ObjModel,
#                                           polygon_number: int,zbuffer: np.ndarray):
def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    # Нормаль к вершинам полигона
    n = np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 - y0, z1 - z0])
    return n

def get_normals_for_points(model):
    normals = [[0, 0, 0]] * len(model.vertices)
    for i in model.faces:
        x0 = model.vertices[i[0]-1][0]
        y0 = model.vertices[i[0]-1][1]
        z0 = model.vertices[i[0]-1][2]

        x1 = model.vertices[i[1] - 1][0]
        y1 = model.vertices[i[1] - 1][1]
        z1 = model.vertices[i[1] - 1][2]

        x2 = model.vertices[i[2] - 1][0]
        y2 = model.vertices[i[2] - 1][1]
        z2 = model.vertices[i[2] - 1][2]

        np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 - y0, z1 - z0])
        norma = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        for j in range(3):
            normals[i[j]-1] += norma
    return normals

def get_guro_shading(lambda1, lambda2, lambda3, norma, light):
    light_norma = np.linalg.norm(light)
    l1 = np.cross(norma[0], light) / (np.linalg.norm(norma[0]) * light_norma)
    l2 = np.cross(norma[1], light) / (np.linalg.norm(norma[1]) * light_norma)
    l3 = np.cross(norma[2], light) / (np.linalg.norm(norma[2]) * light_norma)
    intensity = np.linalg.norm((lambda1 * l1 + lambda2 * l2 + lambda3 * l3))
    return intensity

def draw_guro_shading(x0: float, x1: float, x2: float, y0: float, y1: float,
                    y2: float, old_z0: float, old_z1: float, old_z2: float, image, z_buffer, color, normals):
    light = [0, 0, 1]
    x_min = int(min(x0, x1, x2) - 1)
    y_min = int(min(y0, y1, y2) - 1)
    x_max = int(max(x0, x1, x2) + 1)
    y_max = int(max(y0, y1, y2) + 1)

    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0

    if x_max > image.shape[1]:
        x_max = image.shape[1] - 1
    if y_max > image.shape[0]:
        y_max = image.shape[0] - 1

    divider = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    if divider == 0:
        return

    for y in np.arange(y_min - 1, y_max + 1):
        for x in np.arange(x_min - 1, x_max + 1):
            I0, I1, I2 = determine_baricentric(x, x0, x1, x2, y, y0, y1, y2, divider)
            if I0 >= 0 and I1 >= 0 and I2 >= 0:
                z = (I0 * old_z0 + I1 * old_z1 + I2 * old_z2)
                if z < z_buffer[y, x]:
                    pixel_intensity = get_guro_shading(I0, I1, I2, normals, light)
                    if pixel_intensity < 0:
                        pixel_intensity = 0
                    color_with_intensity = [int(item * pixel_intensity) for item in color]
                    image[y, x] = color_with_intensity
                    z_buffer[y, x] = z


# определяем функцию для проективного преобразования точки
def projective_transform(x, y, z, ax, ay, image):
    # создаем матрицу проективного преобразования
    matrix = [[ax, 0, image.shape[1] / 2],
              [0, ay, 200],
              [0, 0, 1]]
    coord = [x, y, 1]
    # умножаем матрицу проективного преобразования на точку для получения преобразованной точки
    res = np.dot(matrix, coord)
    return res

def draw_model_texture(image: Image, color: list[int], model: ObjModelClass.ObjModel,
                            zbuffer: np.ndarray, texture_np: np.ndarray,
                        rotate_x:int = 0, rotate_y:int = 0, rotate_z:int = 0):
    """
    Отрисовка модели циклом по полигонам
    :param zbuffer:
    :param color:
    :param image:
    :param model:
    :return:
    """
    n = get_normals_for_points(model)
    total_iter = len(model.faces)
    print("Total iterations:", total_iter)
    for i, face in enumerate(model.faces):
        print(f"Итерация {i} / {total_iter}")
        normals = [n[face[0] - 1], n[face[1] - 1], n[face[2] - 1]]
        x0 = model.vertices[face[0] - 1][0]
        y0 = model.vertices[face[0] - 1][1]
        z0 = model.vertices[face[0] - 1][2]

        x1 = model.vertices[face[1] - 1][0]
        y1 = model.vertices[face[1] - 1][1]
        z1 = model.vertices[face[1] - 1][2]

        x2 = model.vertices[face[2] - 1][0]
        y2 = model.vertices[face[2] - 1][1]
        z2 = model.vertices[face[2] - 1][2]

        u0 = model.textures[face[0]-1][0]
        v0 = model.textures[face[0]-1][1]
        u1 = model.textures[face[1] - 1][0]
        v1 = model.textures[face[1] - 1][1]
        u2 = model.textures[face[2] - 1][0]
        v2 = model.textures[face[2] - 1][1]

        scale_a = 12
        scale_b = 12

        R = calculate_matrix_r(rotate_x, rotate_y, rotate_z)

        x0, x1, x2, y0, y1, y2,z0, z1, z2 = calculate_new_point_position(model, x0, x1, x2, y0, y1, y2,z0, z1, z2, R)

        point1 = projective_transform(x0,y0,z0, scale_a, scale_b, image)
        point2 = projective_transform(x1, y1, z1, scale_a, scale_b, image)
        point3 = projective_transform(x2, y2, z2, scale_a, scale_b, image)



        if point1[0] > 0 and point1[1] > 0 and  point2[0] > 0 and point2[1] > 0 and point3[0] > 0 and point3[1] > 0:
            new_x0 = point1[0]
            new_y0 = point1[1]
            new_z0 = point1[2]

            new_x1 = point2[0]
            new_y1 = point2[1]
            new_z1 = point2[2]

            new_x2 = point3[0]
            new_y2 = point3[1]
            new_z2 = point3[2]
            draw_texture(new_x0, new_x1, new_x2, new_y0, new_y1, new_y2, z0, z1, z2,image,zbuffer, normals, texture_np,
                         u0, u1, u2, v0, v1,  v2)


def draw_texture(x0: float, x1: float, x2: float, y0: float, y1: float,
                    y2: float, old_z0: float, old_z1: float, old_z2: float, image, z_buffer,  normals, texture: np.ndarray,
                 u0, u1, u2, v0, v1,  v2
                 ):
    light = [0, 0, 1]
    x_min = int(min(x0, x1, x2) - 1)
    y_min = int(min(y0, y1, y2) - 1)
    x_max = int(max(x0, x1, x2) + 1)
    y_max = int(max(y0, y1, y2) + 1)

    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0

    if x_max > image.shape[1]:
        x_max = image.shape[1] - 1
    if y_max > image.shape[0]:
        y_max = image.shape[0] - 1

    divider = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    if divider == 0:
        return

    for y in np.arange(y_min - 1, y_max + 1):
        for x in np.arange(x_min - 1, x_max + 1):
            I0, I1, I2 = determine_baricentric(x, x0, x1, x2, y, y0, y1, y2, divider)
            if I0 >= 0 and I1 >= 0 and I2 >= 0:
                z = (I0 * old_z0 + I1 * old_z1 + I2 * old_z2)
                if z < z_buffer[y, x]:
                    color_texture = texture[
                        round(texture.shape[1] * (I0 * u0 + I1 * u1 + I2 * u2))][
                        round(texture.shape[0] * (I0 * v0 + I1 * v1 + I2 * v2))
                    ]

                    pixel_intensity = get_guro_shading(I0, I1, I2, normals, light)
                    if pixel_intensity < 0:
                        pixel_intensity = 0
                    color_with_intensity = pixel_intensity * color_texture
                    image[y, x] = color_with_intensity
                    z_buffer[y, x] = z


# определяем функцию для проективного преобразования точки
def projective_transform(x, y, z, ax, ay, image):
    # создаем матрицу проективного преобразования
    matrix = [[ax, 0, image.shape[1] / 2],
              [0, ay, 200],
              [0, 0, 1]]
    coord = [x, y, 1]
    # умножаем матрицу проективного преобразования на точку для получения преобразованной точки
    res = np.dot(matrix, coord)
    return res

def draw_model_guro_shading(image: Image, color: list[int], model: ObjModelClass.ObjModel,
                            zbuffer: np.ndarray):
    """
    Отрисовка модели циклом по полигонам
    :param zbuffer:
    :param color:
    :param image:
    :param model:
    :return:
    """
    n = get_normals_for_points(model)
    print("Total iterations:", len(model.faces))
    for i, face in enumerate(model.faces):
        print("Iteration: ", i)
        normals = [n[face[0] - 1], n[face[1] - 1], n[face[2] - 1]]
        x0 = model.vertices[face[0] - 1][0]
        y0 = model.vertices[face[0] - 1][1]
        z0 = model.vertices[face[0] - 1][2]

        x1 = model.vertices[face[1] - 1][0]
        y1 = model.vertices[face[1] - 1][1]
        z1 = model.vertices[face[1] - 1][2]

        x2 = model.vertices[face[2] - 1][0]
        y2 = model.vertices[face[2] - 1][1]
        z2 = model.vertices[face[2] - 1][2]
        scale_a = 6
        scale_b = 6
        point1 = projective_transform(x0,y0,z0, scale_a, scale_b, image)
        point2 = projective_transform(x1, y1, z1, scale_a, scale_b, image)
        point3 = projective_transform(x2, y2, z2, scale_a, scale_b, image)
        if point1[0] > 0 and point1[1] > 0 and  point2[0] > 0 and point2[1] > 0 and point3[0] > 0 and point3[1] > 0:
            new_x0 = point1[0]
            new_y0 = point1[1]
            new_z0 = point1[2]

            new_x1 = point2[0]
            new_y1 = point2[1]
            new_z1 = point2[2]

            new_x2 = point3[0]
            new_y2 = point3[1]
            new_z2 = point3[2]
            draw_guro_shading(new_x0, new_x1, new_x2, new_y0, new_y1, new_y2, z0, z1, z2,image,zbuffer, color, normals)


