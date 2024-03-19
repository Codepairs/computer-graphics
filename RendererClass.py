from PIL import Image
import numpy as np
import math

import ObjModelClass
import colors


class Renderer:
    def __init__(self):
        self._image_size = self._h, self._w = 600, 800
        self._image_matrix = np.zeros(self._image_size, dtype=np.uint8)

    @staticmethod
    def update_point(image: np.ndarray, pos_y, pos_x, color) -> None:
        """
        Обновляет цвет точки по позиции
        :param image:
        :param pos_y: Позиция Y
        :param pos_x: Позиция X
        :param color: Цвет
        :return: None
        """
        image[pos_y][pos_x] = color

    @staticmethod
    def make_image_colored(image: np.ndarray, color) -> None:
        """
        Заполняет изображение цветом color
        :return: PIL.image
        """
        image.fill(color)

    @staticmethod
    def make_image_gradient(image: np.ndarray) -> None:
        """

        """
        for y in np.arange(0, image.shape[0]):
            for x in np.arange(0, image.shape[1]):
                image[y, x] = ((y + x) // 2) % 256

    @staticmethod
    def algorithm_dotted_line(image: np.ndarray, x0, y0, x1, y1, count, color) -> None:
        """
        :param color:
        :param count:
        :param y1:
        :param x1:
        :param x0:
        :param y0:
        :param image: numpy ndarray
        x = x1 + (1-t)(x0-x1)
        недостатки:
            из-за фиксированного count можем
                            либо нарисовать один пиксель несколько раз
                            либо не нарисовать вообще...
        """
        step = 1.0 / count
        for t in np.arange(0, 1, step):
            x = round((1.0 - t) * x0 + t * x1)
            y = round((1.0 - t) * y0 + t * y1)
            image[y, x] = color

    @staticmethod
    def algorithm_dotted_line_sqrt(image: np.ndarray, x0, y0, x1, y1, color) -> None:
        """
        :param color:
        :param y1:
        :param x1:
        :param y0:
        :param x0:
        :param image: numpy ndarray
        недостатки:
            медленное вычисление корня
        """
        count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        step = 1.0 / count
        for t in np.arange(0, 1, step):
            x = round((1.0 - t) * x0 + t * x1)
            y = round((1.0 - t) * y0 + t * y1)
            image[y, x] = color

    @staticmethod
    def algorithm_x_loop_line(image: np.ndarray, x0, y0, x1, y1, color) -> None:
        """
        :param color:
        :param y1:
        :param x1:
        :param y0:
        :param x0:
        :param image: numpy ndarray
        недостатки:
            если начало правее конца, то не рисует (без левой половины)
            идем по x -> шаг по x<чем по у -> рядом с Оу получим пунктир
        """
        for x in range(x0, x1):
            t = (x - x0) / (x1 - x0)
            y = int(y1 * t + y0 * (1.0 - t))
            image[y, x] = color

    @staticmethod
    def algorithm_x_loop_line_fixed(image: np.ndarray, x0, y0, x1, y1, color) -> None:
        """
        :param x0:
        :param y0:
        :param x1:
        :param y1:
        :param color:
        :param image: numpy ndarray
        недостатки:
            отсутствуют в явном виде
        """
        xchange = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            xchange = True

        if x0 > x1:
            x0, x1, = x1, x0
            y0, y1 = y1, y0

        for x in range(x0, x1):
            t = (x - x0) / (x1 - x0)
            y = int(y1 * t + y0 * (1.0 - t))
            if xchange:
                image[x, y] = color
            else:
                image[y, x] = color

    @staticmethod
    def algorithm_dy(image: np.ndarray, x0, y0, x1, y1, color) -> None:
        """
        :param x0:
        :param y0:
        :param x1:
        :param y1:
        :param color:
        :param image: numpy ndarray
        derror - погрешность вычислений, накапливаем ее, чтобы перейти вверх по y,
         затем сбрасываем
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
        dy = abs((y1 - y0) / (x1 - x0))
        derror = 0.0
        y_update = 1 if y1 > y0 else -1

        for x in range(x0, x1):
            derror += dy
            if derror > 0.5:
                derror -= 1.0
                y += y_update
            if xchange:
                image[x, y] = color
            else:
                image[y, x] = color

    @staticmethod
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

    @staticmethod
    def draw_polygon_face(image: Image, model: ObjModelClass.ObjModel, color: list[int], polygon_number: int):
        """
        Отрисовка полигона
        :param image:
        :param model:
        :param color:
        :return:
        """
        point1, point2, point3 = model.get_points_by_index(polygon_number)
        point1 = point1.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        point2 = point2.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        point3 = point3.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        Renderer.algorithm_bresenham(image, point1.x, point1.y, point2.x, point2.y, color)
        Renderer.algorithm_bresenham(image, point2.x, point2.y, point3.x, point3.y, color)
        Renderer.algorithm_bresenham(image, point3.x, point3.y, point1.x, point1.y, color)

    @staticmethod
    def draw_model_with_faces(image: Image, model: ObjModelClass.ObjModel, color: list[int]):
        """
        Отрисовка модели циклом по полигонам
        :param image:
        :param model:
        :param color:
        :return:
        """
        for i in range(1, len(model.faces) + 1):
            Renderer.draw_polygon_face(image, model, color, i)

    @staticmethod
    def determine_baricentric(x: int, x0: float, x1: float, x2: float, y: int, y0: float, y1: float,
                              y2: float) -> np.array:
        '''

        :param x:
        :param x1:
        :param x2:
        :param y:
        :param y1:
        :param y2:
        :return: np.array: lambda0, lambda1, lambda2
        '''
        divider = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
        lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / divider
        lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / divider
        lambda2 = 1.0 - lambda0 - lambda1

        return np.array([lambda0, lambda1, lambda2])

    @staticmethod
    def draw_triangle(image: np.ndarray, color: int, x0: float, x1: float, x2: float, y0: float, y1: float,
                      y2: float) -> None:
        xmin = min(x0, x1, x2)
        ymin = min(y0, y1, y2)
        xmax = max(x0, x1, x2)
        ymax = max(y0, y1, y2)

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0

        if xmax > image.shape[1]:
            xmax = image.shape[1]
        if ymax > image.shape[0]:
            ymax = image.shape[0]

        for y in np.arange(ymin, ymax):
            for x in np.arange(xmin, xmax):
                baricentrics = Renderer.determine_baricentric(x, x0, x1, x2, y, y0, y1, y2)
                # print(y, '  ', x, '  ', baricentrics)

                if all(lam >= 0 for lam in baricentrics):
                    image[y, x] = color

    @staticmethod
    def draw_model_with_random_color_polygons(image: Image, model: ObjModelClass.ObjModel):
        """
        Отрисовка модели циклом по полигонам
        :param image:
        :param model:
        :param color:
        :return:
        """
        for i in range(1, len(model.faces) + 1):
            Renderer.draw_model_triangle(image, model, i)

    @staticmethod
    def draw_model_triangle(image: np.ndarray, model: ObjModelClass.ObjModel, polygon_number: int) \
            -> None:
        point1, point2, point3 = model.get_points_by_index(polygon_number)
        point1 = point1.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        point2 = point2.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        point3 = point3.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)

        x_min = min(point1.x, point2.x, point3.x)
        y_min = min(point1.y, point2.y, point3.y)
        x_max = max(point1.x, point2.x, point3.x)
        y_max = max(point1.y, point2.y, point3.y)

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0

        if x_max > image.shape[1]:
            x_max = image.shape[1]
        if y_max > image.shape[0]:
            y_max = image.shape[0]

        random_color = colors.get_random_color()
        for y in np.arange(y_min, y_max):
            for x in np.arange(x_min, x_max):
                baricentrics = Renderer.determine_baricentric(x, point1.x, point2.x, point3.x, y, point1.y, point2.y,
                                                              point3.y)
                if all(lam >= 0 for lam in baricentrics):
                    image[y, x] = random_color

    @staticmethod
    def calculate_normal_to_triangle(point0, point1, point2) -> np.array:

        '''
        vec1 = np.array([point1.x - point2.x, point1.y - point2.y, point1.z - point2.z])
        vec2 = np.array([point1.x - point0.x, point1.y - point0.y, point1.z - point0.z])
        '''

        i = (point1.y - point0.y) * (point1.z - point2.z) - (point1.y - point2.y) * (point1.z - point0.z)
        j = ((point1.x - point0.x) * (point1.z - point2.z) - (point1.x - point2.x) * (point1.z - point0.z))
        k = (point1.x - point0.x) * (point1.y - point2.y) - (point1.x - point2.x) * (point1.y - point0.y)

        result = [-i, -j, -k]
        return result

    @staticmethod
    def calculate_cos_to_triangle(point1, point2, point3, light_direction_vector:np.array) -> float:
        normal_coordinates = Renderer.calculate_normal_to_triangle(point1, point2, point3)
        norma_normal_to_triangle = np.linalg.norm(normal_coordinates)

        normalized_vector = normal_coordinates / norma_normal_to_triangle

        norma_light = np.linalg.norm(light_direction_vector)
        normalized_light = light_direction_vector / norma_light
        # return (np.dot(normal_coordinates, light_direction_vector)) / (norma_normal_to_triangle * norma_light)
        result = np.dot(normalized_vector, normalized_light)

        return result

    @staticmethod
    def draw_model_with_random_color_and_cos(image: Image, model: ObjModelClass.ObjModel):
        """
        Отрисовка модели циклом по полигонам
        :param image:
        :param model:
        :return:
        """
        for i in range(1, len(model.faces) + 1):
            Renderer.draw_model_triangle_with_cos(image, model, i)
            # print(f"Итерация {i}")

    @staticmethod
    def draw_model_triangle_with_cos(image: np.ndarray, model: ObjModelClass.ObjModel, polygon_number: int) \
            -> None:
        point1, point2, point3 = model.get_points_by_index(polygon_number)
        point1 = point1.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        point2 = point2.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        point3 = point3.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)

        x_min = min(point1.x, point2.x, point3.x)
        y_min = min(point1.y, point2.y, point3.y)
        x_max = max(point1.x, point2.x, point3.x)
        y_max = max(point1.y, point2.y, point3.y)

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0

        if x_max > image.shape[1]:
            x_max = image.shape[1]
        if y_max > image.shape[0]:
            y_max = image.shape[0]

        random_color = colors.get_random_color()
        light_cos = Renderer.calculate_cos_to_triangle(point1, point2, point3)
        for y in np.arange(y_min, y_max):
            for x in np.arange(x_min, x_max):
                baricentrics = Renderer.determine_baricentric(x, point1.x, point2.x, point3.x, y, point1.y, point2.y,
                                                              point3.y)
                if all(lam >= 0 and light_cos < 0 for lam in baricentrics):
                    image[y, x] = random_color

    @staticmethod
    def draw_model_with_light(image: Image, color: list[int], model: ObjModelClass.ObjModel):
        """
        Отрисовка модели циклом по полигонам
        :param image:
        :param model:
        :return:
        """
        for i in range(1, len(model.faces) + 1):
            Renderer.draw_model_triangle_with_light(image, color, model, i)
            # print(f"Итерация {i}")

    @staticmethod
    def draw_model_triangle_with_light(image: np.ndarray, color: list[int], model: ObjModelClass.ObjModel,
                                       polygon_number: int) \
            -> None:
        point1, point2, point3 = model.get_points_by_index(polygon_number)
        point1 = point1.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        point2 = point2.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        point3 = point3.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)

        x_min = min(point1.x, point2.x, point3.x)
        y_min = min(point1.y, point2.y, point3.y)
        x_max = max(point1.x, point2.x, point3.x)
        y_max = max(point1.y, point2.y, point3.y)

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0

        if x_max > image.shape[1]:
            x_max = image.shape[1]
        if y_max > image.shape[0]:
            y_max = image.shape[0]

        light_cos = Renderer.calculate_cos_to_triangle(point1, point2, point3)
        color_with_light = [-item * light_cos for item in color]
        for y in np.arange(y_min - 1, y_max + 1):
            for x in np.arange(x_min - 1, x_max + 1):
                baricentrics = Renderer.determine_baricentric(x, point1.x, point2.x, point3.x, y, point1.y, point2.y,
                                                              point3.y)
                if all(lam >= 0 and light_cos < 0 for lam in baricentrics):
                    image[y, x] = color_with_light

    @staticmethod
    def draw_triangle_zbuffer(image: np.ndarray, color: list[int], model: ObjModelClass.ObjModel, polygon_number: int,
                              zbuffer: np.ndarray):
        point1, point2, point3 = model.get_points_by_index(polygon_number)
        light_cos = Renderer.calculate_cos_to_triangle(point1, point2, point3)
        if light_cos <= 0:
            return

        point1 = point1.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        point2 = point2.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)
        point3 = point3.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)

        # point1_scaled = point1.transform_to_int(model.scale, model.offset_x, model.offset_y, model.offset_z)

        x_min = (min(point1.x, point2.x, point3.x))
        y_min = (min(point1.y, point2.y, point3.y))
        x_max = (max(point1.x, point2.x, point3.x))
        y_max = (max(point1.y, point2.y, point3.y))

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0

        if x_max > image.shape[1]:
            x_max = image.shape[1]
        if y_max > image.shape[0]:
            y_max = image.shape[0]

        color_with_light = [item * light_cos for item in color]

        for y in np.arange(y_min - 1, y_max + 1):
            for x in np.arange(x_min - 1, x_max + 1):
                baricentrics = Renderer.determine_baricentric(x, point1.x, point2.x, point3.x, y, point1.y, point2.y,
                                                              point3.y)
                if all(lam >= 0 for lam in baricentrics):

                    # z = int(baricentrics[0] * int(point1.z) + baricentrics[1] * int(point2.z) + baricentrics[2] * int(point3.z))
                    z = int(baricentrics[0] * point1.z + baricentrics[1] * point2.z + baricentrics[2] * point3.z)

                    if z < zbuffer[y, x]:
                        image[y, x] = color_with_light
                        zbuffer[y, x] = z

    @staticmethod
    def draw_model_with_zbuffer(image: Image, color: list[int], model: ObjModelClass.ObjModel, zbuffer: np.ndarray):
        """
        Отрисовка модели циклом по полигонам
        :param image:
        :param model:
        :return:
        """
        for i in range(1, len(model.faces) + 1):
            Renderer.draw_triangle_zbuffer(image, color, model, i, zbuffer)
            # print(f"Итерация {i}")

    @staticmethod
    def calculate_new_point_position(model, point1, point2, point3, R):
        model.scale = 6

        point1 = point1.transform_to_int(model.scale)
        point2 = point2.transform_to_int(model.scale)
        point3 = point3.transform_to_int(model.scale)

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

        point1.x += model.offset_x
        point1.y += model.offset_y
        point1.z += model.offset_z

        point2.x += model.offset_x
        point2.y += model.offset_y
        point2.z += model.offset_z

        point3.x += model.offset_x
        point3.y += model.offset_y
        point3.z += model.offset_z

        return point1, point2, point3

    @staticmethod
    def draw_with_light(z_buffer, point1, point2, point3, image, color):
        light_direction_vector = np.array([0.0, 0.0, 1.0])
        light_cos = Renderer.calculate_cos_to_triangle(point1, point2, point3, light_direction_vector)
        if light_cos <= 0:
            return
        x_min = int(min(point1.x, point2.x, point3.x))
        y_min = int(min(point1.y, point2.y, point3.y))
        x_max = int(max(point1.x, point2.x, point3.x))
        y_max = int(max(point1.y, point2.y, point3.y))

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0

        if x_max > image.shape[1]:
            x_max = image.shape[1] - 1
        if y_max > image.shape[0]:
            y_max = image.shape[0] - 1

        color_with_light = [item * light_cos for item in color]
        for y in np.arange(y_min - 1, y_max + 1):
            for x in np.arange(x_min - 1, x_max + 1):
                baricentrics = Renderer.determine_baricentric(x, point1.x, point2.x, point3.x, y, point1.y, point2.y,
                                                              point3.y)
                if all(lam >= 0 for lam in baricentrics):
                    z = int(baricentrics[0] * point1.z + baricentrics[1] * point2.z + baricentrics[2] * point3.z)
                    if z < z_buffer[y, x]:
                        image[y, x] = color_with_light
                        z_buffer[y, x] = z

    @staticmethod
    def draw_with_rotation_by_index(image: Image, color: list[int], model: ObjModelClass.ObjModel, z_buffer: np.ndarray,
                                    R: np.array, polygon_number: int):
        point1, point2, point3 = model.get_points_by_index(polygon_number)
        point1, point2, point3 = Renderer.calculate_new_point_position(model, point1, point2, point3, R)
        Renderer.draw_with_light(z_buffer, point1, point2, point3, image, color)


    @staticmethod
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

    @staticmethod
    def draw_with_rotation(image: Image, color: list[int], model: ObjModelClass.ObjModel, z_buffer: np.ndarray,
                           rotate_x: int, rotate_y: int, rotate_z: int):

        R = Renderer.calculate_matrix_r(rotate_x, rotate_y, rotate_z)
        model.offset_y -=500
        for i in range(1, len(model.faces) + 1):
            Renderer.draw_with_rotation_by_index(image, color, model, z_buffer, R, i)

    @staticmethod
    def draw_triangle_projective_transformation(image: np.ndarray, color: list[int], model: ObjModelClass.ObjModel,
                                                polygon_number: int,
                                                zbuffer: np.ndarray):
        point1, point2, point3 = model.get_points_by_index(polygon_number)

        model.scale = 2000
        point1.z += model.offset_z
        point2.z += model.offset_z
        point3.z += model.offset_z

        point1.x, point1.y = (point1.x / point1.z), (point1.y / point1.z)
        point2.x, point2.y = (point2.x / point2.z), (point2.y / point2.z)
        point3.x, point3.y = (point3.x / point3.z), (point3.y / point3.z)

        point1.x, point1.y = point1.x * model.scale + model.offset_x, \
                             (point1.y * model.scale + model.offset_y // 2)
        point2.x, point2.y = (point2.x * model.scale + model.offset_x), (
                point2.y * model.scale + model.offset_y // 2)
        point3.x, point3.y = (point3.x * model.scale + model.offset_x), (
                point3.y * model.scale + model.offset_y // 2)
        Renderer.draw_with_light(zbuffer, point1, point2, point3, image, color)

    @staticmethod
    def draw_model_projective_transformation(image: Image, color: list[int], model: ObjModelClass.ObjModel,
                                             zbuffer: np.ndarray):
        """
        Отрисовка модели циклом по полигонам
        :param image:
        :param model:
        :return:
        """
        for i in range(1, len(model.faces) + 1):
            Renderer.draw_triangle_projective_transformation(image, color, model, i, zbuffer)
            # print(f"Итерация {i}")
