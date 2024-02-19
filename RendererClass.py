import numpy as np
from PIL import Image
import math

class Renderer:
    def __init__(self):
        self._image_size = self._h, self._w = 600, 800
        self._image_matrix = np.zeros(self._image_size, dtype=np.uint8)

    def update_point(self, pos_y, pos_x, color):
        """
        Обновляет цвет точки по позиции
        :param pos_y: Позиция Y
        :param pos_x: Позиция X
        :param color: Цвет
        :return: None
        """
        self._image_matrix[pos_y][pos_x] = color

    def create_black_image(self):
        """
        Создает полностью черное изображение
        :return: PIL.image
        """
        self._image_matrix = np.zeros(self._image_size, dtype=np.uint8)
        return Image.fromarray(self._image_matrix)

    def create_white_image(self):
        """
        Создает полностью белое изображение
        :return: PIL.image
        """
        self._image_matrix.fill(255)
        return Image.fromarray(self._image_matrix)

    @staticmethod
    def algorithm_dotted_line(image, x0, y0, x1, y1, count, color):
        """
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
    def algorithm_dotted_line_sqrt(image, x0, y0, x1, y1, color):
        """
        недостатки:
            корень вычисляется 5 лет
        """
        count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        step = 1.0 / count
        for t in np.arange(0, 1, step):
            x = round((1.0 - t) * x0 + t * x1)
            y = round((1.0 - t) * y0 + t * y1)
            image[y, x] = color

    @staticmethod
    def algorithm_x_loop_line(image, x0, y0, x1, y1, color):
        """
        недостатки:
            если начало правее конца, то не рисует (без левой половины)
            идем по x -> шаг по x<чем по у -> рядом с Оу получим пунктир
        """
        for x in range(x0, x1):
            t = (x - x0) / (x1 - x0)
            y = int(y1 * t + y0 * (1.0 - t))
            image[y, x] = color

    @staticmethod
    def algorithm_x_loop_line_fixed(image, x0, y0, x1, y1, color):
        """
        недостатки:
            отсутствуют в явном виде
        """
        xchange = False
        if (abs(x0 - x1) < abs(y0 - y1)):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            xchange = True

        if (x0 > x1):
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
    def algorithm_dy(image, x0, y0, x1, y1, color):
        """
        derror - погрешность вычислений, накапливаем ее, чтобы перейти вверх по y,
         затем сбрасываем
        """
        y = y0
        dy = abs((y1 - y0) / (x1 - x0))
        derror = 0.0
        y_update = 1 if y1 > y0 else -1

        for x in range(x0, x1):
            derror += dy
            if (derror > 0.5):
                derror -= 1.0
                y += y_update
            image[y, x] = color

    @staticmethod
    def algorithm_bresenham(image, x0, y0, x1, y1, color):
        y = y0
        dy = 2 * abs(y1 - y0)
        derror = 0.0
        y_update = 1 if y1 > y0 else -1

        for x in range(x0, x1):
            derror += dy
            if (derror > 0.5):
                derror -= 2 * (x1 - x0)
                y += y_update
            image[y, x] = color