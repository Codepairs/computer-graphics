import numpy as np
import math


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
