import numpy as np
from PIL import Image


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

