import numpy
import numpy as np


class ObjManager:
    """
    Класс для парсинга obj-файла.
    """
    def __init__(self, obj_file):
        self.obj_file = obj_file

    def parse_vertices(self):
        """
        Парсинг вершин obj-файла.
        :return: np.array
        """
        with open(self.obj_file, 'r') as f:
            lines = f.readlines()
            vertices = []
            for line in lines:
                if line.startswith('v '):
                    vertex = [float(val) for val in line.split()[1:]]
                    vertices.append(vertex)
            return np.array(vertices)

    def parse_faces(self):
        """
        Парсинг граней obj-файла.
        :return: np.array
        """
        with open(self.obj_file, 'r') as f:
            lines = f.readlines()
            faces = []
            for line in lines:
                if line.startswith('f '):
                    face = [int(val.split('/')[0]) for val in line.split()[1:]]
                    faces.append(face)
            return np.array(faces)

    @staticmethod
    def get_points_from_face(vertices_index: int, faces: np.array, vertices: np.array):
        """
        Возвращает координаты вершин грани.
        ВНИМАНИЕ! Индексация номеров вершин начинается с единицы.
        :param vertices_index:
        :param faces:
        :param vertices:
        :return: np.array
        """
        # Добавляем везде -1, тк мы считаем индексы с единицы.
        if vertices_index <= 0:
            raise ValueError('Неверный индекс вершины! Индексация номеров вершин начинается с единицы!')
        return np.array(vertices[faces[vertices_index-1]-1])

    @staticmethod
    def print_points(point1: np.array, point2: np.array, point3: np.array):
        """
        Выводит в консоль координаты точек
        :param point1:
        :param point2:
        :param point3:
        :return: None
        """
        print(f'X1: {point1[0]}, Y1: {point1[1]}, Z1: {point1[2]}')
        print(f'X2: {point2[0]}, Y2: {point2[1]}, Z2: {point2[2]}')
        print(f'X3: {point3[0]}, Y3: {point3[1]}, Z3: {point3[2]}')
        print()
