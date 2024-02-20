from Point2D import Point2D
from Point3D import Point3D
import numpy as np


class ObjModel:
    """
    Класс obj модели
    """

    def __init__(self, obj_file: str, scale: float = 1, offset=0):
        self.obj_filepath = obj_file
        self.scale = scale
        self.offset = offset
        self.faces = self.parse_faces()
        self.vertices = self.parse_vertices()

    def parse_vertices(self):
        """
        Парсинг вершин obj-файла.
        :return: np.array
        """
        with open(self.obj_filepath, 'r') as f:
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
        with open(self.obj_filepath, 'r') as f:
            lines = f.readlines()
            faces = []
            for line in lines:
                if line.startswith('f '):
                    face = [int(val.split('/')[0]) for val in line.split()[1:]]
                    faces.append(face)

            #Вот тут падает из-за разной размерности полигонов
            #return np.array(faces)
            return faces

    def get_points_by_index(self, vertices_index: int):
        """
        Возвращает координаты вершин грани.
        ВНИМАНИЕ! Индексация номеров вершин начинается с единицы.
        :param vertices_index:
        :return: np.array
        """
        # Добавляем везде -1, тк мы считаем индексы с единицы.
        if vertices_index <= 0:
            raise ValueError('Неверный индекс вершины! Индексация номеров вершин начинается с единицы!')
        points_indexes = self.faces[vertices_index - 1]
        raw_point1 = self.vertices[points_indexes[0] - 1]
        raw_point2 = self.vertices[points_indexes[1] - 1]
        raw_point3 = self.vertices[points_indexes[2] - 1]
        point1 = Point3D(raw_point1[0], raw_point1[1], raw_point1[2])
        point2 = Point3D(raw_point2[0], raw_point2[1], raw_point2[2])
        point3 = Point3D(raw_point3[0], raw_point3[1], raw_point3[2])
        array = np.array([point1, point2, point3])
        return array

    def get_scale(self):
        return self.scale

    def set_scale(self, scale):
        self.scale = scale

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.offset = offset

    def get_path_to_obj_file(self):
        return self.obj_filepath

    def set_path_to_obj_file(self, obj_file):
        self.obj_filepath = obj_file


model = ObjModel(obj_file='obj-files/model_1.obj', scale=3000, offset=500)
point1, point2, point3 = model.get_points_by_index(1)