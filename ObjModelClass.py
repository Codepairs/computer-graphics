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
        self.offset_x = offset
        self.offset_y = offset
        self.offset_z = offset
        self.faces = self.parse_faces()
        self.vertices = self.parse_vertices()

        self.ymax = 0
        self.ymin = 0
        self.ymean = 0

        self.xmax = 0
        self.xmin = 0
        self.xmean = 0

        self.zmax = 0
        self.zmin = 0
        self.zmean = 0

    def parse_vertices(self):
        """
        Парсинг вершин obj-файла.
        :return: np.array
        """
        with open(self.obj_filepath, 'r') as f:
            lines = np.array(f.readlines())
            vertices = []
            for line in lines:
                if line.startswith('v '):
                    vertex = [float(val) for val in line.split()[1:]]
                    vertices.append(vertex)
            return np.array(vertices)

    def fill_coordinates_info(self) -> np.array:
        '''
        :return: np.array: min, max, mean values
        '''
        vertices = self.parse_vertices()
        vertices_x = vertices[:, 0]
        vertices_y = vertices[:, 1]
        vertices_z = vertices[:, 2]
        self.ymin = np.min(vertices_y)
        self.ymax = np.max(vertices_y)
        self.ymean = np.mean(vertices_y)

        self.xmin = np.min(vertices_x)
        self.xmax = np.max(vertices_x)
        self.xmean = np.mean(vertices_x)

        self.zmin = np.min(vertices_z)
        self.zmax = np.max(vertices_z)
        self.zmean = np.mean(vertices_z)


        #return np.array([ymin, ymax, ymean])


    def scale_coordinates(self, resolution: tuple, scale_modificator=2):
        '''

        :param resolution: sizes int
        :return: void
        '''
        limit = min(resolution)
        scale = 1
        model_max = abs(max(abs(self.ymax), abs(self.xmax)))
        if model_max > limit:
            scale_modificator = 1/scale_modificator
            scale = scale_modificator
            while model_max * (scale * (scale_modificator ** 2)) > limit:
                scale *= scale_modificator
            if model_max * scale * scale_modificator > limit:
                scale *= scale_modificator
        else:
            while model_max*(scale * (scale_modificator**2)) < limit:
                scale*=scale_modificator
            if model_max*scale*scale_modificator < limit:
                scale*=scale_modificator
        self.scale = scale
        #return scale


    def offset_coordinates(self, resolution: tuple):
        self.offset_coordinate_x(resolution)
        self.offset_coordinate_y(resolution)
        self.offset_coordinate_z(resolution)

    def offset_coordinate_x(self, resolution: tuple):
        limit = resolution[1]
        offset = 0
        while abs(self.xmax) * self.scale + offset + limit // 10 < limit:
            offset += limit // 10

        self.offset_x = offset

    def offset_coordinate_y(self, resolution: tuple):
        limit = resolution[1]
        offset = 0
        while abs(self.ymax) * self.scale + offset + limit // 10 < limit:
            offset += limit // 10

        self.offset_y = offset

    def offset_coordinate_z(self, resolution: tuple):
        limit = resolution[1]
        offset = 0
        while abs(self.zmax) * self.scale + offset + limit // 10 < limit:
            offset += limit // 10

        self.offset_z = offset


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
