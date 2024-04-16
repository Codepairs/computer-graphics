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
        self.textures = self.parse_texture_coords()
        self.texture_numbers = self.parse_texture_numbers()

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


    def determine_z_offset(self,  resolution: tuple, scale_modificator=2):
        offset_z = 0
        lowest_z = self.zmin

        if (lowest_z >= 0):
            return
        lowest_z = abs(self.zmax - self.zmin) * 3
        self.offset_z = lowest_z




    def scale_coordinate_to_z(self,  resolution: tuple, scale_modificator=2):
        limit = min(resolution)
        scale = 1
        figure_y_area = abs(self.ymax - self.ymin)
        figure_x_area = abs(self.xmax - self.xmin)
        figure_z_area = abs(self.zmax - self.zmin) + self.offset_z

        while ( (max(figure_y_area, figure_x_area)/figure_z_area) * scale * scale_modificator**2 < limit ):
            scale *= scale_modificator
        self.scale = scale





    def offset_coordinates(self, resolution: tuple):
        self.offset_x = resolution[0]//2
        self.offset_y = resolution[1] // 2
        self.offset_z = resolution[1] // 2
        '''
        self.offset_coordinate_y(resolution)
        self.offset_coordinate_z(resolution)
        '''

    def offset_coordinate_z(self, resolution: tuple):
        limit = resolution[1]
        self.offset_z = limit//2


    def parse_faces(self):
        """
        Парсинг граней obj-файла.
        :return: np.array
        """
        with open(self.obj_filepath, 'r') as f:
            lines = f.readlines()
            faces = []
            has_slashes = False
            for line in lines:
                if '/' in line:
                    has_slashes = True
                    break

            for line in lines:
                if line.startswith('f '):
                    if has_slashes:
                        face = [int(val.split('/')[0]) for val in line.split()[1:]]
                    else:
                        face = [int(val) for val in line.split()[1:]]
                    if len(face)>3:
                        faces.append([face[0], face[2], face[3]])
                    faces.append([face[0], face[1], face[2]])


            #Вот тут падает из-за разной размерности полигонов
            #return np.array(faces)
            return faces


    def parse_texture_coords(self):
        with open(self.obj_filepath, 'r') as f:
            lines = np.array(f.readlines())
            vertices_texture = []
            for line in lines:
                if line.startswith('vt '):
                    vertex_texture = [float(val) for val in line.split()[1:]]
                    vertices_texture.append(vertex_texture)
            return np.array(vertices_texture)

    def parse_texture_numbers(self):
        with open(self.obj_filepath, 'r') as f:
            lines = f.readlines()
            textures = []
            for line in lines:
                if line.startswith('f '):
                    #print(line)
                    texture = []
                    for val in line.split()[1:]:
                        #print(val)
                        texture.append(int(val.split('/')[1]))
                    if len(texture) > 3:
                        textures.append([texture[0], texture[2], texture[3]])
                    textures.append([texture[0], texture[1], texture[2]])

            #Вот тут падает из-за разной размерности полигонов
            #return np.array(faces)
            return textures



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

    def get_texture_by_index(self, vertices_index: int):
        """
        Возвращает координаты вершин грани.
        ВНИМАНИЕ! Индексация номеров вершин начинается с единицы.
        :param vertices_index:
        :return: np.array
        """
        # Добавляем везде -1, тк мы считаем индексы с единицы.
        if vertices_index <= 0:
            raise ValueError('Неверный индекс вершины! Индексация номеров вершин начинается с единицы!')
        points_indexes = self.texture_numbers[vertices_index - 1]
        raw_point1 = self.textures[points_indexes[0] - 1]
        raw_point2 = self.textures[points_indexes[1] - 1]
        raw_point3 = self.textures[points_indexes[2] - 1]

        array = np.array([raw_point1, raw_point2, raw_point3])
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
