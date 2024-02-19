import numpy
import numpy as np


class ObjParser:
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
