class ObjParser:
    def __init__(self, obj_file):
        self.obj_file = obj_file

    def parse(self):
        """
        Парсинг obj-файла.
        На выходе получаем список вершин и список граней

        :return:
        """
        with open(self.obj_file, 'r') as f:
            lines = f.readlines()
            vertices = []
            faces = []
            for line in lines:
                if line.startswith('v '):
                    vertex = [float(val) for val in line.split()[1:]]
                    vertices.append(vertex)
                elif line.startswith('f '):
                    face = [int(val.split('/')[0]) for val in line.split()[1:]]
                    faces.append(face)
            return vertices, faces
