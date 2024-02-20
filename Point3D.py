class Point3D:
    """
    Класс, описывающий точку в трех координатах
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def transform(self, scale, offset):
        """
        Масштабирование точки и добавление смещения
        :param scale:
        :param offset:
        :return:
        """
        x = int(self.x * scale)
        y = int(self.y * scale)
        z = int(self.z * scale)
        x += offset
        y += offset
        z += offset
        return Point3D(x, y, z)

    def __str__(self):
        return f'[X:{self.x}, Y:{self.y}, Z:{self.z}]'
