class Point3D:
    """
    Класс, описывающий точку в трех координатах
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def transform_to_int(self, scale, offset_x, offset_y, offset_z):
        """
        Масштабирование точки и добавление смещения
        :param scale:
        :param offset:
        :return:
        """
        x = int(self.x * scale)
        y = int(self.y * scale)
        z = int(self.z * scale)
        x += offset_x
        y += offset_y
        z += offset_z
        return Point3D(x, y, z)

    def scale_point_to_int(self, scale):
        """
        Масштабирование точки и добавление смещения
        :param scale:
        :param offset:
        :return:
        """
        x = int(self.x * scale)
        y = int(self.y * scale)
        z = int(self.z * scale)
        return Point3D(x, y, z)

    def transform_to_float(self, scale, offset_x, offset_y, offset_z):
        x = (self.x * scale)
        y = (self.y * scale)
        z = (self.z * scale)
        x += offset_x
        y += offset_y
        z += offset_z
        return Point3D(x, y, z)

    def __str__(self):
        return f'[X:{self.x}, Y:{self.y}, Z:{self.z}]'
