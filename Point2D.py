class Point2D:
    """
    Класс, описывающий точку в двух координатах
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def transform(self, scale, offset):
        """
        Масштабирование точки и добавление смещения
        :param scale:
        :param offset:
        :return:
        """
        x = self.x * scale
        y = self.y * scale
        x += offset
        y += offset
        return Point2D(x, y)

    def __str__(self):
        return f'[X:{self.x}, Y:{self.y}]'
