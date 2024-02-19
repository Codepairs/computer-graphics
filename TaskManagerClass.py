from RendererClass import Renderer
from ObjParserClass import ObjParser


class TaskManager:
    @staticmethod
    def task1():
        renderer = Renderer()
        black_image = renderer.create_black_image()
        black_image.save('images/black-image.png')
        black_image.show('images/black-image.png')
        white_image = renderer.create_white_image()
        white_image.save('images/white-test.png')
        white_image.show('images/white-image.png')
        # to be continuted...

    @staticmethod
    def task2():
        pass

    @staticmethod
    def task3():
        obj_parser = ObjParser('obj-files/model_1.obj')
        vertices = obj_parser.parse_vertices()
        print(vertices)

    @staticmethod
    def task5():
        obj_parser = ObjParser('obj-files/model_1.obj')
        faces = obj_parser.parse_faces()
        print(faces)
