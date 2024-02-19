from RendererClass import Renderer
from ObjParserClass import ObjParser

class TaskManager:
    def __init__(self):
        pass

    def task1(self):
        renderer = Renderer()
        black_image = renderer.make_image_black()
        black_image.save('images/black-image.png')
        black_image.show('images/black-image.png')
        white_image = renderer.create_white_image()
        white_image.save('images/white-test.png')
        white_image.show('images/white-image.png')
        #to be continuted...

    def task2(self):
        pass

    def task3(self):
        obj_parser = ObjParser('obj-files/model_1.obj')
        vertices = obj_parser.parse_vertices()
        print(vertices)

    def task5(self):
        obj_parser = ObjParser('obj-files/model_1.obj')
        faces = obj_parser.parse_faces()
        print(faces)
