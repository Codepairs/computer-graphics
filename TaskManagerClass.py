from RendererClass import Renderer
from ObjParserClass import ObjParser
from PIL import Image
import numpy as np

class TaskManager:
    @staticmethod
    def task1(matrix_size: tuple):
        matrix_size_3d = matrix_size + (3,)
        renderer = Renderer()
        black_image = np.ndarray(matrix_size, dtype=np.uint8)
        white_image = np.ndarray(matrix_size, dtype=np.uint8)
        volume_image = np.full(shape=(matrix_size_3d), fill_value=[256,0,0], dtype=np.uint8)
        gradient_image = np.ndarray(shape=matrix_size_3d, dtype=np.uint8)


        renderer.make_image_colored(black_image, 0)

        pil_black = Image.fromarray(black_image)
        pil_black.save('images/black-image.png')
        pil_black.show()

        renderer.make_image_colored(white_image, 255)

        pil_white = Image.fromarray(white_image)
        pil_white.save('images/white-test.png')
        pil_white.show()


        pil_volume = Image.fromarray(volume_image)
        pil_volume.save('images/volume-test.png')
        pil_volume.show()

        renderer.make_image_gradient(gradient_image)

        pil_gradient = Image.fromarray(gradient_image)
        pil_gradient.save('images/gradient-test.png')
        pil_gradient.show()
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
