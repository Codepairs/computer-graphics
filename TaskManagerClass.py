import time

from RendererClass import Renderer
from ObjManagerClass import ObjManager
from PIL import Image
import numpy as np
import math

path_to_obj_file = 'obj-files/model_1.obj'


class TaskManager:
    @staticmethod
    def task1(matrix_size: tuple):
        matrix_size_3d = matrix_size + (3,)
        renderer = Renderer()
        black_image = np.ndarray(matrix_size, dtype=np.uint8)
        white_image = np.ndarray(matrix_size, dtype=np.uint8)
        volume_image = np.full(shape=matrix_size_3d, fill_value=[256, 0, 0], dtype=np.uint8)
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

    @staticmethod
    def task2(matrix_size: tuple, color: int):
        single_image = np.ndarray(matrix_size, dtype=np.uint8)
        images = np.tile(single_image, (6, 1, 1))
        renderer = Renderer()
        total_lines = 10
        for line in range(total_lines):
            x0, y0 = matrix_size[1] // 2, matrix_size[0] // 2
            x1, y1 = int(x0 + (x0 - 20) * math.cos((2 * math.pi * line) / total_lines)), int(
                y0 + (y0 - 20) * math.sin((2 * math.pi * line) / total_lines))
            renderer.algorithm_dotted_line(images[0], x0, y0, x1, y1, 20, color)
            renderer.algorithm_dotted_line_sqrt(images[1], x0, y0, x1, y1, color)
            renderer.algorithm_x_loop_line(images[2], x0, y0, x1, y1, color)
            renderer.algorithm_x_loop_line_fixed(images[3], x0, y0, x1, y1, color)
            renderer.algorithm_dy(images[4], x0, y0, x1, y1, color)
            renderer.algorithm_bresenham(images[5], x0, y0, x1, y1, color)

        result_images = [Image.fromarray(x) for x in images]
        for image in result_images:
            #time.sleep(1)
            image.show()

    @staticmethod
    def task3():
        obj_parser = ObjManager(path_to_obj_file)
        vertices = obj_parser.parse_vertices()
        print(vertices)

    @staticmethod
    def task5():
        obj_parser = ObjManager(path_to_obj_file)
        faces = obj_parser.parse_faces()
        print(faces)

    @staticmethod
    def task6(matrix_size: tuple):
        image = np.zeros(matrix_size, dtype=np.uint8)
        render = Renderer()
        obj_manager = ObjManager(path_to_obj_file)
        vertices = obj_manager.parse_vertices()
        faces = obj_manager.parse_faces()
        for i in range(1, len(faces) + 1):
            point1, point2, point3 = obj_manager.get_points_from_face(i, faces, vertices)
            print(i)
            x1, y1, z1 = point1[0], point1[1], point1[2]
            x2, y2, z2 = point2[0], point2[1], point2[2]
            x3, y3, z3 = point3[0], point3[1], point3[2]
            print('Точки для соединения:')
            obj_manager.print_points(point1, point2, point3)

            '''
            new_x1, new_y1 = -int(x1 * 5 + 500), int(x1 * 5 + 500)
            new_x2, new_y2 = -int(z2 * 5 + 500), int(x2 * 5 + 500)
            new_x3, new_y3 = -int(z3 * 5 + 500), int(x3 * 5 + 500)
            print(new_x1, new_y1)
            render.algorithm_bresenham(image, new_x1, new_y1, new_x2, new_y2, 255)
            render.algorithm_bresenham(image, new_x2, new_y2, new_x3, new_y3, 255)
            render.algorithm_bresenham(image, new_x3, new_y3, new_x1, new_y1, 255)
        file_image = Image.fromarray(image, "L")
        file_image.show()
        '''
