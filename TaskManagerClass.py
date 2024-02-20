import time

from RendererClass import Renderer
from ObjManagerClass import ObjManager
from PIL import Image, ImageOps
import numpy as np
import math

path_to_obj_file = 'obj-files/model_2.obj'


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

        pil_volume = Image.fromarray(volume_image, mode="RGB")
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

        result_images = [Image.fromarray(x, mode='L') for x in images]
        for image in result_images:
            image.show()

    @staticmethod
    def task3():
        obj_parser = ObjManager(path_to_obj_file)
        vertices = obj_parser.parse_vertices()
        print(vertices)

    @staticmethod
    def task4(matrix_size: tuple, color, model_num):
        path = ""
        const_mods = [1, 1]
        if model_num==1:
            path = 'obj-files/model_1.obj'
            const_mods = [3000, 500]

        elif model_num==2:
            path = 'obj-files/model_2.obj'
            const_mods = [0.4, 400]

        obj_parser = ObjManager(path)
        renderer = Renderer()

        vertices = obj_parser.parse_vertices()
        image = np.ndarray(matrix_size)

        for (x,y,z) in vertices:
            new_x = int(const_mods[0]*x+const_mods[1]) #2 mod
            new_y = int(const_mods[0]*y+const_mods[1])
            renderer.update_point(image, pos_x=new_x,pos_y=new_y,color=color)

        pil_image = Image.fromarray(image, mode='L')
        pil_image = ImageOps.flip(pil_image)
        pil_image.show()


    @staticmethod
    def task5():
        obj_parser = ObjManager(path_to_obj_file)
        faces = obj_parser.parse_faces()
        print(faces)

    @staticmethod
    def task6(matrix_size: tuple, color:int, model_num: int):
        path = ""
        const_mods = [1, 1]
        if model_num == 1:
            path = 'obj-files/model_1.obj'
            const_mods = [3000, 500]


        elif model_num == 2:
            path = 'obj-files/model_2.obj'
            const_mods = [0.4, 400]


        image = np.zeros(matrix_size, dtype=np.uint8)
        render = Renderer()
        obj_manager = ObjManager(path)
        vertices = obj_manager.parse_vertices()
        faces = obj_manager.parse_faces()


        for i in range(1, len(faces) + 1):
            point1, point2, point3 = obj_manager.get_points_from_face(i, faces, vertices)
            x1, y1, z1 = point1[0], point1[1], point1[2]
            x2, y2, z2 = point2[0], point2[1], point2[2]
            x3, y3, z3 = point3[0], point3[1], point3[2]

            scaled_x1 = int(const_mods[0] * x1 + const_mods[1])
            scaled_y1 = int(const_mods[0] * y1 + const_mods[1])
            scaled_x2 = int(const_mods[0] * x2 + const_mods[1])
            scaled_y2 = int(const_mods[0] * y2 + const_mods[1])
            scaled_x3 = int(const_mods[0] * x3 + const_mods[1])
            scaled_y3 = int(const_mods[0] * y3 + const_mods[1])

            render.algorithm_bresenham(image, scaled_x1, scaled_y1, scaled_x2, scaled_y2, 255)
            render.algorithm_bresenham(image, scaled_x2, scaled_y2, scaled_x3, scaled_y3, 255)
            render.algorithm_bresenham(image, scaled_x3, scaled_y3, scaled_x1, scaled_y1, 255)

        file_image = Image.fromarray(image, "L")
        file_image = ImageOps.flip(file_image)
        file_image.show()

