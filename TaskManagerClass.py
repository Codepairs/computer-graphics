import colors
from RendererClass import Renderer
from PIL import Image, ImageOps
from ObjModelClass import ObjModel
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
        volume_image = np.full(shape=matrix_size_3d, fill_value=[255, 0, 0], dtype=np.uint8)
        gradient_image = np.ndarray(shape=matrix_size_3d, dtype=np.uint8)

        renderer.make_image_colored(black_image, 0)

        pil_black = Image.fromarray(black_image, mode="L")
        pil_black.save('images/black-image.png')
        pil_black.show()

        renderer.make_image_colored(white_image, 255)

        pil_white = Image.fromarray(white_image, mode="L")
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
        single_image = np.zeros(shape=matrix_size, dtype=np.uint8)
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

        result_images = [Image.fromarray(x, mode="L") for x in images]
        for image in result_images:
            image.show()

    @staticmethod
    def task3():
        model = ObjModel(path_to_obj_file)
        vertices = model.parse_vertices()
        print(vertices)



    '''
    @staticmethod
    def choose_model(model_num):
        if model_num == 1:
            path = 'obj-files/model_1.obj'
            scale = 3000
            offset = 500
            model_to_render = ObjModel(path, scale, offset)
            return model_to_render

        elif model_num == 2:
            path = 'obj-files/model_2.obj'
            scale = 0.4
            offset = 400
            model_to_render = ObjModel(path, scale, offset)
            return model_to_render

        elif model_num == 3:
            path = 'obj-files/model_3.obj'
            scale = 700
            offset = 300
            model_to_render = ObjModel(path, scale, offset)
            return model_to_render
        elif model_num == 4:
            path = 'obj-files/model_4.obj'
            scale = 4
            offset = 400
            model_to_render = ObjModel(path, scale, offset)
            return model_to_render
    '''

    @staticmethod
    def choose_model(model_num, resolution: tuple):
        path = 'obj-files/model_{}.obj'.format(model_num)
        model_to_render = ObjModel(obj_file=path)
        model_to_render.fill_coordinates_info()
        model_to_render.scale_coordinates(resolution)
        model_to_render.offset_coordinates(resolution)
        return model_to_render


    @staticmethod
    def task4(matrix_size: tuple, color, model_num):

        model = TaskManager.choose_model(model_num, matrix_size)
        renderer = Renderer()
        image = np.ndarray(matrix_size)

        for (x, y, z) in model.vertices:
            new_x = int(model.get_scale() * x + model.get_offset())  # 2 mod
            new_y = int(model.get_scale() * y + model.get_offset())
            renderer.update_point(image, pos_x=new_x, pos_y=new_y, color=color)

        pil_image = Image.fromarray(image)
        pil_image = ImageOps.flip(pil_image)
        pil_image.show()

    @staticmethod
    def task5():
        model = ObjModel(path_to_obj_file)
        faces = model.parse_faces()
        print(faces)

    @staticmethod
    def task6(matrix_size: tuple, color: list[int], model_num: int):
        model = TaskManager.choose_model(model_num, matrix_size)
        image = np.zeros(matrix_size + (3,), dtype=np.uint8)
        Renderer.draw_model_with_faces(image, model, color)
        file_image = Image.fromarray(image, "RGB")
        file_image = ImageOps.flip(file_image)
        file_image.show()

    @staticmethod
    def task7():
        print("Оно работает верь мне, функции есть в RendererClass.py")


    @staticmethod
    def task8(matrix_size: tuple, color, x0, x1, x2, y0, y1, y2):
        image = np.zeros(matrix_size + (3,), dtype=np.uint8)
        Renderer.draw_triangle(image=image, color=color, x0=x0, x1=x1, x2=x2, y0=y0, y1=y1, y2=y2)
        file_image = Image.fromarray(image, "RGB")
        #file_image = ImageOps.flip(file_image)
        file_image.show()

    @staticmethod
    def task9(matrix_size: tuple, color):
        # completely into bounds
        image1 = np.zeros(matrix_size + (3,), dtype=np.uint8)
        Renderer.draw_triangle(image=image1, color=color, x0=0, x1=200, x2=800, y0=1, y1=500, y2=900)
        Renderer.draw_triangle(image=image1, color=colors.SILVER, x0=100, x1=300, x2=900, y0=10, y1=600, y2=1000)
        file_image1 = Image.fromarray(image1, "RGB")
        file_image1.show()

        # partly out of bounds
        image2 = np.zeros(matrix_size + (3,), dtype=np.uint8)
        Renderer.draw_triangle(image=image2, color=color, x0=-100, x1=200, x2=800, y0=1, y1=5000, y2=900)
        file_image2 = Image.fromarray(image2, "RGB")
        file_image2.show()

    @staticmethod
    def task10(matrix_size: tuple, model_num: int):
        model = TaskManager.choose_model(model_num, matrix_size)
        image = np.zeros(matrix_size + (3,), dtype=np.uint8)
        Renderer.draw_model_with_random_color_polygons(image, model)
        file_image = Image.fromarray(image, "RGB")
        file_image = ImageOps.flip(file_image)
        file_image.show()

    @staticmethod
    def task11():
        print("Оно работает верь мне, функция calculate_normal_to_triangle()")

    @staticmethod
    def task12(matrix_size: tuple, model_num: int):
        model = TaskManager.choose_model(model_num, matrix_size)
        image = np.zeros(matrix_size + (3,), dtype=np.uint8)
        Renderer.draw_model_with_random_color_and_cos(image, model)
        file_image = Image.fromarray(image, "RGB")
        file_image = ImageOps.flip(file_image)
        file_image.show()

    @staticmethod
    def task13(matrix_size: tuple, color: list[int], model_num: int):
        model = TaskManager.choose_model(model_num, matrix_size)
        image = np.zeros(matrix_size + (3,), dtype=np.uint8)
        Renderer.draw_model_with_light(image, color, model)
        file_image = Image.fromarray(image, "RGB")
        file_image = ImageOps.flip(file_image)
        file_image.show()


