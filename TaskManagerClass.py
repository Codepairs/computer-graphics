import colors
import RendererClass as Renderer
import Renderer18 as Renderer18
from PIL import Image, ImageOps
from ObjModelClass import ObjModel
import numpy as np
import math

path_to_obj_file = 'obj-files/model_2.obj'


class TaskManager:
    @staticmethod
    def choose_model(model_num, resolution: tuple):
        path = 'obj-files/model_{}.obj'.format(model_num)
        model_to_render = ObjModel(obj_file=path)
        model_to_render.fill_coordinates_info()
        model_to_render.offset_coordinates(resolution)
        model_to_render.scale_coordinates(resolution)

        return model_to_render

    @staticmethod
    def choose_model_new(model_num, resolution: tuple):
        path = 'obj-files/model_{}.obj'.format(model_num)
        model_to_render = ObjModel(obj_file=path)
        model_to_render.fill_coordinates_info()
        model_to_render.offset_coordinates(resolution)
        model_to_render.determine_z_offset(resolution)
        model_to_render.scale_coordinate_to_z(resolution)
        return model_to_render

    @staticmethod
    def task14(matrix_size: tuple, color: list[int], model_num: int):
        model = TaskManager.choose_model(model_num, matrix_size)
        image = np.zeros(matrix_size + (3,), dtype=np.uint8)

        z_buffer = np.full(matrix_size, 100000, dtype=np.uint32)
        Renderer.draw_model_with_zbuffer(image, color, model, z_buffer)

        file_image = Image.fromarray(image, "RGB")
        file_image = ImageOps.flip(file_image)
        file_image.show()

    @staticmethod
    def task15(matrix_size:tuple, color: list[int], model_num: int, rotate_x: int, rotate_y: int, rotate_z:int):
        model = TaskManager.choose_model_new(model_num, matrix_size)
        image = np.zeros(matrix_size + (3,), dtype=np.uint8)
        z_buffer = np.full(matrix_size, 100000., dtype=np.float32)
        Renderer18.draw_with_rotation(image, color, model, z_buffer, rotate_x, rotate_y, rotate_z)
        file_image = Image.fromarray(image, "RGB")
        file_image = ImageOps.flip(file_image)
        file_image.show()

    @staticmethod
    def task16(matrix_size: tuple, color: list[int], model_num: int):
        model = TaskManager.choose_model_new(model_num, matrix_size)
        image = np.zeros(matrix_size + (3,), dtype=np.uint8)

        z_buffer = np.full(matrix_size, 100000., dtype=np.float32)
        Renderer.draw_model_projective_transformation(image, color, model, z_buffer)

        file_image = Image.fromarray(image, "RGB")
        file_image = ImageOps.flip(file_image)
        file_image.show()

    @staticmethod
    def task17(matrix_size: tuple, color: list[int], model_num: int):
        model = TaskManager.choose_model_new(model_num, matrix_size)
        image = np.zeros(matrix_size + (3,), dtype=np.uint8)

        z_buffer = np.full(matrix_size, np.Infinity, dtype=np.float32)
        Renderer.draw_model_guro_shading(image, color, model, z_buffer, 0,90,0)

        file_image = Image.fromarray(image, "RGB")
        file_image = ImageOps.flip(file_image)
        file_image.show()


    @staticmethod
    def task18(matrix_size: tuple, color: list[int], model_num: int, rotate_x, rotate_y, rotate_z):
        model = TaskManager.choose_model_new(model_num, matrix_size)
        image = np.zeros(matrix_size + (3,), dtype=np.uint8)

        z_buffer = np.full(matrix_size, 100000., dtype=np.float64)

        texture = Image.open(f'./textures/model_{model_num}.jpg')
        texture = ImageOps.flip(texture)
        #texture.show()
        texture_np = np.array(texture)

        Renderer.draw_model_texture(image, color, model, z_buffer, texture_np, rotate_x, rotate_y, rotate_z)

        file_image = Image.fromarray(image, "RGB")
        file_image = ImageOps.flip(file_image)
        file_image.show()

