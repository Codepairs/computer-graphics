import random

DARK_PURPLE = [83, 55, 122]
INDIAN_RED = [205, 92, 92]
PINK = [255, 192, 203]
GOLD = [255, 215, 0]
SANDY_BROWN = [244, 164, 96]
SILVER = [192, 192, 192]
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
LIME = [0, 255, 0]
CYAN = [0, 255, 255]
LIGHT_SKY_BLUE = [135, 206, 250]


def get_random_color():
    return [random.randint(0, 255) for _ in range(3)]
