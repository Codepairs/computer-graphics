from RendererClass import Renderer


def main():
    renderer = Renderer()
    #Task1
    black_image = renderer.create_black_image()
    black_image.save('images/black-image.png')
    black_image.show('images/black-image.png')
    #Task2
    white_image = renderer.create_white_image()
    white_image.save('images/white-test.png')
    white_image.show('images/white-image.png')



if __name__ == '__main__':
    main()
