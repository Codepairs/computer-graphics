from TaskManagerClass import TaskManager
import colors


def main():
    # TaskManager.task1((256, 256))
    # TaskManager.task2((256, 256), 255)
    # TaskManager.task3()
    # TaskManager.task4((1000, 1000), 255, 2)
    # TaskManager.task5()
    TaskManager.task6((1000, 1000), color=colors.CYAN, model_num=4)


if __name__ == '__main__':
    main()
