import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_matrix(m):
    plt.matshow(m, cmap=plt.cm.Blues)
    plt.show()


def plot_Matrix(confusion, type=0):

    image_type_list = ['0 - NotHate', '1 - Racist', '2 - Sexist', '3 - Homophobe', '4 - Religion', '5 - OtherHate']

    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.colorbar()

    indices = range(len(confusion))
    classes = [image_type_list[type], 'Else']
    plt.xticks(indices, classes)  # 设置横坐标方向，rotation=45为45度倾斜
    plt.yticks(indices, classes, rotation='90')

    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.title('Confusion matrix')

    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    normalize = True
    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.

    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(second_index, first_index, format(confusion[first_index][second_index], fmt),
                     horizontalalignment="center",
                     color="white" if confusion[first_index, second_index] > thresh else "black")

    plt.show()


def compute_confusion_matrix(mat):
    nums = [12401, 1228, 367, 388, 16, 581]

    ret = []
    for type, li in enumerate(mat):
        res = []

        a_a = 0
        a_b = 0
        b_a = 0
        b_b = 0
        for i, item in enumerate(li):
            if i == type:
               a_a += round(nums[i] * item)
               a_b += round(nums[i] * (1 - item))
            else:
               b_a += round(nums[i] * (1 - item))
               b_b += round(nums[i] * item)

        res.append([[a_a / (a_a + a_b), a_b / (a_a + a_b)], [b_a / (b_a + b_b), b_b / (b_a + b_b)]])
        ret.append(res)

    return ret


if __name__ == '__main__':
    cnf_m2 = [[[0.9108136440609629, 0.08918635593903718], [0.12248062015503876, 0.8775193798449612]],
              [[0.6921824104234527, 0.30781758957654726], [0.4085654039118738, 0.5914345960881262]],
              [[0.6457765667574932, 0.3542234332425068], [0.12789106336389763, 0.8721089366361023]],
              [[0.5798969072164949, 0.42010309278350516], [0.38511615157952445, 0.6148838484204756]],
              [[1.0, 0.0], [0.0036752422318743734, 0.9963247577681257]],
              [[0.4354561101549053, 0.5645438898450946], [0.19854166666666667, 0.8014583333333334]]]
    type = 5
    plot_Matrix(np.array(cnf_m2[type]), type)


