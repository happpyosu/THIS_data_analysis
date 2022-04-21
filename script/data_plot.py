import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_matrix(m):
    plt.matshow(m, cmap=plt.cm.Blues)
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
        - cm : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
        """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    cnf_m = [[0.32521570841061204, 0.255700325732899, 0.5667574931880109, 0.47164948453608246, 0.6875, 0.4199655765920826],
         [0.3289250866865575, 0.26302931596091206, 0.5722070844686649, 0.422680412371134, 0.6875, 0.387263339070568],
         [0.32731231352310297, 0.2768729641693811, 0.6049046321525886, 0.44329896907216493, 0.6875, 0.3993115318416523],
         [0.3158616240625756, 0.25895765472312704, 0.5858310626702997, 0.46134020618556704, 0.6875, 0.3769363166953528],
         [0.3230384646399484, 0.254885993485342, 0.5504087193460491, 0.4484536082474227, 0.6875, 0.4165232358003442],
         [0.32118377550197563, 0.253257328990228, 0.553133514986376, 0.42783505154639173, 0.6875, 0.423407917383821]]
    plot_matrix(cnf_m)