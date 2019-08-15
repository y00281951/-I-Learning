import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator

def file2matrix(filename):
    """
    Desc:
        导入训练数据
    parameters:
        filename: 数据文件路径
    return:
        数据矩阵 returnMat 和对应的类别 classLabelVector
    """
    f = open(filename)
    num_Lines = len(f.readlines())
    featMat= np.zeros((num_Lines, 3))
    lableVector = []
    index = 0

    f = open(filename)
    for line in f.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        featMat[index:] = listFromLine[0:3]
        lableVector.append(int(listFromLine[-1]))
        index += 1

    return featMat, lableVector

def featLablePlot(featMat, lable):
    """
    Desc:
       　打印特征和标签
    parameters:
         featMat: 特征
         lable: 数据标签
    return:
         None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(featMat[:, 0], featMat[:, 1], 15.0 * np.array(lable), 15.0 * np.array(lable))
    plt.show()

def autoNorm(dataSet):
    """
    Desc:
        归一化特征值，消除特征之间量级不同导致的影响
    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到
        归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max
        分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    #print(dataSet.shape)
    minVals = dataSet.min(0)
    #print(minVals.shape)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros((dataSet.shape))
    m = dataSet.shape[0]

    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals

def classify(inX, dataSet, lables, k):
    """
    Desc:
        分类函数
    parameters:
        inX: 输入特征
        dataSet:　样本特征
        lable: 数据标签
        k: k值
    return:
         None
    """
    dataSetSize = dataSet.shape[0]
    difMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDifMat = difMat ** 2
    sqDistances = sqDifMat.sum(axis = 1)
    distances = sqDistances ** 0.5

    sortDistances = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = lables[sortDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


if __name__ == "__main__":
    resultList = ['not at all', 'in small doses', 'in large doses']

    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    feat, lable = file2matrix('datingTestSet2.txt')
    #featLablePlot(feat, lable)
    norFeat, ranges, minVals = autoNorm(feat)

    inArray = np.array([ffMiles, percentTats, iceCream])

    classifierResult = classify((inArray - minVals) / ranges, norFeat, lable, 3)
    print("resutl is %s " % resultList[classifierResult - 1])






