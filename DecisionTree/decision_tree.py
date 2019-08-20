import math
import copy

def creatDataset():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no'],
            ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        curLabel = featVec[-1]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel] = 0
        labelCounts[curLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts.keys():
        prob = (float)(labelCounts[key]) / numEntries
        # 计算香农熵，以 2 为底求对数
        shannonEnt -= prob * math.log(prob, 2)

    return shannonEnt


def splitDataSet(dataSet, index, value):
    """
       splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
       就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
       dataSet 数据集                 待划分的数据集
       index 表示每一行的index列        划分数据集的特征
       value 表示index列对应的value值   需要返回的特征的值。
    Returns:
       index列为value的数据集【该数据集需要排除index列】
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[: index]
            reducedFeatVec.extend(featVec[index + 1 :])
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """chooseBestFeatureToSplit(选择最好的特征)
    Args:
        dataSet 数据集
    Returns:
        bestFeature 最优的特征列
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntroy = calcShannonEnt(dataSet)
    # 最优的信息增益值, 和最优的Featurn编号
    bestInfoGain, bestFeature = 0.0, -1
    # iterate over all the features
    for i in range(numFeatures):
        # create a list of all the examples of this feature
        # 获取对应的feature下的所有数据
        featList = [example[i] for example in dataSet]
        #get unique values
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 遍历某一列的value集合，计算该列的信息熵
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            subDataset = splitDataSet(dataSet, i, value)
            prob = len(subDataset) / (float)(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataset)

        infoGain = baseEntroy - newEntropy

        print('infoGain = %f, bestFeature = %d,  %f,  %f' % (infoGain, i, baseEntroy, newEntropy))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

def majorityCnt(classList):
    """majorityCnt(选择出现次数最多的一个结果)
    Args:
        classList label列的集合
    Returns:
        bestFeature 最优的特征列
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def creatTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
        # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree

def classify(inputTree, featLabels, testVec):
    """classify(给输入的节点，进行分类)
    Args:
        inputTree  决策树模型
        featLabels Feature标签对应的名称
        testVec    测试输入的数据
    Returns:
        classLabel 分类的结果值，需要映射label才能知道名称
    """
    # 获取tree的根节点对于的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat

    return classLabel


def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = creatDataset()
    myTree = creatTree(myDat, copy.deepcopy(labels))
    print(myTree)
    print(classify(myTree, labels, [1, 1]))

    #print(get_tree_height(myTree))

    #dtPlot.creatPlot(myTree)

if __name__ == "__main__":
    fishTest()
