import math
import numpy as np

def loadDataSet():
    """
    创建数据集
    return: 单词列表postingList,所属类别的classVec
    """
    postingList = [ ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'] ]
    classVec = [0, 1, 0, 1, 0, 1] # 1 is abusive, 0 not]

    return postingList, classVec


def creatVocabList(dataSet):
    """
    获取所有单词的集合
    param:
        dataSet: 数据集
    return:
        所有单词的集合（不含重复元素的单词列表）
    """
    vocabSet = set([]) #creat empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    return vocabSet


def setOfWords2Vec(vocabList, inputSet):
    """
    遍历查看该单词是否出现，出现该单词则将该单词署1
    param:
        vocabList:所有单词的列表集合
        inputSet:输入数据集
    return:
        匹配列表[0,1,0,1...]，其中1和0表示词汇表中的单词是否出现在输入的数据集中
    """
    #创建一个与词汇表等长的向量
    returnVec = [0] * len(vocabList)
    #遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文件向量中的对应值设置为1
    for word in inputSet:
        if word in vocabList:
            returnVec[list(vocabList).index(word)] = 1

    return returnVec


def _trainNB0(trainMatrix, trainCategory):
    """
    训练数据原版
    param:
        trainMatrix: 文件单词矩阵
        trainCategory:文件对应的类别
    return:
    """
    #文件数
    numTrainDocs = len(trainMatrix)
    #单词数
    numWords = len(trainMatrix[0])
    #侮辱性文件出现的概率，即trainCategory中1的个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    #构造单词出现次数列表
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)


    #整个数据集单词出现的次数
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            #如果是侮辱性文件，对侮辱性文件的向量进行加和
            p1Num += trainMatrix[i] #[0, 1, ,1...] + [0, 1, 1,...] = [0, 2, 2, ...]
            #对向量中的所有元素进行求和，也就是计算所有侮辱性文件中出现的单词总数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    #类别1，即侮辱性文档的[P(F1|C1), P(F2|C1), P(F3|C1), ....]列表
    #即在1类别下，每个单词出现的概率
    p1Vect = np.log(p1Num / p1Denom)
    #即在0类别下，每个单词出现的概率
    p0Vect = np.log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    使用方法：
        蒋乘法转换为加法
        P(F1|C) * P(F2|C) ...P(Fn|C) -> log(P(F1|C)) + log(P(F2|C)) + ...+ log(P(Fn|C))
    param:
        vec2Classify: 待测数据[0,1,1,1,...]，即要分类的向量
        p0Vec:类别0，即正常文档的[log(F1|C0),log(F2|C0)....]
        p1Vec:类别1，即侮辱文档的[log(F1|C1),log(F2|C1)....]
        pClass1: 类别1，侮辱性文件的出现概率
    return:
        类别1 or 0
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1) #p(w|c1) * p(c1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1) #p(w|c8) * p(c0)

    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
    测试相互贝叶斯算法
    """
    #1.加载数据集
    listOfPosts, listClasses = loadDataSet()
    #2.创建单词集合
    myVocabList = creatVocabList(listOfPosts)
    #3.计算单词是否出现在数据矩阵
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #4.训练数据
    p0V, p1V, pAb = _trainNB0(np.array(trainMat), np.array(listClasses))
    #5.测试数据
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))

    result = classifyNB(thisDoc, p0V, p1V, pAb)

    print("%s is %d" % ((testEntry), result))

    testEntry = ['love', 'stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))

    result = classifyNB(thisDoc, p0V, p1V, pAb)

    print("%s is %d" % ((testEntry), result))


if __name__ == '__main__':
    testingNB()
