"""
《统计学习方法》第5章 决策树
熵，条件熵，信息增益，信息增益比的计算
"""

from math import log
import numpy


def calcEntropy(dataSet, attrIndex=-1, logBase=2):
    """
	计算一个数据集再第attrIndex个属性上的熵。默认是最后一个属性，一般将样本类别最为最后一列.
	默认的底数为2。
	"""
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        curLabel = featVec[attrIndex]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel] = 0
        labelCounts[curLabel] += 1
    shannonEnt = 0.0
    for k in labelCounts.keys():
        p = float(labelCounts[k]) / numEntries
        shannonEnt -= p * log(p, logBase)
    return shannonEnt


def calcConditionalEntropy(dataSet, xIndex, yIndex=-1, logBase=2):
    """
	计算xIndex对应属性的条件熵
	"""
    # 计算P(X=xi)
    numEntries = len(dataSet)
    labelCounts = {}
    px = {}
    for featVec in dataSet:
        curLabel = featVec[xIndex]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel] = 0
        labelCounts[curLabel] += 1
    for k in labelCounts.keys():
        p = float(labelCounts[k]) / numEntries
        px[k] = p
    condEnt = 0.0
    for i in px.keys():
        di = [ds for ds in dataSet if ds[xIndex] == i]
        entDi = calcEntropy(di)
        condEnt += px[i] * entDi
    return condEnt


def fit(X, y, method):
    data = numpy.column_stack((X, y))
    testSet = data.tolist()
    fnum = len(testSet[0]) - 1

    Hd = calcEntropy(testSet)

    res = []
    for i in range(fnum):
        Hcai = calcConditionalEntropy(testSet, i)
        if method == 'ig':
            gDA = Hd - Hcai
            res.append((gDA, i))
        elif method == 'gr':
            Hai = calcEntropy(testSet, i)
            res.append((Hai, i))

        res = sorted(res, reverse=True)
        importances = []
        for e in res:
            importances.append(e[1])

    return importances
