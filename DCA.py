import copy
import random

import numpy
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# 决策树
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from Antigen import Antigen
from DC import DC
from FeatureSelection import InformationGain as ig
from FeatureSelection import SymmetricUncertainty as su
from sklearn.metrics import auc,roc_curve
import matplotlib.pyplot as plt

class DCA(object):
    # 类变量
    # 权值矩阵
    weights_paper_0 = {'csm': [2, 1, 2], 'semi': [0, 0, 1], 'mat': [2, 1, -1.5]}
    weights_paper_1 = {'csm': [2, 1, 2], 'semi': [0, 0, 2], 'mat': [0, 1, -2]}
    weights_paper_2 = {'csm': [2, 1, 2], 'semi': [0, 0, 3], 'mat': [0, 1, -3]}
    # DC细胞状态：半成熟
    semiType = 'semi'
    # DC细胞状态：成熟
    matureType = 'mature'

    def __init__(self, weightMat, safeType, dangerType, cellPoolNum, antigenCellNum,
                 iterNum, fname, solver, predictType):
        # 权值矩阵
        self.weightMat = weightMat
        # safeType
        self.safeType = safeType
        # dangerType
        self.dangerType = dangerType
        # 初始化未成熟DC细胞池的容量
        self.cellPoolNum = cellPoolNum
        # 每次随机选取的DC细胞的数量
        self.antigenCellNum = antigenCellNum
        # 增加迭代次数，默认一次 (有的论文提到了多次采集可能会增加准确率？目前没用到)
        self.iterNum = iterNum
        # 三个输入信号的最大值
        self.max_signal = None
        # csm阈值
        self.csmThreshold = None
        # mcav阈值
        self.mcavThreshold = None
        # 当前运行数据名称
        self.fname = fname
        # 当前运行的特征选择方法
        self.solver = solver
        # 保留的特征个数
        self.k = 5
        # predit函数的返回类型，True只返回y_pred, Fasle返回混淆矩阵
        self.predictType = predictType

        # 抗原对象列表
        self.antigenObjectPool = []

    def getCSMThreshold(self):
        """
        计算DC的迁移区间
        """
        max_PAMP = self.max_signal[0]
        max_ds = self.max_signal[1]
        max_ss = self.max_signal[2]

        median_threshold = 0.5 * (self.weightMat['csm'][0] * max_PAMP
                                  + self.weightMat['csm'][1] * max_ds
                                  + self.weightMat['csm'][2] * max_ss)

        return 0.5 * median_threshold, 1.5 * median_threshold

    # 1. 数据预处理部分
    def dataPreProcess(self, X, y):

        # k 数据维数
        k = len(X[0])
        # self.k 预定义为5 或 6

        # 如果书的特征维数超过self.k,那么只保留self.k个
        # 否则,保留所有
        if k >= self.k:
            k = self.k

        # 选用特征重要性排名第一的特征生成pamp和ss
        pamp_ss_index = 0
        # 剩下的生成ds
        ds_index = [i for i in range(1, k)]

        # 先转成数据框格式
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        # 标准差
        if self.solver == 'std':
            std = X.std()
            # 特征排名
            importances = pd.DataFrame({'importance': std, 'feature': X.columns}).sort_values(
                'importance', ascending=False)
            # 选取前k个特征
            importances = numpy.array(importances.head(k))[:, 1]
            importances = importances.astype(numpy.int32)
            importances = importances.tolist()

        # 信息增益
        elif self.solver == 'ig':
            importances = ig.fit(X, y, self.solver)
            importances = importances[0: k]

        # 相关系数
        elif self.solver == 'corr':
            data = numpy.column_stack((X, y))
            data = pd.DataFrame(data)
            importances = data.corr().iloc[:, -1].sort_values(ascending=False).dropna()
            feature = importances.index.tolist()
            importances = feature[1:k + 1]

        # 对称不确定性
        elif self.solver == 'su':
            importances = su.fit(X, y.values.ravel(), k)

        # 支持向量机
        elif self.solver == 'svm':
            if self.fname == 'kdd99':
                svm = LinearSVC(penalty="l1", dual=False)
            else:
                svm = LinearSVC(penalty="l1", dual=False, max_iter=10000)
            svm.fit(X, y.values.ravel())
            importances = pd.DataFrame({'importance': svm.coef_.reshape(-1, ),
                                        'feature': X.columns}).sort_values('importance', ascending=False)
            importances = numpy.array(importances.head(k))[:, 1]
            importances = importances.astype(numpy.int32)

        # 决策树
        elif self.solver == 'dt':
            dt = DecisionTreeClassifier(criterion="entropy")
            dt.fit(X, y.values.ravel())
            importances = pd.DataFrame({'importance': dt.feature_importances_, 'feature': X.columns}).sort_values(
                'importance', ascending=False)
            importances = numpy.array(importances.head(k))[:, 1]
            importances = importances.astype(numpy.int32)
            importances = importances.tolist()

        # 卡方检验，有些数据不适合
        elif self.solver == 'chi2':
            k_best = SelectKBest(chi2, k=k)
            k_best.fit_transform(X, y)
            p_values = pd.DataFrame({'column': X.columns, 'p_value': k_best.pvalues_}).sort_values('p_value')
            # print(p_values)
            importances = numpy.array(p_values.head(k))[:, 0]
            importances = importances.astype(numpy.int32)

        elif self.solver == 'ga':
            std = X.std()
            # 特征排名
            importances = pd.DataFrame({'importance': std, 'feature': X.columns}).sort_values(
                'importance', ascending=False)
            # 选取前k个特征
            importances = numpy.array(importances.head(k))[:, 1]
            importances = importances.astype(numpy.int32)
            importances = importances.tolist()

        # 其他
        else:
            print('solver error')
            return

        # 返回结果
        return importances, pamp_ss_index, ds_index, k

    # 2. 进行抗原信号映射
    def datasetToInput(self, data_x, data_y, importances, pamp_ss_index, ds_index, k):

        importances = importances[0:k]
        print('feature_importance_index: ', importances)

        data_x = data_x[:, importances]

        # 似乎mm不适合标准化？
        if self.fname != 'mm':
            min_max = MinMaxScaler(feature_range=(0, 100))
            data_x = min_max.fit_transform(data_x)

        safyNum = 0
        dangerNum = 0

        for e in data_y:
            if e == self.safeType:
                safyNum += 1
            else:
                dangerNum += 1

        self.mcavThreshold = dangerNum / (safyNum + dangerNum)

        print('mcavThreshold', self.mcavThreshold)

        # 数据集
        dataset_before = copy.deepcopy(data_x)
        # 抗原集（即数据集进行信号映射后的）
        dataset_after = []
        # 三个输入信号中最大的
        max_pamp = 0
        max_ss = 0
        max_ds = 0
        # 遍历抗原数据集，将数据集中的属性映射成输入信号
        v_mean = numpy.mean(dataset_before[pamp_ss_index])

        for row in dataset_before:
            # 第一特征计算pamp ss
            if row[pamp_ss_index] > v_mean:
                pamp = abs(row[pamp_ss_index] - v_mean)
                ss = 0
            else:
                ss = abs(row[pamp_ss_index] - v_mean)
                pamp = 0

            # 其余特征用于计算ds
            ds = numpy.mean(row[ds_index])

            # 顺便计算三个输入信号的最大值
            if pamp > max_pamp:
                max_pamp = pamp
            if ds > max_ds:
                max_ds = ds
            if ss > max_ss:
                max_ss = ss

            dataset_after.append([pamp, ds, ss])

        # 加入最大信号集
        self.max_signal = [max_pamp, max_ds, max_ss]
        # 计算迁移阈值范围
        self.csmThreshold = self.getCSMThreshold()

        print('self.csmThreshold', self.csmThreshold)

        # 为X添加上label
        dataset_after = numpy.insert(dataset_after, 3, data_y, axis=1)

        # 抗原对象列表
        self.antigenObjectPool.clear()
        for row in dataset_after:
            pamp = float(row[0])
            ds = float(row[1])
            ss = float(row[2])
            lable = float(row[3])
            self.antigenObjectPool.append(Antigen(pamp, ds, ss, lable))

    # 3. 输入信号进行信号转换成输出信号
    def inputToOutput(self):
        # 初始化未成熟DC池
        immature = [DC(self.csmThreshold) for x in range(self.cellPoolNum)]
        # iterNum
        for r in range(self.iterNum):
            # 对于每个抗原向量（即数据集中的每条数据）
            for anti in self.antigenObjectPool:
                # 随机选取DC细胞来处理此抗原
                tempCells = [immature[x] for x in random.sample(range(self.cellPoolNum), self.antigenCellNum)]
                # 对于每个选取的细胞
                for cell in tempCells:
                    # 将此细胞记录到抗原对应的细胞列表里
                    anti.cells.append(cell)
                    # 计算该细胞的输出值，累加

                    # cell.csm += (self.weightMat['csm'][0] * anti.pamp + self.weightMat['csm'][1] * anti.ds
                    #              + self.weightMat['csm'][2] * anti.ss) / \
                    #             (abs(self.weightMat['csm'][0]) + abs(self.weightMat['csm'][1])
                    #              + abs(self.weightMat['csm'][2]))
                    #
                    # cell.semi += (self.weightMat['semi'][0] * anti.pamp + self.weightMat['semi'][1] * anti.ds +
                    #               self.weightMat['semi'][2] * anti.ss) / \
                    #              (abs(self.weightMat['semi'][0]) + abs(self.weightMat['semi'][1]) +
                    #               abs(self.weightMat['semi'][2]))
                    #
                    # cell.mat += (self.weightMat['mat'][0] * anti.pamp + self.weightMat['mat'][1] * anti.ds +
                    #              self.weightMat['mat'][2] * anti.ss) / \
                    #             (abs(self.weightMat['mat'][0]) + abs(self.weightMat['mat'][1]) +
                    #              abs(self.weightMat['mat'][2]))
                    #
                    cell.csm += (self.weightMat['csm'][0] * anti.pamp + self.weightMat['csm'][1] * anti.ds
                                 + self.weightMat['csm'][2] * anti.ss)

                    cell.semi += (self.weightMat['semi'][0] * anti.pamp + self.weightMat['semi'][1] * anti.ds +
                                  self.weightMat['semi'][2] * anti.ss)

                    cell.mat += (self.weightMat['mat'][0] * anti.pamp + self.weightMat['mat'][1] * anti.ds +
                                 self.weightMat['mat'][2] * anti.ss)
                    # 判断细胞是否迁移
                    if cell.isMigrate():
                        # 判断迁移状态
                        if cell.semi > cell.mat:
                            cell.type = self.semiType
                        else:
                            cell.type = self.matureType
                        # 将此细胞从未成熟DC池中移除
                        immature.remove(cell)
                        # 未成熟DC池中重新添加一个新的DC细胞（这点很重要）
                        immature.append(DC(self.csmThreshold))

    # 模型计算过程
    def fit(self, data_x, data_y):  # data_x, data_y numpy类型
        # 1. 数据预处理部分
        importances, pamp_ss_index, ds_index, k = self.dataPreProcess(data_x, data_y)
        # 2. 数据集进行信号映射
        self.datasetToInput(data_x, data_y, importances, pamp_ss_index, ds_index, k)
        # 3. 输入信号进行信号转换成输出信号
        self.inputToOutput()

    # 输出模型的预测（分类结果）
    def predict(self, y):
        y_pred = []
        # 在完成了所有的抗原提呈工作后，逐个判断他们状态，即预测（分类）过程
        total = 0
        for antigen in self.antigenObjectPool:
            total += 1
            count_mat = 0
            count_semi = 0
            for c in antigen.cells:
                if c.type == self.matureType:
                    count_mat += 1
                elif c.type == self.semiType:
                    count_semi += 1

            # if count_mat + count_semi != 10:
            #     print(total)
            if count_mat + count_semi == 0:
                mcav = 0
            else:
                mcav = count_mat / (count_mat + count_semi)

            # mcav = count_mat / 10

            if mcav > self.mcavThreshold:
                antigen.type = self.dangerType
                y_pred.append(1)
            else:
                antigen.type = self.safeType
                y_pred.append(0)
        # 计算混淆矩阵
        tp, tn, fp, fn = 0, 0, 0, 0
        for antigen in self.antigenObjectPool:
            if antigen.originalType == self.dangerType and antigen.type == self.dangerType:
                tn += 1
            elif antigen.originalType == self.dangerType and antigen.type == self.safeType:
                fp += 1
            elif antigen.originalType == self.safeType and antigen.type == self.safeType:
                tp += 1
            elif antigen.originalType == self.safeType and antigen.type == self.dangerType:
                fn += 1
            else:
                print('classify error')

        # 只返回y_pred是模拟sklearn的函数接口，为了用于遗传算法
        if self.predictType:
            return y_pred
        else:
            return tp, tn, fp, fn, y_pred

    @staticmethod
    def report(tp, tn, fp, fn, y, y_pred):
        # 用于计算准确度
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        # 用于计算特效度，误报率
        if (tn + fp) == 0:
            specificity = 0
            fpr = 1
        else:
            specificity = tn / (tn + fp)
            fpr = fp / (tn + fp)
        # 用于计算精确度
        if (tp + fp) == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        # 用于计算召回率，误报率
        if (tp + fn) == 0:
            recall = 0
            fnr = 0
        else:
            recall = tp / (tp + fn)
            fnr = fn / (tp + fn)
        # 用于计算F1分数
        if (precision + recall) == 0:
            fm = 0
        else:
            fm = 2 * precision * recall / (precision + recall)

        # MCC
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
            MCC = 0
        else:
            MCC = (tp * tn - fp * fn) / (numpy.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        
        # 用于计算AUC
        y_pred = [ int(x) for x in y_pred ]
        y = [ int(x) for x in y ]

        fpr1, tpr1, thresholds = roc_curve(y, y_pred)
        a = auc(fpr1,tpr1)
        plt.plot(fpr1, tpr1, 'k--', label='ROC(area={0:.2f})'.format(a), lw=2)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

        print('\n数据集总条数： ', (tn + fn + tp + fp))
        print('tn异常数据：预测异常的条数： ', tn)
        print('fn正常数据：预测异常的条数：', fn)
        print('tp正常数据：预测正常的条数：', tp)
        print('fp异常数据：预测正常的条数：', fp)
        print('\n\n【召回率，对正例的识别能力】recall： ', recall)
        print('【特效度，对负类的识别能力】specificity： ', specificity)
        print('【分类器的准确度↑】Acuuracy： ', accuracy)
        print('【漏报率↓】FNR： ', fnr)
        print('【误报率↓】FPR： ', fpr)
        print('【精度，正例中正确的比例】precision： ', precision)
        print('【综合评价指标（F-1） ↑】', fm)
        print('【MCC】', MCC)
        print('【auc】', a)

        res = {'recall': recall, 'specificity': specificity, 'accuracy': accuracy, 'f1': fm, 'auc': a}

        return res
