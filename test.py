import time
from copy import deepcopy
from DCA import DCA
import numpy
import pandas as pd


def str_column_to_int(dataset, column):
    for row in dataset:
        try:
            row[column] = int(row[column].strip())
        except ValueError:
            # print("str_column_to_int Error with row", column, ":", row[column])
            row[column] = 0

# 把datasettofloat中的第column列转变为浮点型
def str_column_to_float(dataset, column):
    for row in dataset:
        try:
            row[column] = float(row[column].strip())
        except ValueError:
            if row[column] == 'b':
                row[column] = 1.0
            elif row[column] == 'g':
                row[column] = 0.0


def import_data(file, sortData, asc, header):
    if header:
        data = pd.read_csv('D:/Learning/17-Algorithms/DCA-凯林/DCA/data/' + file)
    else:
        data = pd.read_csv('D:/Learning/17-Algorithms/DCA-凯林/DCA/data/' + file, header=None)

    data_x = numpy.array(data.iloc[:, :-1])
    data_y = numpy.array(data.iloc[:, -1])

    data = numpy.column_stack((data_x, data_y))

    if sortData:
        data = data[numpy.argsort(data[:, -1], )]
        if asc:
            data = data[::-1]

    data_y = data[:, -1]
    data_x = data[:, :-1]

    return data_x, data_y


def import_data_ga(file, sortData, asc, header):
    if header:
        data = pd.read_csv("D:/Learning/17-Algorithms/DCA-凯林/DCA/data/" + file)
    else:
        data = pd.read_csv('D:/Learning/17-Algorithms/DCA-凯林/DCA/data/' + file, header=None)

    data_x = numpy.array(data.iloc[:, :-1])
    data_y = numpy.array(data.iloc[:, -1])

    data = numpy.column_stack((data_x, data_y))

    if sortData:
        data = data[numpy.argsort(data[:, -1], )]
        if asc:
            data = data[::-1]

    data_y = data[:, -1]
    data_x = data[:, :-1]

    return data_x, data_y


# 计算DCA
def dca_model(X, y, run_num, solver, safeType, dangerType, iterNum, fname, predictType):
    print('----------------------------', solver, '--------------------------')

    start_time, end_time, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0

    run_num = run_num

    for i in range(run_num):
        model_dca = DCA(weightMat=DCA.weights_paper_0, safeType=safeType, dangerType=dangerType, cellPoolNum=100,
                        antigenCellNum=10, iterNum=iterNum, fname=fname, solver=solver, predictType=predictType)

        t_start_time = time.time()
        model_dca.fit(X, y)
        t_tp, t_tn, t_fp, t_fn, y_pre = model_dca.predict(y)
        t_end_time = time.time()

        start_time += t_start_time
        end_time += t_end_time
        tp += t_tp
        tn += t_tn
        fp += t_fp
        fn += t_fn
        print(i, '【用时：】', str((t_end_time - t_start_time)), 'sec')

    print('执行', str(run_num), '次后的平均结果：')
    res = DCA.report(tp / run_num, tn / run_num, fp / run_num, fn / run_num, y, y_pre)
    ttt = (end_time - start_time) / run_num
    res['time'] = ttt
    return {solver: res}


def job(file, run_num):
    run_num = run_num
    iterNum = 1
    sortData = True
    header = False
    asc = True

    if file == 'kdd99':
        asc = False
    else:
        asc = True

    safeType = 0
    dangerType = 1

    X, y = import_data(file=file + '.csv', sortData=sortData, asc=asc, header=header)

    print(" X.shape: ", X.shape)
    print(" y.shape: ", y.shape)
    print('\n\n')

    solvers = []

    e = dca_model(deepcopy(X), deepcopy(y), run_num=run_num, safeType=safeType,
                  dangerType=dangerType, iterNum=iterNum, fname=file, solver='std', predictType=False)
    solvers.append(e)

    e = dca_model(deepcopy(X), deepcopy(y), run_num=run_num, safeType=safeType,
                  dangerType=dangerType, iterNum=iterNum, fname=file, solver='ig', predictType=False)
    solvers.append(e)

    e = dca_model(deepcopy(X), deepcopy(y), run_num=run_num, safeType=safeType,
                  dangerType=dangerType, iterNum=iterNum, fname=file, solver='su', predictType=False)
    solvers.append(e)

    e = dca_model(deepcopy(X), deepcopy(y), run_num=run_num, safeType=safeType,
                  dangerType=dangerType, iterNum=iterNum, fname=file, solver='dt', predictType=False)
    solvers.append(e)

    e = dca_model(deepcopy(X), deepcopy(y), run_num=run_num, safeType=safeType,
                  dangerType=dangerType, iterNum=iterNum, fname=file, solver='svm', predictType=False)
    solvers.append(e)

    return solvers


if __name__ == '__main__':

    pd.set_option('display.max_columns', 20)
    pd.set_option('expand_frame_repr', False)

    # files = ['spambase', 'kdd99']
    files = ['mm', 'cancer', 'heart', 'cc', 'wine', 'irist', 'park']
    # files = ['cancer']
    # files = ['DS1','DS2','DS3','DS4']
    # files = ['Gansu-DS1-indicators-345','Qinghai-DS2-indicators-345','Sichuan-DS3-indicators-345','Yunnan-DS4-indicators-345']

    solvers = {}

    for f in files:
        print('\n', f, '\n')
        res = job(f, 1)
        solvers[f] = res

    recall = []
    specificity = []
    accuracy = []
    f1 = []
    auc = []
    time = []

    v_index = []
    v_columns = []
    for k in solvers:
        v_index.append(k)
        methods = solvers[k]
        sub_recall = []
        sub_specificity = []
        sub_accuracy = []
        sub_f1 = []
        sub_auc = []
        sub_time = []
        for m in methods:
            for kk in m:
                v_columns.append(kk)
                sub_recall.append(round(m[kk]['recall'], 4))
                sub_specificity.append(round(m[kk]['specificity'], 4))
                sub_accuracy.append(round(m[kk]['accuracy'], 4))
                sub_f1.append(round(m[kk]['f1'], 4))
                sub_auc.append(round(m[kk]['auc'], 4))
                sub_time.append(round(m[kk]['time'], 4))
        recall.append(sub_recall)
        specificity.append(sub_specificity)
        accuracy.append(sub_accuracy)
        f1.append(sub_f1)
        auc.append(sub_auc)
        time.append(sub_time)

    # v_columns = set(v_columns)
    len_columns = len(v_columns)
    len_index = len(v_index)
    print(v_columns)
    v_columns = v_columns[:(len_columns // len_index)]

    print(v_columns)
    print(v_index)
    print(recall)
    print(specificity)
    print(accuracy)
    print(f1)
    print(time)

    recall = pd.DataFrame(data=recall, index=v_index, columns=v_columns)
    mean_before = pd.DataFrame(round(recall.mean(), 4), columns=['average'])
    mean_after = pd.DataFrame(mean_before.values.T, index=mean_before.columns, columns=mean_before.index)
    recall = recall.append(mean_after)

    specificity = pd.DataFrame(data=specificity, index=v_index, columns=v_columns)
    mean_before = pd.DataFrame(round(specificity.mean(), 4), columns=['average'])
    mean_after = pd.DataFrame(mean_before.values.T, index=mean_before.columns, columns=mean_before.index)
    specificity = specificity.append(mean_after)

    accuracy = pd.DataFrame(data=accuracy, index=v_index, columns=v_columns)
    mean_before = pd.DataFrame(round(accuracy.mean(), 4), columns=['average'])
    mean_after = pd.DataFrame(mean_before.values.T, index=mean_before.columns, columns=mean_before.index)
    accuracy = accuracy.append(mean_after)

    f1 = pd.DataFrame(data=f1, index=v_index, columns=v_columns)
    mean_before = pd.DataFrame(round(f1.mean(), 4), columns=['average'])
    mean_after = pd.DataFrame(mean_before.values.T, index=mean_before.columns, columns=mean_before.index)
    f1 = f1.append(mean_after)

    auc = pd.DataFrame(data=auc, index=v_index, columns=v_columns)
    mean_before = pd.DataFrame(round(auc.mean(), 4), columns=['average'])
    mean_after = pd.DataFrame(mean_before.values.T, index=mean_before.columns, columns=mean_before.index)
    auc = auc.append(mean_after)

    time = pd.DataFrame(data=time, index=v_index, columns=v_columns)
    mean_before = pd.DataFrame(round(time.mean(), 4), columns=['average'])
    mean_after = pd.DataFrame(mean_before.values.T, index=mean_before.columns, columns=mean_before.index)
    time = time.append(mean_after)

    print("---------------------recall---------------------------")
    print(recall)
    print("---------------------specificity---------------------------")
    print(specificity)
    print("---------------------accuracy---------------------------")
    print(accuracy)
    print("---------------------f1---------------------------")
    print(f1)
    print("---------------------auc---------------------------")
    print(auc)
    print("---------------------time---------------------------")
    print(time)
