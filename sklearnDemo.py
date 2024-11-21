import random

import numpy as np
import pandas as pd
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn import datasets  # 引入数据集,sklearn包含众多数据集
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 数据标准化
from sklearn.decomposition import PCA, TruncatedSVD  # 主成分分析
from imblearn.over_sampling import SMOTE  # 上采样
from sklearn.model_selection import train_test_split, RepeatedKFold, TimeSeriesSplit  # 将数据分为测试集和训练集
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest

from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.linear_model import LogisticRegression, Lasso  # Logistic
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier  # adaboost, 随机森林
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.svm import SVC, LinearSVC  # SVM
from xgboost import XGBClassifier  # xgboost
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score  # 评分指标


def kFold_score(X, y, n_splits, n_repeats, model):
    sumP = 0
    sumR = 0
    sumF1 = 0
    sumA = 0
    sumAUC = 0

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    for train_index, test_index in rkf.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        # 训练模型
        model = model.fit(X_train, y_train)

        # # 训练集性能
        # y_hat = model.predict(X_train)
        # y_real = y_train
        #
        # P = precision_score(y_real, y_hat)
        # R = recall_score(y_real, y_hat)
        # F1 = f1_score(y_real, y_hat)
        # A = accuracy_score(y_real, y_hat)
        # AUC = roc_auc_score(y_real, y_hat)
        # P = format(P, '.4f')
        # R = format(R, '.4f')
        # F1 = format(F1, '.4f')
        # A = format(A, '.4f')
        # AUC = format(AUC, '.4f')
        # print('训练集性能\t', end='')
        # print(f'P:{P}\tR:{R}\tF1:{F1}\tA:{A}\tAUC:{AUC}\t', end='')

        # 测试集性能
        y_hat = model.predict(X_test)
        y_real = y_test

        sumP += precision_score(y_real, y_hat)
        sumR += recall_score(y_real, y_hat)
        sumF1 += f1_score(y_real, y_hat)
        sumA += accuracy_score(y_real, y_hat)
        sumAUC += roc_auc_score(y_real, y_hat)

    P = format(sumP / (n_splits * n_repeats), '.4f')
    R = format(sumR / (n_splits * n_repeats), '.4f')
    F1 = format(sumF1 / (n_splits * n_repeats), '.4f')
    A = format(sumA / (n_splits * n_repeats), '.4f')
    AUC = format(sumAUC / (n_splits * n_repeats), '.4f')
    print('测试集性能\t', end='')
    print(f'P:{P}\tR:{R}\tF1:{F1}\tA:{A}\tAUC:{AUC}')
    return A


def time_split(model, X, y):
    len = X.shape[0]
    split = int(len / 5)
    X_train, X_test, y_train, y_test = X[:int(split * 4)], X[int(split * 4):], y[:int(split * 4)], y[int(split * 4):]
    # 训练模型
    model = model.fit(X_train, y_train)
    # 测试集性能
    y_hat = model.predict(X_test)
    y_real = y_test
    P = precision_score(y_real, y_hat)
    R = recall_score(y_real, y_hat)
    F1 = f1_score(y_real, y_hat)
    A = accuracy_score(y_real, y_hat)
    AUC = roc_auc_score(y_real, y_hat)
    P = format(P, '.4f')
    R = format(R, '.4f')
    F1 = format(F1, '.4f')
    A = format(A, '.4f')
    AUC = format(AUC, '.4f')
    print('测试集性能\t', end='')
    print(f'P:{P}\tR:{R}\tF1:{F1}\tA:{A}\tAUC:{AUC}')
    return P, R, F1, A, AUC


def random_split(X, y, model, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=random_state)
    # 训练模型
    model = model.fit(X_train, y_train)
    # 测试集性能
    y_hat = model.predict(X_test)
    y_real = y_test
    P = precision_score(y_real, y_hat)
    R = recall_score(y_real, y_hat)
    F1 = f1_score(y_real, y_hat)
    A = accuracy_score(y_real, y_hat)
    AUC = roc_auc_score(y_real, y_hat)
    P = format(P, '.4f')
    R = format(R, '.4f')
    F1 = format(F1, '.4f')
    A = format(A, '.4f')
    AUC = format(AUC, '.4f')
    print('测试集性能\t', end='')
    print(f'P:{P}\tR:{R}\tF1:{F1}\tA:{A}\tAUC:{AUC}')
    return P, R, F1, A, AUC


def timeSeries_split(model, X, y, n_splits=10, max_train_size=1000):
    sumP = 0
    sumR = 0
    sumF1 = 0
    sumA = 0
    sumAUC = 0
    tss = TimeSeriesSplit(max_train_size=max_train_size, n_splits=n_splits)

    for train_index, test_index in tss.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        # 训练模型
        model = model.fit(X_train, y_train)

        # # 训练集性能
        # y_hat = model.predict(X_train)
        # y_real = y_train
        #
        # P = precision_score(y_real, y_hat)
        # R = recall_score(y_real, y_hat)
        # F1 = f1_score(y_real, y_hat)
        # A = accuracy_score(y_real, y_hat)
        # AUC = roc_auc_score(y_real, y_hat)
        # P = format(P, '.4f')
        # R = format(R, '.4f')
        # F1 = format(F1, '.4f')
        # A = format(A, '.4f')
        # AUC = format(AUC, '.4f')
        # print('训练集性能\t', end='')
        # print(f'P:{P}\tR:{R}\tF1:{F1}\tA:{A}\tAUC:{AUC}\t', end='')

        # 测试集性能
        y_hat = model.predict(X_test)
        y_real = y_test

        sumP += precision_score(y_real, y_hat)
        sumR += recall_score(y_real, y_hat)
        sumF1 += f1_score(y_real, y_hat)
        sumA += accuracy_score(y_real, y_hat)
        sumAUC += roc_auc_score(y_real, y_hat)

    P = format(sumP / n_splits, '.4f')
    R = format(sumR / n_splits, '.4f')
    F1 = format(sumF1 / n_splits, '.4f')
    A = format(sumA / n_splits, '.4f')
    AUC = format(sumAUC / n_splits, '.4f')
    print('测试集性能\t', end='')
    print(f'P:{P}\tR:{R}\tF1:{F1}\tA:{A}\tAUC:{AUC}')
    return A


def result(model, X, y, random_state):
    kFold_score(X, y, n_splits=5, n_repeats=4, model=model)
    # time_split(model, X, y)
    # random_split(X, y, model, random_state)
    # timeSeries_split(model, X, y, n_splits=10)


def getXy0(spl, df):
    print(f"*******************FS*******************")

    X = df.iloc[:, :spl]
    y = df.iloc[:, -1]
    columns = X.columns  # 先保存特征名称
    # 标准化  # 归一化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Lasso
    X = pd.DataFrame(X, columns=columns)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    # lasso.coef_相关系数列表
    coef = pd.DataFrame(lasso.coef_, index=X.columns)
    # 返回相关系数是否为0的布尔数组
    mask = lasso.coef_ != 0.0
    # print(coef[mask])
    # 对特征进行选择
    X = X.values
    X = X[:, mask]
    y = y.values
    print(X.shape)
    return X, y


def getXy1(spl, n_components, df):
    print(f"*******************WT*******************")

    X1 = df.iloc[:, spl:-1]
    y = df.iloc[:, -1]

    # 标准化 归一化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X1)

    # 主成分分析压缩维度
    pca = PCA(n_components=n_components, svd_solver='full')
    X1 = pca.fit_transform(X1)

    X = X1
    y = y.values
    print(X.shape)
    return X, y


def getXy2(spl, n_components, df):
    print(f"*******************FS+BW*******************")

    X = df.iloc[:, :spl]
    y = df.iloc[:, -1]
    X1 = df.iloc[:, spl:-1]

    # 标准化  # 归一化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Lasso
    # columns = X.columns  # 先保存特征名称
    # X = pd.DataFrame(X, columns=columns)

    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    # lasso.coef_相关系数列表
    # coef = pd.DataFrame(lasso.coef_, index=X.columns)
    # 返回相关系数是否为0的布尔数组
    mask = lasso.coef_ != 0.0
    # print(mask)
    # 对特征进行选择
    X = X[:, mask]

    # 标准化 # 归一化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X1)

    # 主成分分析压缩维度
    pca = PCA(n_components=n_components, svd_solver='full')
    X1 = pca.fit_transform(X1)

    X = np.concatenate((X, X1), axis=1)
    y = y.values
    print(X.shape)
    return X, y


def getXy3(years):
    print(f"*******************WA+WT+BS+BW*******************")

    df1 = pd.read_csv(f'data/1比1_Word2Vec_average_{years}years.csv', encoding='utf-8', dtype='float32')
    X1 = df.iloc[:, -201:-1]

    df2 = pd.read_csv(f'data/1比1_Word2Vec_tf-idf_{years}years.csv', encoding='utf-8', dtype='float32')
    X2 = df2.iloc[:, -201:-1]

    df3 = pd.read_csv(f'data/1比1_bert_sentences_{years}years.csv', encoding='utf-8', dtype='float32')
    X3 = df3.iloc[:, -769:-1]

    df4 = pd.read_csv(f'data/1比1_bert_words_{years}years.csv', encoding='utf-8', dtype='float32')
    X4 = df4.iloc[:, -769:-1] / 223

    X = pd.concat([X1, X2], axis=1)
    y = df1.iloc[:, -1]

    # 标准化 归一化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # 主成分分析压缩维度
    pca = PCA(n_components=0.85, svd_solver='full')
    X = pca.fit_transform(X)

    y = y.values
    print(X.shape)
    return X, y


if __name__ == '__main__':
    name_ls = ['Word2Vec_average_', 'Word2Vec_tf-idf_', 'bert_sentences_', 'bert_words_']
    alpha = 0.005  # lasso L1系数
    n_components = 0.85  # pca保留特征信息量
    spl = -769  # 从哪截断， word2vec 200维; bert 768维
    # # 数据集
    for i in range(4):
        years = 2 + i
        filename = 'Word2Vec_tf-idf_'
        if "Word2Vec" in filename:
            spl = -201
        elif "bert" in filename:
            spl = -769
        else:
            spl = -74
        path = f'data/1比1_{filename}{years}years.csv'
        df = pd.read_csv(path, encoding='utf-8', dtype='float32')
        df = df.iloc[::-1]
        print(f"*******************{years}*******************")
        ls = [0, 1, 2]
        for target in ls:
            if target == 0:  # 数值
                X, y = getXy0(spl, df)
            elif target == 1:  # 文本
                X, y = getXy1(spl, n_components, df)
            elif target == 2:  # 结合
                X, y = getXy2(spl, n_components, df)

            for j in range(1):
                random.seed(10 * j + 1)
                random_state = random.randint(0, 100000)

                # # KNN
                # n_neighbors = int(X.shape[1] / 3)
                # knn = KNeighborsClassifier()
                # print('     knn:\t\t', end='')
                # result(knn, X, y, random_state)

                # logistic
                logistic = LogisticRegression(penalty="l2", max_iter=5000, solver='liblinear')
                print('logistic:\t\t', end='')
                result(logistic, X, y, random_state)

                # # adaboost
                # adaboost = AdaBoostClassifier()
                # print('adaboost:\t\t', end='')
                # result(adaboost, X, y, random_state)
                #
                # # 决策树
                # decisionTree = DecisionTreeClassifier()
                # print('decisionTree:\t', end='')
                # result(decisionTree, X, y, random_state)

                # 随机森林
                randomForest = RandomForestClassifier(n_estimators=200)
                print('randomForest:\t', end='')
                result(randomForest, X, y, random_state)

                # SVM
                SVM = SVC()  # cache_size 允许使用的内存大小
                print('     SVM:\t\t', end='')
                result(SVM, X, y, random_state)

                # xgboost
                xgboost = XGBClassifier(n_estimators=200)
                print('xgboost:\t\t', end='')
                result(xgboost, X, y, random_state)

                # # 神经网络
                # mlp = MLPClassifier(hidden_layer_sizes=(10,), activation="relu", alpha=0.0001, learning_rate_init=0.03,
                #                     max_iter=200)
                # print('神经网络:\t\t\t', end='')
                # result(mlp, X, y,random_state)

                # # lightgbm
                # gbm = lgb.LGBMClassifier()
                # print('lightgbm:\t\t', end='')
                # result(gbm, X, y, random_state)
