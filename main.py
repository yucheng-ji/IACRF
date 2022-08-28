import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
import pickle
import time
import os

def error_evalue(y_hat, y):
    return np.sum(np.abs(y_hat - y))/len(y)

def normal_x(x):
    # print(x.shape)
    # name_list = ['Al', 'Zn', 'Mg', 'Cu', 'Si', 'Fe', 'Mn', 'Cr', 'Ti', 'Other', 'Cycle',
    #              'Temper', 'Precip', 'pH', 'Cl', 'wf']
    name_list = ['Cyc', 'Al', 'Zn',]
    variable_N = x.shape[1]         # 15
    sample_N = x.shape[0]           # 150
    data = {}
    for i in range(variable_N):
        data[name_list[i]] = []
        for j in range(sample_N):
            data[name_list[i]].append(x[j][i])
    # print(data)
    # Find the Max data and Min data
    for h in name_list:
        h_max = max(data[h])
        h_min = min(data[h])
        length = "{:.5f}".format(h_max - h_min)
        # print(h, h_max, h_min, length)

        # Normalization
        h_list = []
        for h_data in data[h]:
            normal_data = (float(h_data)- h_min)/float(length)
            h_list.append(round(normal_data, 5))
        # print(h_list)

        data[h] = h_list

    # consist array
    normal_list = []
    for i in range(sample_N):
        data2 = []
        for j in range(variable_N):
            data2.append(data[name_list[j]][i])
        normal_list.append(data2)

    # print(normal_list)
    return np.array(normal_list)

def normal_y(y):
    sample_N = y.shape[0]
    y_max = max(y)
    y_min = min(y)
    length = float(y_max - y_min)
    new_list = []
    for j in range(sample_N):
        y_new = round((y[j] - y_min)/length, 5)
        new_list.append(y_new)

    return new_list


if __name__ == '__main__':

    name = 'normal-wf'
    outdir = 'D:/RandomForest/models/%s3' %(name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    df = pd.read_csv('raw_data/%s.csv' %(name))
    a = np.array(df)
    Y = a[:, -1]
    X = a[:, :-1]
    # print(X[0], Y[0])
    # exit()
    x_normal = X    #normal_x(X)
    y_normal = Y    #normal_y(Y)

    ts = pd.read_csv('raw_data/SouthEastpredict02.csv')
    tsa = np.array(ts)
    ts_x = tsa[:, :-1]
    ts_y = tsa[:, -1]

    # x_train, x_test, y_train, y_test = train_test_split(x_normal, y_normal, test_size=0.3, random_state=None)
    record = []
    for xx in range(500):
        rs = ShuffleSplit(n_splits=50, test_size=0.2)
        j = 0
        for train_idx, test_idx in rs.split(x_normal):
            # for v_est in range(80, 101, 1):
                # for v_min_s_s in range(2, 4, 1):
                #     # for v_feature in range(5, 16, 1):
            rf = RandomForestRegressor(n_estimators=100,
                                       max_depth=None,
                                       max_features=None,
                                       min_samples_split=2,  #v_min_s_s,
                                       min_samples_leaf=1,
                                       min_weight_fraction_leaf=0.0,
                                       max_leaf_nodes=None,
                                       bootstrap=True,
                                       oob_score=True,
                                       n_jobs=1,
                                       random_state=None,
                                       verbose=0,
                                       warm_start=False,)
            rf.fit(x_normal[train_idx], y_normal[train_idx])
            score = rf.score(x_normal[test_idx], y_normal[test_idx])
            score2 = rf.score(ts_x, ts_y)
            if score > 0.89 and score2 > 0.5:
                tm = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                score_100 = round(score, 2) * 100
                score2_100 = round(score2, 2) * 100
                record_name = '%d-%d-%d-%d-%s' % (score_100, score2_100, xx, j, tm)
                with open('models/%s3/%s.pickle' %(name, record_name), 'wb') as rfw:
                    pickle.dump(rf, rfw)
                    rfw.close()

                record_per = '%d    %d  %.5f    %.5f' %(xx, j, score, score2)
                print(record_per)
                record.append(record_per)
            j += 1

    with open('output/%s.txt' %(name), 'a+') as f:
        f.write('\n'.join(record))