import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def creat_dataset(read_data, model, image_path):
    window_size = 35
    # データの読み込み、plot関数
    def create_values(read_data):
        # データの読み込み
        data = pd.read_csv(read_data)

        # データのrssi、metreをそれぞれx、yとして定義
        x = data.rssi

        return x

    list_X = create_values(read_data)

    # Xを正の値に直して正規化
    min_ = 40
    max_ = 120
    list_X = (abs(list_X) - min_) / (max_ - min_)

    list_rssi = []  # rssiのlist

    # list_Xをwindow_sizeごとにサンプリング
    for i in range(0, len(list_X) - window_size, 1):
        list_rssi.append(list_X[i:i + window_size].values)

    data = np.array(list_rssi)

    data = data.reshape(len(data), window_size, 1, 1)

    print("start predict")
    pred = model.predict(data)

    pred = np.argmax(pred, axis=1)

    pre = pd.DataFrame({'label': np.array(pred)})
    label = np.identity(4)[pre.label]

    bar = np.argmax(label, axis=1)
    plt.figure(figsize=(20, 10))
    for t in range(4):
        plt.bar(range(len(bar)), pre.label == t, width=1)

    plt.savefig(image_path)

    print("finish predict")

    return 'aa'