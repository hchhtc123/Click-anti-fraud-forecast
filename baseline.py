import sys 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from catboost import CatBoostClassifier
import gc
import re
import warnings
warnings.filterwarnings("ignore")

# 定义所需的特征工程处理函数：
# 数据特征处理和类别转换
def data_preprocessing(train, test1):
    df1 = train.drop(['Unnamed: 0'], axis=1)
    df2 = test1.drop(['Unnamed: 0'], axis=1)
    df2["label"] = -1
    li = []
    for df_index in [df1,df2]:
        # 类型转换
        for col in ["android_id", "apptype", "carrier", "ntt", "media_id", "cus_type", "package", 'fea1_hash', "location"]:
            df_index[col] = df_index[col].astype("object")
        for col in ["fea_hash"]:
            df_index[col] = df_index[col].map(lambda x: 0 if len(str(x)) > 16 else int(x))
        for col in ["dev_height", "dev_ppi", "dev_width", "fea_hash", "label"]:
            df_index[col] = df_index[col].astype("int64")
        # 时间特征处理和转换
        df_index["truetime"] = pd.to_datetime(df_index['timestamp'], unit='ms', origin=pd.Timestamp('1970-01-01'))
        df_index["day"] = df_index.truetime.dt.day
        df_index["hour"] = df_index.truetime.dt.hour
        df_index["minute"] = df_index.truetime.dt.minute
        df_index.set_index("sid", drop=True, inplace=True)
        df_index.dev_height[df_index.dev_height == 0] = None
        df_index.dev_width[df_index.dev_width == 0] = None
        df_index.dev_ppi[df_index.dev_ppi == 0] = None
        li.append(df_index)
    df2["label"] = None
    return li

# 类别预处理
def process_category(df1, df2, col):
    le = preprocessing.LabelEncoder()  # 特征编码
    df1[col] = le.fit_transform(df1[col])
    df1[col] = df1[col].astype("object")
    df2[col] = le.transform(df2[col])
    df2[col] = df2[col].astype("object")
    return df1, df2


def dict_category(df1, df2, col, dict1):
    print(col, dict1)
    df1[col] = df1[col].map(dict1)
    df1[col] = df1[col].astype("object")
    df2[col] = df2[col].map(dict1)
    df2[col] = df2[col].astype("object")
    return df1, df2

def filter_value(df1, df2, col, top, other=-1):
    set1 = set(df1[col].value_counts().head(top).index)
    def process_temp(x):
        if x in set1:
            return x
        else:
            return other
    df1[col] = df1[col].apply(process_temp)
    df2[col] = df2[col].apply(process_temp)
    return df1, df2

def special_category(df1, df2, col):
    if col == "apptype":
        df1, df2 = filter_value(df1, df2, col, 75, -1)
    if col == "media_id":
        df1, df2 = filter_value(df1, df2, col, 200, -1)
    if col == "version":
        df2[col] = df2[col].replace("20", "0").replace("21", "0")
    if col == "lan":
        def foreign_lan(x):
            set23 = {'zh-CN', 'zh', 'cn', 'zh_CN', 'Zh-CN', 'zh-cn', 'ZH', 'CN', 'zh_CN_#Hans'}
            if x in set23:
                return 0
            elif x == "unk":
                return 2
            else:
                return 1
        df1["vpn"] = df1["lan"].apply(foreign_lan)
        df2["vpn"] = df2["lan"].apply(foreign_lan)
        set12 = {'zh-CN', 'zh', 'cn', 'zh_CN', 'Zh-CN', 'zh-cn', 'ZH', 'CN', 'tw', 'en', 'zh_CN_#Hans', 'ko'}
        def process_lan(x):
            if x in set12:
                return x
            else:
                return "unk"
        df1[col] = df1[col].apply(process_lan)
        df2[col] = df2[col].apply(process_lan)
    if col == "package":
        df1, df2 = filter_value(df1, df2, col, 800, -1)
    if col == "fea1_hash":
        df1, df2 = filter_value(df1, df2, col, 850, -1)
    if col == "fea_hash":
        df1, df2 = filter_value(df1, df2, col, 850, -1)
    df1, df2 = process_category(df1, df2, col)
    return df1, df2


def feature(df1, df2):
    def divided(x):
        if x % 40 == 0:
            return 2
        elif not x:
            return 1
        else:
            return 0

    # 特征构造
    df1["160_height"] = df1.dev_height.apply(divided)
    df2["160_height"] = df2.dev_height.apply(divided)
    df1["160_width"] = df1.dev_width.apply(divided)
    df2["160_width"] = df2.dev_width.apply(divided)
    df1["160_ppi"] = df1.final_ppi.apply(divided)
    df2["160_ppi"] = df2.final_ppi.apply(divided)
    df1["hw_ratio"] = df1.dev_height / df1.dev_width
    df2["hw_ratio"] = df2.dev_height / df2.dev_width
    df1["hw_matrix"] = df1.dev_height * df1.dev_width
    df2["hw_matrix"] = df2.dev_height * df2.dev_width
    df1["inch"] = (df1.dev_height ** 2 + df1.dev_width ** 2) ** 0.5 / df1.final_ppi
    df2["inch"] = (df2.dev_height ** 2 + df2.dev_width ** 2) ** 0.5 / df2.final_ppi
    return df1, df2


def rf_cast(df1, df2):
    c1 = df1.dev_width.notnull()
    c2 = df1.dev_height.notnull()
    c3 = df1.dev_ppi.isna()
    c4 = df1.dev_ppi.notnull()
    df1["mynull1"] = c1 & c2 & c3
    df1["mynull2"] = c1 & c2 & c4

    predict = df1[
        ["apptype", "carrier", "dev_height", "dev_ppi", "dev_width", "media_id", "ntt", "mynull1", "mynull2"]]

    df_notnans = predict[predict.mynull2 == True]

    # 75训练25预测
    X_train, X_test, y_train, y_test = train_test_split(
        df_notnans[["apptype", "carrier", "dev_height", "dev_width", "media_id", "ntt"]], df_notnans["dev_ppi"],
        train_size=0.75, random_state=6)
    
    # 随机森林分类
    regr_multirf = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=0, n_jobs=-1)
    regr_multirf.fit(X_train, y_train)
    score = regr_multirf.score(X_test, y_test)
    print("prediction score is {:.2f}%".format(score * 100))
    df_nans = predict[predict.mynull1 == True].copy()
    df_nans["dev_ppi_pred"] = regr_multirf.predict(
        df_nans[["apptype", "carrier", "dev_height", "dev_width", "media_id", "ntt"]])
    df1 = pd.merge(df1, df_nans[["dev_ppi_pred"]], on="sid", how="left")
    c1 = df2.dev_width.notnull()
    c2 = df2.dev_height.notnull()
    c3 = df2.dev_ppi.isna()
    c4 = df2.dev_ppi.notnull()
    df2["mynull1"] = c1 & c2 & c3
    df2["mynull2"] = c1 & c2 & c4
    predict_test = df2[
        ["apptype", "carrier", "dev_height", "dev_ppi", "dev_width", "media_id", "ntt", "mynull1", "mynull2"]]
    df_nans = predict_test[predict_test.mynull1 == True].copy()
    df_nans["dev_ppi_pred"] = regr_multirf.predict(
        df_nans[["apptype", "carrier", "dev_height", "dev_width", "media_id", "ntt"]])
    df2 = pd.merge(df2, df_nans[["dev_ppi_pred"]], on="sid", how="left")

    def recol_ppi(df):
        a = df.dev_ppi.fillna(0).values
        b = df.dev_ppi_pred.fillna(0).values
        c = []
        for i in range(len(a)):
            c.append(max(a[i], b[i]))
        c = np.array(c)
        df["final_ppi"] = c
        df["final_ppi"][df["final_ppi"] == 0] = None
        return df

    df1 = recol_ppi(df1)
    df2 = recol_ppi(df2)
    gc.collect()
    return df1, df2

def process_osv(df1, df2):
    def process_osv1(x):
        x = str(x)
        if not x:
            return -1
        elif x.startswith("Android"):
            x = str(re.findall("\d{1}\.*\d*\.*\d*", x)[0])
            return x
        elif x.isdigit():
            return x
        else:
            try:
                x = str(re.findall("\d{1}\.\d\.*\d*", x)[0])
                return x
            except:
                return 0

    df1.osv = df1.osv.apply(process_osv1)
    df2.osv = df2.osv.apply(process_osv1)
    set3 = set(df1["osv"].value_counts().head(70).index)

    def process_osv2(x):
        if x in set3:
            return x
        else:
            return 0

    df1["osv"] = df1["osv"].apply(process_osv2)
    df2["osv"] = df2["osv"].apply(process_osv2)

    le8 = preprocessing.LabelEncoder()
    df1.osv = le8.fit_transform(df1.osv.astype("str"))
    df1["osv"] = df1["osv"].astype("object")

    df2.osv = le8.transform(df2.osv.astype("str"))
    df2["osv"] = df2["osv"].astype("object")
    return df1, df2

# 定义catboost训练和预测函数
def catboost_train_predict(train_path,test_path):
    feature_train = pd.read_pickle(train_path)  # 训练
    feature_test = pd.read_pickle(test_path)    # 测试
    # 特征类别转换
    for col in ["dev_height", "dev_width", "hw_ratio", "hw_matrix", "inch", "lan"]:
        if col in feature_train.columns:
            feature_train[col] = feature_train[col].astype("float64")
            feature_test[col] = feature_test[col].astype("float64")
    
    # 所使用的特征
    cate_feature = ['apptype', 'carrier', 'media_id', 'os', 'osv', 'package', 'version', 'location', 'cus_type',
                    "fea1_hash", "fea_hash", "ntt", "os", 'fea1_hash_ntt_combine', 'fea_hash_carrier_combine',
                    'cus_type_osv_combine', 'fea1_hash_apptype_combine', 'fea_hash_media_id_combine',
                    'cus_type_version_combine', 'apptype_ntt_combine', 'media_id_carrier_combine',
                    'version_osv_combine', 'package_lan_combine', 'lan']

    y_col = 'label'
    x_col = ['apptype', 'carrier', 'dev_height',
             'dev_width', 'lan', 'media_id', 'ntt', 'osv', 'package',
             'timestamp', 'version', 'fea_hash', 'location', 'fea1_hash', 'cus_type',
             'hour', 'minute',
             '160_height',
             'hw_ratio', 'hw_matrix', 'inch']

    cate_feature = [x for x in cate_feature if x in x_col]
    for item in cate_feature:
        if item in ['fea1_hash_ntt_combine', 'fea_hash_carrier_combine', 'cus_type_osv_combine',
                    'fea1_hash_apptype_combine', 'fea_hash_media_id_combine', 'cus_type_version_combine',
                    'apptype_ntt_combine', 'media_id_carrier_combine', 'version_osv_combine', 'package_lan_combine']:
            set4 = set(feature_train[item].value_counts().head(300).index)

            def process_fea_hash(x):
                if x in set4:
                    return x
                else:
                    return -1

            feature_train[item] = feature_train[item].apply(process_fea_hash).astype("str")
            feature_test[item] = feature_test[item].apply(process_fea_hash).astype("str")
        le = preprocessing.LabelEncoder()
        feature_train[item] = le.fit_transform(feature_train[item])
        feature_test[item] = le.transform(feature_test[item])

    df_prediction = feature_test[x_col]
    df_prediction['label'] = 0

    # 树模型参数设置：通过控制变量的方式进行动态调整
    model = CatBoostClassifier(
        loss_function="Logloss",    # 分类任务常用损失函数
        eval_metric="Accuracy",     # 表示用于过度拟合检测和最佳模型选择的度量标准；
        learning_rate=0.08,         # 表示学习率
        iterations=10000,
        random_seed=42,             # 设置随机种子进行固定
        od_type="Iter",
        metric_period=20,           # 与交叉验证folds数匹配
        max_depth = 8,              # 表示树模型最大深度
        early_stopping_rounds=500,  # 早停步数
        use_best_model=True,
        # task_type="GPU",          # 数据量较小，GPU加速效果不明显
        bagging_temperature=0.9,
        leaf_estimation_method="Newton",
    )

    li_f = []
    df_importance_list = []
    n = 20  # 设置20折交叉验证
    kfold = KFold(n_splits=n, shuffle=True, random_state=42)
    # weight = [0.1, 0.11, 0.1, 0.11, 0.11, 0.11, 0.05, 0.11, 0.1, 0.1]
    # assert sum(weight) == 1 and len(weight) == n
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(feature_train[x_col], feature_train[y_col])):
        X_train = feature_train.iloc[trn_idx][x_col]
        Y_train = feature_train.iloc[trn_idx][y_col]

        X_val = feature_train.iloc[val_idx][x_col]
        Y_val = feature_train.iloc[val_idx][y_col]

        print('\nFold_{} Training ================================\n'.format(fold_id + 1))
        cat_model = model.fit(
            X_train,
            Y_train,
            cat_features=cate_feature,  # 特征
            # # eval_names=['train', 'valid'],
            eval_set=(X_val, Y_val),
            verbose=100,
        )

        pred_val = cat_model.predict_proba(X_val, thread_count=-1)[:, 1]
        df_oof = feature_train.iloc[val_idx].copy()
        df_oof['pred'] = pred_val
        li_f.append(df_oof)

        pred_test = cat_model.predict_proba(feature_test[x_col], thread_count=-1)[:, 1]
        df_prediction['label'] += pred_test / n

        df_importance = pd.DataFrame({
            'column': x_col,
            'importance': cat_model.feature_importances_,
        })
        df_importance_list.append(df_importance)
    return df_prediction, li_f, feature_train, feature_test


# 定义预测结果保存函数
def save(file_path, pred, df1, df2, threshold=0.5):
    a = pd.DataFrame(pred.index)
    a['label'] = pred["label"].values

    # 由于输出结果为0或1，故需要对分数结果进行后处理操作：大于threshold的为1，小于或等于threshold则为0。 threshold为设定的阈值
    a.label = a.label.apply(lambda x: 1 if x > threshold else 0)
    user_label = pd.DataFrame()

    user_label["uid"] = df1.android_id.values
    user_label["ntt"] = df1.ntt.values
    temp = pd.DataFrame(df1.groupby(["android_id", "ntt"]).label.mean())
    temp = temp.reset_index()
    temp.rename(columns={"android_id": "uid", "label": "label_prior"}, inplace=True)
    user_label = pd.merge(user_label, temp, on=["uid", "ntt"], how="left")
    user_label.drop_duplicates(inplace=True)
    a["uid"] = df2.android_id.values
    a["ntt"] = df2.ntt.values
    a = pd.merge(a, user_label, how="left", on=["uid", "ntt"])

    def post(label, prior):
        n = len(label)
        count = 0
        for i in range(n):
            if 0 <= prior[i] <= 0.1 and label[i] == 1:
                label[i] = 0
                count += 1
            elif 0.9 <= prior[i] <= 1 and label[i] == 0:
                label[i] = 1
                count += 1
            else:
                pass
        print(count)
        return label.values

    a.label = post(a.label, a.label_prior)
    a = a[["sid", "label"]]
    a.to_csv(file_path, index=False)
    return a

if __name__ == '__main__':
    train = pd.read_csv('./train.csv')  # 训练数据
    test = pd.read_csv('./test.csv')    # 测试数据

    # 特征工程处理:
    df = data_preprocessing(train,test)
    df1 = df[0]
    df2 = df[1]
    for col in ["location", "os", "ntt", "cus_type"]:
        df1, df2 = process_category(df1, df2, col)
    for col, dict1 in zip(["carrier"], [{0.0: 0, 46000.0: 1, 46001.0: 2, 46003.0: 3, -1.0: -1}]):
        df1, df2 = dict_category(df1, df2, col, dict1)
    for col in ["apptype", "media_id", "version", "lan", "package", "fea1_hash", "fea_hash"]:
        df1, df2 = special_category(df1, df2, col)
    df1, df2 = process_osv(df1, df2)
    df1, df2 = rf_cast(df1, df2)
    df1, df2 = feature(df1, df2)
    df1.to_pickle("./train.jlz")
    df2.to_pickle("./test.jlz")

    # 进行树模型的训练和预测
    df_prediction, li_f, feature_train, feature_test  = catboost_train_predict("./train.jlz","./test.jlz")  

    # 保存预测结果文件
    filename = './submission.csv'  # 设置保存结果文件名
    # 可以通过修改threshold的值来修改阈值
    save(filename, df_prediction, feature_train, feature_test,threshold=0.5)
    print("success")
    
    # 查看结果文件格式是否符合要求：sid,label
    result = pd.read_csv('./submission.csv')
    print(result.head())