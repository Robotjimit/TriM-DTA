import numpy as np
import subprocess
from math import sqrt
from sklearn.metrics import average_precision_score
from scipy import stats
import random
import torch
import os
from tqdm import tqdm
import pandas as pd
import datetime
device = torch.device('cuda:0')
logmsg = ""  # 全局变量，存储日志信息
saveDefault = True  # 是否默认保存日志到全局变量和文件


def log(msg, save=None, oneline=False, log_file="log.txt"):
    """
    记录日志信息到控制台，并可选择性地写入文件。

    参数：
    - msg: 日志内容
    - save: 是否保存到文件，默认使用 `saveDefault`
    - oneline: 是否使用单行显示（用于动态更新日志）
    - log_file: 日志文件名（默认 "log.txt"）
    """
    global logmsg
    global saveDefault

    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 规范化时间格式
    log_entry = f"{time_str}: {msg}"

    # 保存到全局变量
    if save is not None:
        if save:
            logmsg += log_entry + "\n"
    elif saveDefault:
        logmsg += log_entry + "\n"

    # 记录到日志文件
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except Exception as e:
        print(f"日志写入失败: {e}")

    # 控制台打印日志
    if oneline:
        print(log_entry, end="\r")
    else:
        print(log_entry)

def calculate_metrics_and_return(Y, P, dataset='kiba'):
    # aupr = get_aupr(Y, P)
    # cindex = get_cindex(Y, P)  # DeepDTA
    cindex = get_ci(Y, P)
    rm2 = get_rm2(Y, P)  # DeepDTA
    mse = get_mse(Y, P)

    print('metrics for ', dataset)
    # print('aupr:', aupr)
    print('cindex:', cindex)

    print('rm2:', rm2)
    print('mse:', mse)
    return cindex,rm2,mse

def train(model, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    loss_fn = torch.nn.MSELoss()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        data = [x.to(device) for x in data]
        optimizer.zero_grad()
        output,y = model(data)
        loss = loss_fn(output.float(),  y.float().to(device)).float()

        loss.backward()
        optimizer.step()
def predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = [x.to(device) for x in data]

            output, y = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def get_ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
atom_dict = {5: 'C',
             6: 'C',
             9: 'O',
             12: 'N',
             15: 'N',
             21: 'F',
             23: 'S',
             25: 'Cl',
             26: 'S',
             28: 'O',
             34: 'Br',
             36: 'P',
             37: 'I',
             39: 'Na',
             40: 'B',
             41: 'Si',
             42: 'Se',
             44: 'K',
             }



def get_aupr(Y, P, threshold=7.0):
    # print(Y.shape,P.shape)
    Y = np.where(Y >= 7.0, 1, 0)
    P = np.where(P >= 7.0, 1, 0)
    aupr = average_precision_score(Y, P)
    return aupr


def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i != j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def calculate_metrics(Y, P, dataset='kiba'):
    # aupr = get_aupr(Y, P)
    cindex = get_cindex(Y, P)  # DeepDTA
    rm2 = get_rm2(Y, P)  # DeepDTA
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)

    print('metrics for ', dataset)
    # print('aupr:', aupr)
    print('cindex:', cindex)

    print('rm2:', rm2)
    print('mse:', mse)
    print('pearson', pearson)
    print('spearman',spearman)

def create_fold_setting_cold(df, fold_seed, frac, entities):
    """create cold-split where given one or multiple columns, it first splits based on
    entities in the columns and then maps all associated data points to the partition

    Args:
            df (pd.DataFrame): dataset dataframe
            fold_seed (int): the random seed
            frac (list): a list of train/valid/test fractions
            entities (Union[str, List[str]]): either a single "cold" entity or a list of
                    "cold" entities on which the split is done

    Returns:
            dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    if entities == 'target_key':
        entities = 'new'
        df['new'] = df['target_key'].astype(str)+'_'+df['target_sequence_start'].astype(str)+'_'+df['target_sequence_end'].astype(str)
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e].drop_duplicates().sample(frac=test_frac,
                                       replace=False,
                                       random_state=fold_seed).values
        for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a "
            "less stringent splitting strategy.")

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e].drop_duplicates().sample(frac=val_frac / (1 - test_frac),
                                              replace=False,
                                              random_state=fold_seed).values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac "
            "or a less stringent splitting strategy.")

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]
    train_indices = df[df.isin(train)].dropna().index
    val_indices = df[df.isin(val)].dropna().index
    test_indices = df[df.isin(test)].dropna().index
    # return {
    #     "train": train.reset_index(drop=True),
    #     "valid": val.reset_index(drop=True),
    #     "test": test.reset_index(drop=True),
    # }
    return {
        "train": train_indices,
        "valid": val_indices,
        "test": test_indices,
    }
