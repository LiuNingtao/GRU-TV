import torch
import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt

import numpy as np
import os
from tqdm import tqdm
from data.dataset_physionet import Physionet2012, PhysoinetDatasetCache, PhysoinetDatassetResample
import pandas as pd
import sys
import shutil
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
now_time = datetime.now()

from sklearn.metrics import roc_auc_score, mean_absolute_error

sample_rate=50
result_path = '/path/to/result'
TASK_NAME = ''
result_path = os.path.join(result_path, TASK_NAME)
os.makedirs(result_path, exist_ok=True)
shutil.copy(os.path.join(os.path.abspath(__file__)), 
            os.path.join(result_path, os.path.split(__file__)[1]))

val_log = {'epoch': [], 'auc_MOR': [], 'auc_LoS': [], 'auc_CAR': [], 'auc_SUR': [], 'auc_macro': [], 'auc_micro': [], 'mae': []}
test_log = {'epoch': [], 'auc_MOR': [], 'auc_LoS': [], 'auc_CAR': [], 'auc_SUR': [], 'auc_macro': [], 'auc_micro': [], 'mae': []}

def fit(model, train_x, train_y, val_x, val_y, test_x, test_y):
  model.fit(train_x, train_y)

  val_predict = model.predict(val_x)
  val_predict_score = model.predict_proba(val_x)
  auc_val, mae_val = get_performance(val_y, val_predict)
  np.save(os.path.join(result_path, 'pred_best_VAL.npy'), val_predict)
  np.save(os.path.join(result_path, 'pred_best_VAL_score.npy'), val_predict_score)
  np.save(os.path.join(result_path, 'label_best_VAL.npy'), val_y)
  val_log['epoch'].append(0)
  for task_name in task_names[:4]:
    val_log[f'auc_{task_name}'].append(auc_val[task_name])
  val_log['auc_macro'].append(auc_val['macro'])
  val_log['auc_micro'].append(auc_val['micro'])
  val_log['mae'].append(mae_val)
  
  test_predict = model.predict(test_x)
  test_predict_score = model.predict(test_x)
  auc_test, mae_test = get_performance(test_y, test_predict)
  np.save(os.path.join(result_path, 'pred_best_TEST.npy'), test_predict)
  np.save(os.path.join(result_path, 'pred_best_TEST_score.npy'), test_predict_score)
  np.save(os.path.join(result_path, 'label_best_TEST.npy'), test_y)
  test_log['epoch'].append(0)
  for task_name in task_names[:4]:
    test_log[f'auc_{task_name}'].append(auc_test[task_name])
  test_log['auc_macro'].append(auc_test['macro'])
  test_log['auc_micro'].append(auc_test['micro'])
  test_log['mae'].append(mae_test)
  
  pd.DataFrame(test_log).to_csv(os.path.join(result_path, 'test_log.csv'), index=False)
  pd.DataFrame(val_log).to_csv(os.path.join(result_path, 'val_log.csv'), index=False)

def get_performance(labels, pred_cls, pred_reg=None):
    if len(labels.shape) > 2:
      labels = np.squeeze(labels)
    labels_cls = labels[:, :4]
    labels_cls[labels_cls<0.5] = 0
    labels_cls[labels_cls!=0] = 1

    auc_dict = {}
    for idx, cls_name in enumerate(task_names[:4]):
       auc_dict[cls_name] = roc_auc_score(labels_cls[:, idx], pred_cls[:, idx])
    auc_dict['macro'] = roc_auc_score(labels_cls, pred_cls, average='macro')
    auc_dict['micro'] = roc_auc_score(labels_cls, pred_cls, average='micro')
    
    if output_size > 1 and 'LoS_REG' in task_names:
      # pred_reg = np.round(pred_reg)
      labels_reg = labels[:, 4]
      mae = mean_absolute_error(labels_reg, pred_reg)
    else:
      mae = -1

    return auc_dict, mae


def data_set(dataset_dir, data_root):
  set_dict = {}
  for mode in ('train', 'val', 'test'):
    list_df = pd.read_csv(os.path.join(dataset_dir, f'{mode}_listfile.csv'))
    set_dict[mode] = Physionet2012(list_df, data_root)
  return set_dict

if __name__ == '__main__':    
  input_size = 33 
  hidden_size = 64

  task_names = ['MOR', 'LoS', 'CAR', 'SUR']
  output_size = len(task_names)
  
  learning_rate = 0.001
  learning_rate_decay = 5
  n_epochs = 100

  dataset_dir = '/path/to/data_set_dir'
  data_dir = f'/path/to/data_dir/{sample_rate}'

  set_dict = data_set(dataset_dir, data_dir)
  train_set, dev_set, test_set = set_dict['train'], set_dict['val'], set_dict['test']
  train_x, train_y = train_set.ML_features, train_set.labels
  val_x, val_y = dev_set.ML_features, dev_set.labels
  test_x, test_y = test_set.ML_features, test_set.labels

  # model = RandomForestClassifier(max_depth=3, random_state=0)
  # model = LogisticRegression(multi_class='multinomial', solver="newton-cg", max_iter=20)
  model = SVC(probability=True)
  model = MultiOutputClassifier(estimator=model, n_jobs=output_size)

  train_x = np.reshape(train_x, (train_x.shape[0], -1))
  train_x = np.nan_to_num(train_x)
  val_x = np.reshape(val_x, (val_x.shape[0], -1))
  val_x = np.nan_to_num(val_x)
  test_x = np.reshape(test_x, (test_x.shape[0], -1))
  test_x = np.nan_to_num(test_x)


  train_y = train_y[:, :4]
  val_y = val_y[:, :4]
  test_y = test_y[:, :4]
  fit(model, train_x, train_y, val_x, val_y, test_x, test_y)

  