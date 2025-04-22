import numpy as np
import torch
import torch.utils.data as utils


from sklearn.metrics import roc_curve, auc, f1_score, precision_score
from itertools import cycle
import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from comparison_DL.CL_ImpPreNet.model import Model


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
series_length_limit = 100

raw_PhysioNet_path = '/path/to/raw/PhysioNet2012'
result_path = '/path/to/result'
TASK_NAME = ''
result_path = os.path.join(result_path, TASK_NAME)
if not os.path.exists(result_path):
    os.makedirs(result_path)

task_list = ["Mean", "Mortality", "Length of Stay", "Cardiac", "Surgery"]
performance_dict_val = dict(zip(task_list, [[] for i in task_list]))
performance_dict_val['Epoch'] = []
performance_dict_test = dict(zip(task_list, [[] for i in task_list]))
performance_dict_test['Epoch'] = []

loss_train_list = []
loss_val_list = []

class Physionet2012(utils.Dataset):
    def __init__(self, list_df, data_root):
        data_list = []
        for idx, row in list_df.iterrows():
            origin_file = os.path.join(raw_PhysioNet_path, str(int(row['RecordID']))+'.txt')
            base_info_lines = open(origin_file, 'r').readlines()[:7]
            
            age = 0
            gender = 0
            for line in base_info_lines:
                if 'Age' in line:
                    age = int(line.split(',')[2].replace('\n', ''))
                if 'Gender' in line:
                    gender = int(line.split(',')[2].replace('\n', ''))

            record_id = str(int(row['RecordID']))
            self.data_root = data_root
            series = np.load(os.path.join(self.data_root, 'timeseries', record_id+'.npy'))
            if len(series.shape) != 2 or len(series) <= 0:
                print(series)
            mask = np.load(os.path.join(self.data_root, 'mask', record_id+'.npy'))
            delta = np.load(os.path.join(self.data_root, 'delta', record_id+'.npy'))
            dt = np.load(os.path.join(self.data_root, 'dt', record_id+'.npy'))

            series = torch.tensor(series, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)

            delta = torch.tensor(delta, dtype=torch.float32)
            dt = torch.tensor(dt, dtype=torch.float32)
            # features = torch.tensor(features, dtype=torch.float32).view(-1)

            los = row['Length_of_stay']
            mor = row['In-hospital_death']
            ICU_type = row['ICUType']
            los_cls = 1 if los > 3 else 0
            car = 1 if ICU_type == 2 else 0
            sur = 1 if ICU_type == 4 else 0
            y = torch.tensor((mor, los_cls, car, sur, los), dtype=torch.float32)
            data = {
                'series': series,
                'mask': mask,
                'delta': delta,
                'dt': dt,
                'base': torch.tensor((age, gender), dtype=torch.float32),
                'label': y,
            }
            data_list.append(data)   
        # los_0_list = [data for data in data_list if int(data['label'][0]) == 0]
        # los_1_list = [data for data in data_list if int(data['label'][0]) == 1]
        # if len(los_0_list) > len(los_1_list):
        #     los_0_list = los_0_list[:len(los_1_list)]
        # else:
        #     los_1_list = los_1_list[:len(los_0_list)]
        # data_list = los_0_list + los_1_list
        self.data_list = data_list
    
    def feature_label(self):
        feature_list = [d['feature'] for d in self.data_list]
        y_list = np.array(d['label'] for d in self.data_list)
        return np.stack(feature_list), np.stack(y_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data


model = Model(
    emb_f_size=1,
    input_v_size=33,
    emb_v_size=3,
    proj1_e_size=99,
    proj2_e_size=28,
    base_size=2,
    base_emb_size=1,
    hid1_size=66,
    hid2_size=55,
    phi=0.56,
    drop_p=0.1, 
    task='Prediction'
    )

model.to(device)
f_list = ["Cholesterol", "TroponinI", "TroponinT", "Albumin", "Alkaline phosphatase", "Aspartate transaminase", "Aspartate transaminase",
          "Bilirubin", "Lactate","SaO2", "White blood cell count", "Glucose", "Na", "Mg", "HCO3", "Blood urea nitrogen", 
          "Creatinine", "Platelets", "K", "HCT", "Partial pressure of arterial O2", 
        "partial pressure of arterial CO2", "pH", "Fractional inspired O2", "RespRate", 
        "Glasgow Coma Score", "Temperature ", "Weight", "Urine output", "Invasive mean arterial blood pressure", "Invasive diastolic arterial blood pressure", 
        "Invasive systolic arterial blood pressure", "Heart rate"]
def str_lower(str_list):
    return [str_list[i].lower() for i in range(len(str_list))]

f_list = [str_lower(f_list[i].split()) for i in range(len(f_list))]
lens = [len(f_list[i]) for i in range(len(f_list))]
name_dic_feature = list(set((np.concatenate(f_list).flat)))
f_idx_list = []
for i in range(len(f_list)):
    tmp_idx_list = []
    for j in range(len(f_list[i])):
        idx = name_dic_feature.index(f_list[i][j]) + 1
        tmp_idx_list.append(idx)
    if len(tmp_idx_list) < max(lens):
        tmp_idx_list = tmp_idx_list + [0] * (max(lens) - len(tmp_idx_list))
    f_idx_list.append(tmp_idx_list)
f_idx = torch.LongTensor(f_idx_list).to(device)



def fit(model, criterion, learning_rate,\
        train_dataloader, dev_dataloader, test_dataloader,\
        learning_rate_decay, n_epochs):
  epoch_losses = []
  best_val = -1
  best_test = -1
  no_increase = 0
  for epoch in range(0, n_epochs):   
    learning_rate = learning_rate * 0.5**int(epoch/learning_rate_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
      
    # train the model
    losses = []
    model.train()

    dh_sim_list = []
    for batch_data in tqdm(train_dataloader):
      series, mask, delta, base = batch_data['series'], batch_data['mask'], batch_data['delta'], batch_data['base']
      series_length = series.size(1)

      series, mask, delta, base =  series.cuda(), mask.cuda(), delta.cuda(), base.cuda()
      train_label = batch_data['label']
      dt = batch_data['dt']
      dt = torch.squeeze(dt)
      time = [torch.sum(dt[:i]) for i in range(len(dt))]
      time = torch.tensor(time, dtype=torch.float32).cuda()
      time = time.unsqueeze(0)
      optimizer.zero_grad() 
      # train_data = torch.squeeze(train_data)
      train_label = torch.squeeze(train_label)
      train_label = train_label.cuda()
      train_label = train_label[:4]
      train_label = torch.unsqueeze(train_label, 0)
      if series_length < series_length_limit:
         series = torch.cat((series, torch.zeros((series.size(0), series_length_limit - series_length, series.size(2))).cuda()), dim=1)
         time = torch.cat((time, torch.zeros((time.size(0), series_length_limit - series_length)).cuda()), dim=1)
      else:
         series = series[:, :series_length_limit, :]
         time = time[:, :series_length_limit]
      y_pred, _ = model(f_idx, series, time, base, None)

      loss = criterion(y_pred, train_label)

      losses.append(loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    train_loss = np.mean(losses)
    loss_train_list.append(float(train_loss))
    torch.save(model.state_dict(), os.path.join(result_path, 'last.pkl'))
    print('='*35+str(epoch)+'='*35)
    print('DH SIM TRAIN: {}'.format(np.mean(dh_sim_list)))
    need_test = False
    auc_mean = test_model(epoch, model, dev_dataloader, 'VAL', best_val)

    if auc_mean > best_val:
      best_val = auc_mean
      no_increase = 0
      need_test = True
      print('='*35+'IMPROVED'+'='*35)
    else:
      no_increase += 1

    if no_increase >= 3 and epoch > 25:
      break


    if need_test:
        auc_mean = test_model(epoch, model, test_dataloader, 'TEST', best_test)
        if auc_mean > best_test:
          best_test = auc_mean
          print('='*35+'NEW BEST'+'='*35)

def test_model(epoch, model, dev_dataloader, task, best_per):
    losses = []
    pred, label = list(), list()
    model.eval()
    dh_sim_list = []
    for batch_data in tqdm(dev_dataloader):
      series, mask, delta, base = batch_data['series'], batch_data['mask'], batch_data['delta'], batch_data['base']
      series_length = series.size(1)

      series, mask, delta, base =  series.cuda(), mask.cuda(), delta.cuda(), base.cuda()
      dev_label = batch_data['label']
      dt = batch_data['dt']
      dt = torch.squeeze(dt)
      time = [torch.sum(dt[:i]) for i in range(len(dt))]
      time = torch.tensor(time, dtype=torch.float32).cuda()
      time = time.unsqueeze(0)
      # train_data = torch.squeeze(train_data)
      dev_label = torch.squeeze(dev_label)
      dev_label = dev_label.cuda()
      dev_label = dev_label[:4]
      dev_label = torch.unsqueeze(dev_label, 0)
      dt = torch.squeeze(dt) 
      if series_length < series_length_limit:
         series = torch.cat((series, torch.zeros((series.size(0), series_length_limit - series_length, series.size(2))).cuda()), dim=1)
         time = torch.cat((time, torch.zeros((time.size(0), series_length_limit - series_length)).cuda()), dim=1)
      else:
         series = series[:, :series_length_limit, :]
         time = time[:, :series_length_limit]
      y_pred, _ = model(f_idx, series, time, base, None)
      
      #pred.append(torch.argmax(y_pred,dim=0).item())
      pred.append(y_pred.cpu().detach().numpy().tolist())
      label.append(dev_label.cpu().detach().numpy().tolist())
      #loss = criterion(y_pred.view(1,-1), train_label.long())AUC
      loss = criterion(y_pred, dev_label)
      # acc.append(
      #     accuracy_score([dev_label.item()], [(y_pred.item()>0.5)+0])
      # )
      losses.append(loss.item())
          
    # dev_acc = np.mean(acc)
    dev_loss = np.mean(losses)
    if task == 'VAL':
      loss_val_list.append(float(dev_loss))
      loss_dict = {
        'train_loss': loss_train_list,
        'val_loss': loss_val_list
      }
      pd.DataFrame(loss_dict).to_csv(os.path.join(result_path, 'loss_curve.csv'))
    
    pred = np.asarray(pred)
    label = np.asarray(label)
    pred = np.squeeze(pred)
    label = np.squeeze(label)

    auc_mean, roc_auc = get_performance(predicts=pred, labels=label, best_per=best_per, save_path=result_path, task=task)
    if auc_mean > best_per:
      best_per = auc_mean
      np.save(os.path.join(result_path, 'pred_best_{}.npy'.format(task)) , pred)
      np.save(os.path.join(result_path, 'label_best_{}.npy'.format(task)), label)
      np.save(os.path.join(result_path, 'dh_sim_best_{}.npy'.format(task)), np.array(dh_sim_list))
      torch.save(model.state_dict(), os.path.join(result_path, 'model_best_{}.pkl'.format(task)))
    
    auc_score = roc_auc_score(label, pred)
    auc_score_1 = roc_auc_score(label[:,0], pred[:,0])
    auc_score_2 = roc_auc_score(label[:,1], pred[:,1])
    auc_score_3 = roc_auc_score(label[:,2], pred[:,2])
    auc_score_4 = roc_auc_score(label[:,3], pred[:,3])
    
    print('='*35+task+'='*35)
    print("AUC score: {:.4f}".format(
       auc_score))
    print("Mortality: {:.4f}, Length of Stay: {:.4f}, Cardiac: {:.4f}, Surgery: {:.4f}".format(
        auc_score_1, auc_score_2, auc_score_3, auc_score_4))
    print("Mean: {:.4f}, Mortality: {:.4f}, Length of Stay: {:.4f}, Cardiac: {:.4f}, Surgery: {:.4f}".format(
        auc_mean, roc_auc['class_0'], roc_auc['class_1'], roc_auc['class_2'], roc_auc['class_3']))
    print(f"DH sim: {str(np.mean(dh_sim_list))}")
    if task == 'VAL':
      performance_dict_val['Epoch'].append(epoch)
      performance_dict_val['Mean'].append(auc_mean)
      performance_dict_val['Mortality'].append(auc_score_1)
      performance_dict_val['Length of Stay'].append(auc_score_2)
      performance_dict_val['Cardiac'].append(auc_score_3)
      performance_dict_val['Surgery'].append(auc_score_4)
      pd.DataFrame(performance_dict_val).to_csv(os.path.join(result_path, 'val.csv'), index=False)
    elif task == 'TEST':
      performance_dict_test['Epoch'].append(epoch)
      performance_dict_test['Mean'].append(auc_mean)
      performance_dict_test['Mortality'].append(auc_score_1)
      performance_dict_test['Length of Stay'].append(auc_score_2)
      performance_dict_test['Cardiac'].append(auc_score_3)
      performance_dict_test['Surgery'].append(auc_score_4)
      pd.DataFrame(performance_dict_test).to_csv(os.path.join(result_path, 'test.csv'), index=False)
    return auc_mean


def get_performance(predicts, labels, best_per, save_path, task):
    num_class = len(task_list)
    if not isinstance(predicts, np.ndarray):
        predicts = np.array(predicts)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    predicts = np.squeeze(predicts)
    labels = np.squeeze(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_total = 0

    labels[labels<0.5]=0
    labels[labels!=0]=1
    for i in range(num_class):
        class_name = 'class_{}'.format(str(i))
        fpr[class_name], tpr[class_name], _ = roc_curve(labels[:, i], predicts[:, i])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
        auc_total += roc_auc[class_name]
    
    auc_mean = auc_total / num_class
    roc_auc['macro']= roc_auc_score(labels, predicts, average='macro')
    roc_auc['micro'] = roc_auc_score(labels, predicts, average='micro')

    colors = cycle(['blue', 'red', 'green', 'black'])

    plt.figure(figsize=(40, 25))
    for i, color in zip(fpr.keys(), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                label='class {0} ({1:0.5f})'
                ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC')
    plt.legend(loc="lower right")
    if auc_mean > best_per:
        plt.savefig(os.path.join(save_path, '{}_best.png'.format(task)))
    print('*'*40+task+'*'*40)
    # print('FPR:{}'.format('  '.join([str(round(fpr[x], 2)) for x in fpr.keys()])))
    # print('TPR:{}'.format('  '.join([str(round(tpr[x], 2)) for x in tpr.keys()])))
    print('AUC:{}'.format('  '.join([str(round(roc_auc[x], 2)) for x in roc_auc.keys()])))
    print('AUC_MEAN:{}'.format(auc_mean))
    plt.close()
    return auc_mean, roc_auc

def data_dataloader(dataset_dir, data_root):
  loader_dict = {}
  for mode in ('train', 'val', 'test'):
    list_df = pd.read_csv(os.path.join(dataset_dir, f'{mode}_listfile.csv'))
    loader = utils.DataLoader(Physionet2012(list_df, data_root), batch_size=1)
    loader_dict[mode] = loader
  return loader_dict
    
if __name__ == '__main__':    
  input_size = 33 
  output_size = 4
  sample_rate = 100
  

  criterion = torch.nn.BCELoss()
  
  learning_rate = 0.01
  learning_rate_decay = 5
  n_epochs = 100
  dataset_dir = '/path/to/data_set_dir'
  data_dir = f'/path/to/data_dir/{sample_rate}'

  loader_dict = data_dataloader(dataset_dir, data_dir)
  train_dataloader, dev_dataloader, test_dataloader = loader_dict['train'], loader_dict['val'], loader_dict['test']
  model = model.cuda()
  fit(model, criterion, learning_rate,\
      train_dataloader, dev_dataloader, test_dataloader,\
      learning_rate_decay, n_epochs)
  loss_dict = {
    'train_loss': loss_train_list,
    'val_loss': loss_val_list
  }
  pd.DataFrame(loss_dict).to_csv(os.path.join(result_path, 'loss_curve.csv'))