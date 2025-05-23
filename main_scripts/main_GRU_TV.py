import torch
import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt
import torch.nn as nn

import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_score
from itertools import cycle
import os
from tqdm import tqdm
from dataset import PhysoinetDatassetResample
from torch.utils.data.dataloader import DataLoader
import pandas as pd

from sklearn.metrics import roc_auc_score
result_path = r''
TASK_NAME = ''
# rootpath = r'F:\OriginF\Physionet2012'
result_path = os.path.join(result_path, TASK_NAME)
if not os.path.exists(result_path):
  os.makedirs(result_path)
os.makedirs(result_path, exist_ok=True)
mean_value = [1, 0, 59.0, 0.21, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,\
  0, 0, 0, 0, 1, 128.8, 86, 170.0, 77.0, 98.0, 19.0, 118.0, 36.6, 81.0, 7.4]
mean_value = torch.tensor(mean_value).cuda()
#all_x_add = np.load(rootpath + 'input/all_x_add.npy', allow_pickle=True)
#dataset = np.load(rootpath + 'input/dataset.npy', allow_pickle=True)
# dataset = np.load(os.path.join(rootpath, 'X.npy'), allow_pickle=True)
# dt = np.load(os.path.join(rootpath, 'dt.npy'), allow_pickle=True)
# 0:death 1:length of stay(<3) 2:Cardical 3:Surgery
# y = np.load(os.path.join(rootpath, 'y.npy'), allow_pickle=True)
# y1 = y[:,0:1] 
class_list = ['Acute and unspecified renal failure', 
              'Acute cerebrovascular disease', 
              'Acute myocardial infarction',
  	          'Cardiac dysrhythmias',	
              'Chronic kidney disease',	
              'Chronic obstructive pulmonary disease and bronchiectasis',
              'Complications of surgical procedures or medical care',	
              'Conduction disorders',	
              'Congestive heart failure; nonhypertensive',
        	    'Coronary atherosclerosis and other heart disease', 
              'Diabetes mellitus with complications',	
              'Diabetes mellitus without complication',	
              'Disorders of lipid metabolism',
            	'Essential hypertension',
              'Fluid and electrolyte disorders',	
              'Gastrointestinal hemorrhage', 
              'Hypertension with complications',
              'Other liver diseases',
              'Other lower respiratory disease',
              'Other upper respiratory disease',
              'Pleurisy; pneumothorax; pulmonary collapse',	
              'Pneumonia',
              'Respiratory failure; insufficiency; arrest',	
              'Septicemia',	
              'Shock']
label_index = [0, 1, 20, 21, 22, 23, 24]
task_list = []
for i in label_index:
  task_list.append(class_list[i])
performance_dict_val = dict(zip(task_list, [[] for i in task_list]))
performance_dict_val['Epoch'] = []
performance_dict_val['Mean'] = []
performance_dict_test = dict(zip(task_list, [[] for i in task_list]))
performance_dict_test['Epoch'] = []
performance_dict_test['Mean'] = []

class GRUD_ODECell(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    
    self.inputzeros = torch.autograd.Variable(torch.zeros(input_size)).cuda()
    self.hiddenzeros = torch.autograd.Variable(torch.zeros(hidden_size)).cuda()
    
    self.w_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
    self.w_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size,input_size))
    self.b_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
    self.b_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))

    self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.lin_hu = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_su = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_sz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_sr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_mu = torch.nn.Linear(input_size, hidden_size, bias=False)
    self.lin_mz = torch.nn.Linear(input_size, hidden_size, bias=False)
    self.lin_mr = torch.nn.Linear(input_size, hidden_size, bias=False)

  def forward(self, h, x, m, d, prex, dh):
    gamma_x = torch.exp(-torch.max(self.inputzeros, (self.w_dg_x*d + self.b_dg_x)))
    gamma_h = torch.exp(-torch.max(self.hiddenzeros, (torch.matmul(self.w_dg_h, d) + self.b_dg_h)))

    # use gamma x
    # x = m * x + (1 - m) * (gamma_x * prex + (1 - gamma_x) * mean_value)
    x = m * x + (1 - m) * prex

    # use gamma h
    # h = gamma_h * h
    
    # with volecity perception
    r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h) + self.lin_sr(dh) +self.lin_mr(m))
    z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h) + self.lin_sz(dh) + self.lin_mz(m))
    u = torch.tanh(self.lin_xh(x) + self.lin_hu(r * h) + self.lin_su(dh) + self.lin_mu(m))

    h_post = (1-z) * h + z * u
    dh = z * (u - h)

    return h_post, dh, x

class GRU_TV(torch.nn.Module):  
  def __init__(self, input_size, hidden_size, output_size, dropout_type, dropout):
    super().__init__()

    self.dropout_type = dropout_type
    self.dropout_rate = dropout
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.cell = GRUD_ODECell(input_size=input_size, hidden_size=hidden_size)
    self.cell = self.cell.cuda()

    self.lin = torch.nn.Linear(hidden_size, output_size, bias=True)
    self.reset_parameters()
    self.dropout = torch.nn.Dropout(p=self.dropout_rate)
  def reset_parameters(self):
    for params in self.parameters():
      torch.nn.init.normal_(params, mean=0, std=0.1)
  def forward(self, X, Mask, Delta, dt):
    h = torch.autograd.Variable(torch.zeros(1, self.hidden_size)).cuda()
    c = torch.autograd.Variable(torch.zeros(1, self.hidden_size)).cuda()
    dh = torch.autograd.Variable(torch.zeros(self.hidden_size)).cuda()
    prex = torch.autograd.Variable(torch.zeros(self.input_size)).cuda()
    if len(X.shape) != 2:
      print(X.shape)
    for layer in range(X.shape[0]):

      x = X[layer, :]
      x = torch.unsqueeze(x, dim=0)
      m = Mask[layer, :]
      d = Delta[layer, :]
      if self.dropout_rate == 0:
        h_post, dh, prex = self.cell(h, x, m, d, prex, dh)
        h = h + dt[layer]*dh
      elif self.dropout_type == 'Moon':
        h_post, dh, prex = self.cell(h, x, m, d, prex, dh)
        h = h + dt[layer]*dh
        h = self.dropout(h)
        dh = self.dropout(dh)
      elif self.dropout_type == 'Gal':
        h = self.dropout(h)
        h_post, dh, prex = self.cell(h, x, m, d, prex, dh)
        h = h + dt[layer]*dh
      elif self.dropout_type == 'Mloss':
        h_post, dh, prex = self.cell(h, x, m, d, prex, dh)
        dh = self.dropout(dh)
        h = h + dt[layer]*dh
      else:
        raise NotImplementedError 
    output = self.lin(h_post)      
    output = torch.sigmoid(output)
    return output


def fit(model, criterion, learning_rate,\
        train_dataloader, dev_dataloader, test_dataloader,\
        learning_rate_decay, n_epochs, ckpt):
  epoch_losses = []
  best_val = -1
  best_test = -1
  no_increase = 0
  if ckpt:
    epoch_start = ckpt['epoch']
  else:
    epoch_start = 0
  
  for epoch in range(epoch_start, n_epochs):   
    if learning_rate_decay != 0:
      current_lr = learning_rate/(2**(epoch//learning_rate_decay))
      optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-4)
    else:
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
      
    losses = []
    model.train()
    for train_data, mask, delta, dt, train_label in tqdm(train_dataloader):
      optimizer.zero_grad() 
      train_data = torch.squeeze(train_data, 0).cuda()
      mask = torch.squeeze(mask, 0).cuda()
      delta = torch.squeeze(delta, 0).cuda()
      dt = torch.squeeze(dt, 0).cuda()
      train_label = torch.squeeze(train_label, 0).cuda()
      y_pred = model(train_data, mask, delta, dt).cuda()

      assert torch.max(y_pred) <= 1 and torch.min(y_pred) >=0, print(y_pred)
      assert torch.max(train_label) <= 1 and torch.min(train_label) >=0, print(train_label)
      y_pred = torch.squeeze(y_pred)
      loss = criterion(y_pred, train_label)

      losses.append(loss.item())

      loss.backward()
      optimizer.step()
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, os.path.join(result_path, 'last.pt'))
    torch.save(model.state_dict(), os.path.join(result_path, 'last.pth'))
    print('='*35+str(epoch)+'='*35)
    need_test = False
    auc_mean = test_model(model, dev_dataloader, 'VAL', best_val, epoch)
    if auc_mean > best_val:
      best_val = auc_mean
      no_increase = 0
      need_test = True
      print('='*35+'IMPROVED'+'='*35)
    else:
      no_increase += 1

    if no_increase >= 4 and epoch > 25:
      break

    if need_test:
        auc_mean = test_model(model, test_dataloader, 'TEST', best_test, epoch)
        if auc_mean > best_test:
          best_test = auc_mean
          print('='*35+'NEW BEST'+'='*35)

def test_model(model, dev_dataloader, task, best_per, epoch):
    losses = []
    pred, label = list(), list()
    model.eval()
    # series, mask, delta, dt, y
    for dev_data, mask, delta, dt, dev_label in tqdm(dev_dataloader):
      dev_data = torch.squeeze(dev_data, 0).cuda()
      mask = torch.squeeze(mask, 0).cuda()
      delta = torch.squeeze(delta, 0).cuda()
      dev_label = torch.squeeze(dev_label, 0).cuda()
      dt = torch.squeeze(dt, 0).cuda()

      y_pred = model(dev_data, mask, delta, dt)
      
      pred.append(y_pred.cpu().detach().numpy().tolist())
      label.append(dev_label.cpu().detach().numpy().tolist())
      y_pred = torch.squeeze(y_pred)
      loss = criterion(y_pred, dev_label)
      losses.append(loss.item())
          
    dev_loss = np.mean(losses)
    
    pred = np.asarray(pred)
    label = np.asarray(label)

    pred = np.squeeze(pred)
    label = np.squeeze(label)

    auc_mean, roc_auc = get_performance(predicts=pred, labels=label, best_per=best_per, save_path=result_path, task=task)
    if auc_mean > best_per:
      best_per = auc_mean
      np.save(os.path.join(result_path, 'pred_best_{}.npy'.format(task)) , pred)
      np.save(os.path.join(result_path, 'label_best_{}.npy'.format(task)), label)
      torch.save(model.state_dict(), os.path.join(result_path, 'model_best_{}.pkl'.format(task)))
    
    print('='*35+task+'='*35)
    print("MEAN: {:.4f}".format(auc_mean))
    for idx, class_index in enumerate(label_index):
      print(f'{class_list[class_index]}: {round(roc_auc_score(label[:,idx], pred[:,idx]), 4)}')
    print('')
    if task == 'VAL':
      performance_dict_val['Epoch'].append(epoch)
      performance_dict_val['Mean'].append(auc_mean)
      for idx, class_index in enumerate(label_index):
        performance_dict_val[class_list[class_index]].append(round(roc_auc_score(label[:,idx], pred[:,idx]), 4))
      pd.DataFrame(performance_dict_val).to_csv(os.path.join(result_path, 'performance_val.csv'))
    elif task == 'TEST':
      performance_dict_test['Epoch'].append(epoch)
      performance_dict_test['Mean'].append(auc_mean)
      for idx, class_index in enumerate(label_index):
        performance_dict_test[class_list[class_index]].append(round(roc_auc_score(label[:,idx], pred[:,idx]), 4))
      pd.DataFrame(performance_dict_test).to_csv(os.path.join(result_path, 'performance_test.csv'))
    return auc_mean


def get_performance(predicts, labels, best_per, save_path, task):
    if not isinstance(predicts, np.ndarray):
        predicts = np.array(predicts)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_total = 0

    labels[labels<0.5]=0
    labels[labels!=0]=1

    for i in range(len(label_index)):
        class_name = 'class_{}'.format(str(i))
        a, b = labels[:, i], predicts[:, i]
        fpr[class_name], tpr[class_name], _ = roc_curve(a, b)
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
        auc_total += roc_auc[class_name]
    
    auc_mean = auc_total / len(label_index)
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

    print('AUC:{}'.format('  '.join([str(round(roc_auc[x], 2)) for x in roc_auc.keys()])))
    print('AUC_MEAN:{}'.format(auc_mean))
    plt.close()
    return auc_mean, roc_auc

def data_dataloader(dataset, outcomes, dt, \
                    train_proportion = 0.7, dev_proportion = 0.15, test_proportion=0.15):
    
  train_index = int(np.floor(dataset.shape[0] * train_proportion))
  val_index = int(np.floor(dataset.shape[0] * dev_proportion))
  val_index = train_index + val_index
  # split dataset to tarin/dev/test set
  train_data, train_label = dataset[:train_index,:,:,:], outcomes[:train_index,:]
  dev_data, dev_label = dataset[train_index: val_index,:,:,:], outcomes[train_index:val_index,:]  
  test_data, test_label = dataset[val_index: ,:,:,:], outcomes[val_index: ,:]  
  train_dt, dev_dt, test_dt = dt[:train_index,:], dt[train_index:val_index,:], dt[val_index: ,:]
    
  # ndarray to tensor
  train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
  dev_data, dev_label = torch.Tensor(dev_data), torch.Tensor(dev_label)
  test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
  train_dt, dev_dt, test_dt = torch.Tensor(train_dt), torch.Tensor(dev_dt), torch.Tensor(test_dt)
  
  # tensor to dataset
  train_dataset = utils.TensorDataset(train_data, train_label, train_dt)
  dev_dataset = utils.TensorDataset(dev_data, dev_label, dev_dt)
  test_dataset = utils.TensorDataset(test_data, test_label, test_dt)
  
  # dataset to dataloader 
  train_dataloader = utils.DataLoader(train_dataset)
  dev_dataloader = utils.DataLoader(dev_dataset)
  test_dataloader = utils.DataLoader(test_dataset)
  
  return train_dataloader, dev_dataloader, test_dataloader



if __name__ == '__main__':    
  input_size = 33
  hidden_size = 64 
  output_size = 4
  
  #dropout_type : Moon, Gal, mloss
  model = GRU_TV(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=0)
  model = nn.DataParallel(model)
  model = model.cuda()

  ckpt = None
  criterion = torch.nn.BCELoss()
  
  learning_rate = 0.015
  learning_rate_decay = 3
  n_epochs = 100
  train_dataset = PhysoinetDatassetResample(pd.read_csv('/path/to/train_listfile.csv'), 
                                                        '/path/to/Discretization/train',
                                                        label_index,
                                                        '/path/to/sample_index.json')
  val_dataset =  PhysoinetDatassetResample(pd.read_csv('/path/to/val_listfile.csv'), 
                                                        '/path/to/Discretization/val',
                                                        label_index,
                                                        '/path/to/sample_index.json')
  test_dataset =  PhysoinetDatassetResample(pd.read_csv('/path/to/test_listfile.csv'), 
                                                        '/path/to/Discretization/test',
                                                        label_index,
                                                        '/path/to/sample_index.json')
  train_dataloader = DataLoader(train_dataset, num_workers=0)
  val_dataloader = DataLoader(val_dataset, num_workers=0)
  test_dataloader = DataLoader(test_dataset, num_workers=0)
  fit(model, criterion, learning_rate,\
      train_dataloader, val_dataloader, test_dataloader,\
      learning_rate_decay, n_epochs, ckpt)
