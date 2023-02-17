import json
import torch


from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

class PhysoinetDatasset(Dataset):
    def __init__(self, list_df, data_root, label_index) -> None:
        super().__init__()
        self.list_df = list_df
        self.data_root = data_root
        self.label_indx = label_index
    
    def __len__(self):
        return len(self.list_df)
    

    def __getitem__(self, index):
        stay = self.list_df['stay'][index]
        series = np.load(os.path.join(self.data_root, 'timeseries', stay.replace('.csv', '.npy')))
        if len(series.shape) != 2 or len(series) <= 0:
            print(stay)
        mask = np.load(os.path.join(self.data_root, 'mask', stay.replace('.csv', '.npy')))
        delta = np.load(os.path.join(self.data_root, 'delta', stay.replace('.csv', '.npy')))
        dt = np.load(os.path.join(self.data_root, 'dt', stay.replace('.csv', '.npy')))
        y = np.array(self.list_df.iloc[index, 2:], dtype=np.uint8)
        y = y[self.label_indx]
        series = torch.tensor(series, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        delta = torch.tensor(delta, dtype=torch.float32)
        dt = torch.tensor(dt, dtype=torch.float32)
        a, b = torch.min(dt), torch.max(dt)
        dt = dt / 12

        y = torch.tensor(y, dtype=torch.float32)

        return series, mask, delta, dt, y

class PhysoinetDatassetResample(Dataset):
    def __init__(self, list_df, data_root, label_index, index_file, cache_size=1000):
        super().__init__()
        self.index_dict = json.load(open(index_file, mode='r'))
        self.list_df = list_df
        self.data_root = data_root
        self.label_indx = label_index
        self.cache_size = cache_size
        self.colums = ['Hours', 'Capillary refill rate_0', 'Capillary refill rate_1',
                       'Diastolic blood pressure', 'Fraction inspired oxygen', 
                       'Glascow coma scale eye opening_0', 'Glascow coma scale eye opening_1', 
                       'Glascow coma scale eye opening_2', 'Glascow coma scale eye opening_3', 
                       'Glascow coma scale eye opening_4', 'Glascow coma scale motor response_0', 
                       'Glascow coma scale motor response_1', 'Glascow coma scale motor response_2', 
                       'Glascow coma scale motor response_3', 'Glascow coma scale motor response_4', 
                       'Glascow coma scale motor response_5', 'Glascow coma scale motor response_6', 
                       'Glascow coma scale total_0', 'Glascow coma scale total_1', 'Glascow coma scale total_2', 
                       'Glascow coma scale total_3', 'Glascow coma scale total_4', 'Glascow coma scale total_5', 
                       'Glascow coma scale total_6', 'Glascow coma scale total_7', 'Glascow coma scale total_8', 
                       'Glascow coma scale total_9', 'Glascow coma scale total_10', 'Glascow coma scale total_11', 
                       'Glascow coma scale total_12', 'Glascow coma scale verbal response_0', 
                       'Glascow coma scale verbal response_1', 'Glascow coma scale verbal response_2', 
                       'Glascow coma scale verbal response_3', 'Glascow coma scale verbal response_4', 'Glucose', 'Heart Rate', 
                       'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 
                       'Temperature', 'Weight', 'pH']

        self.label_dict = self.get_label_dict()
        self.cache_buffer = self.cache_data()
        
    def get_label_dict(self):
        label_dict = {}
        for idx, row in self.list_df.iterrows():
            label_dict[row['stay']] = np.array(row[2:].tolist(), dtype=np.uint8)[self.label_indx]
        return label_dict
    
    def get_delta(self, hours_list, mask_df: pd.DataFrame):
        columns = self.colums[1:]
        prior_time_dict = dict(zip(columns, [0 for i in range(len(columns))]))
        delta_dict = dict(zip(columns, [[1] for i in range(len(columns))]))
        for idx, row in mask_df.iterrows():
            if idx == 0:
                continue
            for col in columns:
                if row[col] == 0:
                    prior_time_dict[col] = prior_time_dict[col] + (hours_list[idx] - hours_list[idx-1])
                else:
                    prior_time_dict[col] = (hours_list[idx] - hours_list[idx-1])
                dt_value = prior_time_dict[col]
                delta_dict[col].append(dt_value)

        return delta_dict
        
    
    def cache_data(self):
        cache_buffer = {}
        stay_list = self.list_df['stay'].tolist()
        for stay in tqdm(stay_list):
            sampled_idx = self.index_dict[stay]
            sampled_idx.sort()
            series = pd.read_csv(os.path.join(self.data_root, stay)).iloc[sampled_idx].reset_index(drop=True)
            mask = pd.read_csv(os.path.join(self.data_root, stay.replace('.csv', '_mask.csv'))).iloc[sampled_idx].reset_index(drop=True)

            delta = pd.DataFrame(self.get_delta(list(series['Hours']), mask))
            y = self.label_dict[stay]
            

            dt_list = list(series['Hours'].diff())
            dt_list[0] = 1
            dt = np.array(dt_list)

            series_array = []
            mask_array = []
            delta_array = []
            for i in range(len(series)):
                series_list = []
                mask_list = []
                delta_list = []
                for col in self.colums[1:]:
                    series_list.append(series[col][i])
                    mask_list.append(mask[col][i])
                    delta_list.append(delta[col][i])
                series_array.append(series_list)
                mask_array.append(mask_list)
                delta_array.append(delta_list)

            cache_buffer[stay] = [np.array(series_array), np.array(mask_array), np.array(delta_array), dt, y]
        return cache_buffer
    
    def __getitem__(self, index):
        stay = self.list_df['stay'][index]
        series, mask, delta, dt, y = self.cache_buffer[stay]
        series = torch.tensor(series, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        delta = torch.tensor(delta, dtype=torch.float32)
        dt = torch.tensor(dt, dtype=torch.float32)
        dt = dt / 12
        y = torch.tensor(y, dtype=torch.float32)

        return series, mask, delta, dt, y


    def __len__(self):
        return len(self.list_df)

class PhysoinetDatassetCache(Dataset):
    def __init__(self, list_df, data_root, label_index, cache_size=1000) -> None:
        super().__init__()
        self.list_df = list_df
        self.data_root = data_root
        self.label_indx = label_index
        self.cache_size = cache_size
        self.label_dict = self.get_label_dict()
        self.cache_buffer = self.cache_data()
        
    def get_label_dict(self):
        label_dict = {}
        for idx, row in self.list_df.iterrows():
            label_dict[row['stay']] = np.array(row[2:].tolist(), dtype=np.uint8)[self.label_indx]
        return label_dict

    def __len__(self):
        return len(self.list_df)
    
    def cache_data(self):
        cache_buffer = {}
        stay_list = self.list_df['stay'].tolist()
        for stay in tqdm(stay_list):
            series = np.load(os.path.join(self.data_root, 'timeseries', stay.replace('.csv', '.npy')))
            if len(series.shape) != 2 or len(series) <= 0:
                print(stay)
            mask = np.load(os.path.join(self.data_root, 'mask', stay.replace('.csv', '.npy')))
            delta = np.load(os.path.join(self.data_root, 'delta', stay.replace('.csv', '.npy')))
            dt = np.load(os.path.join(self.data_root, 'dt', stay.replace('.csv', '.npy')))
            y = self.label_dict[stay]
            cache_buffer[stay] = [series, mask, delta, dt, y]
        return cache_buffer

    def __getitem__(self, index):
        stay = self.list_df['stay'][index]
        series, mask, delta, dt, y = self.cache_buffer[stay]

        series = torch.tensor(series, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        delta = torch.tensor(delta, dtype=torch.float32)
        dt = torch.tensor(dt, dtype=torch.float32)
        dt = dt / 12
        y = torch.tensor(y, dtype=torch.float32)

        return series, mask, delta, dt, y
    
if __name__ == '__main__':
   pass