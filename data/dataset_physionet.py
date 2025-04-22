import json
import torch


from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import random
statistic = json.load(open(r'/path/to/statistic.json'))
channel_info = json.load(open(r'/path/to/channel_info.json'))

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
    def __init__(self, list_df, data_root, label_index, index_file, cache_size=1000, max_len=1000):
        super().__init__()
        if index_file:
            self.index_dict = json.load(open(index_file, mode='r'))
        else:
            self.index_dict = None
        self.list_df = list_df
        self.list_df = list_df
        self.data_root = data_root
        self.label_indx = label_index
        self.cache_size = cache_size
        self.max_len = max_len
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

        
        los_dict, label_dict = self.get_label_dict()
        self.los_dict = los_dict
        self.label_dict = label_dict
        self.cache_buffer = self.cache_data()
        # ML_features, labels = self.feature_label()
        # self.ML_features = ML_features
        # self.labels = labels
   
    def get_label_dict(self):
        label_dict = {}
        los_dict =  {}
        for idx, row in self.list_df.iterrows():
            label_dict[row['stay']] = np.array(row[2:].tolist(), dtype=np.uint8)[self.label_indx]
            los_dict[row['stay']] = float(row[1])/24
        return los_dict, label_dict
    
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
            if self.index_dict:
                sampled_idx = self.index_dict[stay]
                sampled_idx.sort()
                series = pd.read_csv(os.path.join(self.data_root, stay)).iloc[sampled_idx].reset_index(drop=True)
                mask = pd.read_csv(os.path.join(self.data_root, stay.replace('.csv', '_mask.csv'))).iloc[sampled_idx].reset_index(drop=True)
            else:
                series = pd.read_csv(os.path.join(self.data_root, stay))
                mask = pd.read_csv(os.path.join(self.data_root, stay.replace('.csv', '_mask.csv')))


            delta = pd.DataFrame(self.get_delta(list(series['Hours']), mask))
            if len(series)>5:
                series_before_3 = list(series['Hours']<=24*3)
                series_before_3 = max(np.sum(np.array(series_before_3)*1), 5)
            else:
                series_before_3 = len(series)
            series = series[:series_before_3]
            mask = mask[:series_before_3]
            delta = delta[:series_before_3]

            y = self.label_dict[stay]
            y_los = self.los_dict[stay]
            

            dt_list = list(series['Hours'].diff())
            dt_list[0] = 1
            dt = np.array(dt_list)
            
            if self.max_len < len(series) and self.max_len != -1:
                indexes = [i for i in range(len(series))]
                sampled_idx = random.sample(indexes, self.max_len)
                series = series.iloc[sampled_idx].reset_index(drop=True)
                mask = mask.iloc[sampled_idx].reset_index(drop=True)
                delta = delta.iloc[sampled_idx].reset_index(drop=True)
                dt = dt[sampled_idx]

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
            features = self.get_feature(np.array(series_array), np.array(mask_array))
            cache_buffer[stay] = [np.array(series_array), np.array(mask_array), np.array(delta_array), dt, y, y_los]
        return cache_buffer
    
    def feature_label(self):
        feature_list = [self.cache_buffer[stay][-1] for stay in self.list_df['stay']]
        y_list = np.array([self.cache_buffer[stay][-2] for stay in self.list_df['stay']])
        feature_list = np.array(feature_list, dtype=np.float32)
        return feature_list, y_list

    def get_feature(self, series, mask):
        series_len = len(series)
        sub_index = [
                        [0, series_len],
                        [0, int(series_len*0.1)],
                        [0, int(series_len*0.25)], 
                        [0, int(series_len*0.5)],
                        [int(series_len*0.5), series_len],
                        [int(series_len*0.75), series_len],
                        [int(series_len*0.0), series_len],
                   ]
        f_list = []
        for f_idx in range(series.shape[1]):
            single_f_futures = []
            for idx in sub_index:
                begin, end = idx
                f_s = series[begin: end, f_idx]
                f_m = mask[begin: end, f_idx]

                value_num = np.sum(f_m)
                if value_num == 0:
                    max_value, min_value = np.max(series[:, f_idx]), np.min(series[:, f_idx])
                    mean, std = np.mean(series[:, f_idx]), np.std(series[:, f_idx])
                    skew = std
                else:
                    f_s[f_m==0] = np.nan
                    max_value, min_value = np.nanmax(f_s), np.nanmin(f_s)
                    mean, std = np.nanmean(f_s), np.nanstd(f_s)
                    skew = pd.Series(f_s).skew(skipna=True)
                if np.isnan(max_value):
                    max_value =  0
                if np.isnan(min_value):
                    min_value = 0
                if np.isnan(mean):
                    mean =  0
                if np.isnan(std):
                    std =  0
                if np.isnan(skew):
                    skew =  0  
                single_f_futures.append([min_value, max_value, mean, std, skew])
            f_list.append(single_f_futures)
        return np.array(f_list)

    def __getitem__(self, index):
        stay = self.list_df['stay'][index]
        series, mask, delta, dt, y, y_los = self.cache_buffer[stay]
        series = torch.tensor(series, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        delta = torch.tensor(delta, dtype=torch.float32)
        dt = torch.tensor(dt, dtype=torch.float32)
        dt = dt / 12
        y = torch.tensor(y, dtype=torch.float32)
        y_los = torch.tensor(y_los, dtype=torch.float32)

        return series, mask, delta, dt, y, y_los


    def __len__(self):
        return len(self.list_df)

class PhysoinetDatasetCache(Dataset):
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

class Physionet2012(Dataset):
    def __init__(self, list_df, data_root):
        data_list = []
        for idx, row in list_df.iterrows():
            record_id = str(int(row['RecordID']))
            self.data_root = data_root
            series = np.load(os.path.join(self.data_root, 'timeseries', record_id+'.npy'))
            if len(series.shape) != 2 or len(series) <= 0:
                print(series)
            mask = np.load(os.path.join(self.data_root, 'mask', record_id+'.npy'))
            delta = np.load(os.path.join(self.data_root, 'delta', record_id+'.npy'))
            dt = np.load(os.path.join(self.data_root, 'dt', record_id+'.npy'))

            features = self.get_feature(series=series, mask=mask)
            series = torch.tensor(series, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)

            delta = torch.tensor(delta, dtype=torch.float32)
            dt = torch.tensor(dt, dtype=torch.float32)
            features = torch.tensor(features, dtype=torch.float32).view(-1)

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
                'label': y,
                'feature': features
            }
            data_list.append(data)           
        self.data_list = data_list
        ML_features, labels = self.feature_label()
        self.ML_features = ML_features
        self.labels = labels
    
    def feature_label(self):
        feature_list = [d['feature'] for d in self.data_list]
        y_list = np.array([d['label'].numpy().tolist() for d in self.data_list])
        feature_list = np.array(feature_list, dtype=np.float32)
        return feature_list, y_list

    def get_feature(self, series, mask):
        series_len = len(series)
        sub_index = [
                        [0, series_len],
                        [0, int(series_len*0.1)],
                        [0, int(series_len*0.25)], 
                        [0, int(series_len*0.5)],
                        [int(series_len*0.5), series_len],
                        [int(series_len*0.75), series_len],
                        [int(series_len*0.9), series_len],
                   ]
        f_list = []
        for f_idx in range(series.shape[1]):
            single_f_futures = []
            for idx in sub_index:
                begin, end = idx
                f_s = series[begin: end, f_idx]
                f_m = mask[begin: end, f_idx]

                value_num = np.sum(f_m)
                if value_num == 0:
                    channel_name = channel_info['id_to_channel'][f_idx]
                    max_value, min_value = statistic['max'][channel_name], statistic['min'][channel_name]
                    mean, std = statistic['mean'][channel_name], statistic['std'][channel_name]
                    skew = std
                else:
                    f_s[f_m==0] = np.nan
                    max_value, min_value = np.nanmax(f_s), np.nanmin(f_s)
                    mean, std = np.nanmean(f_s), np.nanstd(f_s)
                    skew = pd.Series(f_s).skew(skipna=True)
                    if np.isnan(max_value):
                        max_value =  statistic['max'][channel_name]
                    if np.isnan(min_value):
                        min_value =  statistic['min'][channel_name] 
                    if np.isnan(mean):
                        mean =  statistic['mean'][channel_name]
                    if np.isnan(std):
                        std =  statistic['std'][channel_name]
                    if np.isnan(skew):
                        skew =  std  
                single_f_futures.append([mean, std, max_value, min_value, skew])
            f_list.append(single_f_futures)
        return np.array(f_list)



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data
    