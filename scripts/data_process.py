import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


mimic_column_list = ['Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight']
'''
{'Capillary refill rate': 0.9978967485022489, 'Diastolic blood pressure': 0.29778155055021244, 'Fraction inspired oxygen': 0.9329051927965725, 
'Glascow coma scale eye opening': 0.8137343336118469, 'Glascow coma scale motor response': 0.8144784981600327, 'Glascow coma scale total': 0.8880247462267338, 
'Glascow coma scale verbal response': 0.8142584131837656, 'Glucose': 0.835828696378731, 'Heart Rate': 0.2488421539172637, 'Height': 0.9983232298092478, 
'Mean blood pressure': 0.30242217916125935, 'Oxygen saturation': 0.2483328296385842, 'Respiratory rate': 0.24047999146681837, 'Systolic blood pressure': 0.29754333256297666, 
'Temperature': 0.7921389841958365, 'Weight': 0.9809388277541733}

Hours:-121.92902422170073
Capillary refill rate_0:0.9978749896492742
Capillary refill rate_1:0.9978749896492742
Diastolic blood pressure:0.2988337081608424
Fraction inspired oxygen:0.9326489069379581
Glascow coma scale eye opening_0:0.8137278310314504
Glascow coma scale eye opening_1:0.8137278310314504
Glascow coma scale eye opening_2:0.8137278310314504
Glascow coma scale eye opening_3:0.8137278310314504
Glascow coma scale eye opening_4:0.8137278310314504
Glascow coma scale motor response_0:0.8144644470985201
Glascow coma scale motor response_1:0.8144644470985201
Glascow coma scale motor response_2:0.8144644470985201
Glascow coma scale motor response_3:0.8144644470985201
Glascow coma scale motor response_4:0.8144644470985201
Glascow coma scale motor response_5:0.8144644470985201
Glascow coma scale motor response_6:0.8144644470985201
Glascow coma scale total_0:0.8879589181599229
Glascow coma scale total_1:0.8879589181599229
Glascow coma scale total_2:0.8879589181599229
Glascow coma scale total_3:0.8879589181599229
Glascow coma scale total_4:0.8879589181599229
Glascow coma scale total_5:0.8879589181599229
Glascow coma scale total_6:0.8879589181599229
Glascow coma scale total_7:0.8879589181599229
Glascow coma scale total_8:0.8879589181599229
Glascow coma scale total_9:0.8879589181599229
Glascow coma scale total_10:0.8879589181599229
Glascow coma scale total_11:0.8879589181599229
Glascow coma scale total_12:0.8879589181599229
Glascow coma scale verbal response_0:0.8142353341011074
Glascow coma scale verbal response_1:0.8142353341011074
Glascow coma scale verbal response_2:0.8142353341011074
Glascow coma scale verbal response_3:0.8142353341011074
Glascow coma scale verbal response_4:0.8142353341011074
Glucose:0.8357757659410067
Heart Rate:0.2503448125210507
Height:0.9982702222700881
Mean blood pressure:0.303474897394462
Oxygen saturation:0.24893508177200363
Respiratory rate:0.24183994501288064
Systolic blood pressure:0.29859545096397644
Temperature:0.7912526588029938
Weight:0.9803359070868054
pH:0.9312772770199663
'''

def discretization():
    sample_rate = 0.5
    for mode in ['train', 'val', 'test']:
        data_root = r''
        data_save = r''
        channel_info = json.load(open(r''))
        discretization_info = json.load(open(r''))
        columns = channel_info['id_to_channel']
        new_columns = ['Hours']
        for column in columns:
            if channel_info["is_categorical_channel"][column]:
                new_columns.extend([column+'_'+str(i) for i in range(len(discretization_info[column]['category']))])
            else:
                new_columns.append(column)
        list_df = pd.read_csv(os.path.join(data_root, f'{mode}_listfile.csv'))
        list_dict = list_df.to_dict(orient='list')
        if mode == 'test':
            sub_folder = 'test'
        else:
            sub_folder = 'train'
        for idx, row in tqdm(list_df.iterrows()):
            sequence_name = row['stay']
            # print(sequence_name)
            last = channel_info['normal_values']
            delta_value_dict = dict(zip(columns, [1 for i in range(len(columns))]))
            sequence_df = pd.read_csv(os.path.join(data_root, sub_folder, sequence_name))
            sequence_df = sequence_df.sample(frac=sample_rate, random_state=42).sort_index().reset_index()
            if len(sequence_df) < 5:
                continue
            hours_list = sequence_df['Hours'].tolist()
            try:
                list_dict['period_length'][idx] = hours_list[-1]
            except:
                print(sequence_name)
            new_sequence_dict = dict(zip(new_columns, [[] for col in new_columns]))
            delta_dict = dict(zip(new_columns, [[] for col in new_columns]))
            mask_dict = dict(zip(new_columns, [[] for col in new_columns]))
            last_mask = dict(zip(columns, [1 for i in range(len(columns))]))
            for idx_1, row_1 in sequence_df.iterrows():
                for column in columns:
                    value = row_1[column]
                    if pd.isnull(value) or (not value == value) or value == '':
                        value = last[column]
                        curr_mask = 0

                    else:
                        last[column] = value
                        curr_mask = 1

                    if idx_1 > 0:
                        if last_mask[column] == 1:
                            delta_value_dict[column] = hours_list[idx_1] - hours_list[idx_1-1]
                        else:
                            delta_value_dict[column] = delta_value_dict[column] + (hours_list[idx_1] - hours_list[idx_1-1])
                    last_mask[column] = curr_mask
                    if channel_info["is_categorical_channel"][column]:
                        try:
                            value = str(int(float(value)))
                        except ValueError:
                            value = str(value)
                        digital_value = discretization_info[column]['values'][value]
                        category_idx = discretization_info[column]['category'].index(digital_value)
                        for i in range(len(discretization_info[column]['category'])):
                            new_sequence_dict[column+'_'+str(i)].append(0)
                            mask_dict[column+'_'+str(i)].append(curr_mask)
                            delta_dict[column+'_'+str(i)].append(delta_value_dict[column])
                        new_sequence_dict[column+'_'+str(category_idx)][-1] = 1
                    else:
                        new_sequence_dict[column].append(float(value))
                        mask_dict[column].append(curr_mask)
                        delta_dict[column].append(delta_value_dict[column])
            new_sequence_dict['Hours'] = sequence_df['Hours'].tolist()
            mask_dict['Hours'] = sequence_df['Hours'].tolist()
            delta_dict['Hours'] = sequence_df['Hours'].tolist()
            pd.DataFrame(new_sequence_dict).to_csv(os.path.join(data_save, sub_folder, sequence_name), index=False)
            pd.DataFrame(mask_dict).to_csv(os.path.join(data_save, sub_folder, sequence_name.replace('.csv', '_mask.csv')), index=False)
            pd.DataFrame(delta_dict).to_csv(os.path.join(data_save, sub_folder, sequence_name.replace('.csv', '_delta.csv')), index=False)
        
        pd.DataFrame(list_dict).to_csv(os.path.join(data_save, f'{mode}_listfile.csv'), index=False)


def serialization():
    for mode in ['train', 'test', 'val']:
        data_folder = r''
        save_data = r''
        if mode == 'test':
            sub_folder = 'test'
        else:
            sub_folder = 'train'

        list_df = pd.read_csv(os.path.join(data_folder, f'{mode}_listfile.csv'))
        for stay_name in tqdm(list_df.to_dict(orient='list')['stay']):
            stay_path = os.path.join(data_folder, sub_folder, stay_name)
            if not os.path.exists(stay_path):
                print(stay_name)
                continue
            series_df = pd.read_csv(stay_path)
            mask_df = pd.read_csv(os.path.join(data_folder, sub_folder, stay_name.replace('.csv', '_mask.csv')))
            delta_df = pd.read_csv(os.path.join(data_folder, sub_folder, stay_name.replace('.csv', '_delta.csv')))

            dt_array = np.array(series_df.diff()['Hours'], dtype=np.float32)
            dt_array[0] = 1
            np.save(os.path.join(save_data, mode, 'dt', stay_name.split('.')[0]+'.npy'), dt_array)
            np.save(os.path.join(save_data, mode, 'timeseries',stay_name.split('.')[0]+'.npy'), np.array(series_df.iloc[:, 1:], dtype=np.float32))
            np.save(os.path.join(save_data, mode, 'mask',stay_name.split('.')[0]+'.npy'), np.array(mask_df.iloc[:, 1:], dtype=np.float32))
            np.save(os.path.join(save_data, mode, 'delta',stay_name.split('.')[0]+'.npy'), np.array(delta_df.iloc[:, 1:], dtype=np.float32))


def statistic_data_length():
    dt_path = r''

    dt_array = np.load(dt_path)
    dt_array = np.squeeze(dt_array)
    length_list = []
    for dt in dt_array:
        try:
            length = np.where(dt==0)[0][0]+1
        except:
            length = len(dt)
        length_list.append(length)
    length_list = np.array(length_list)
    mean, std = length_list.mean(), length_list.std()
    print(mean, std)

def statistic_data_time_interval():
    dt_list = []
    data_path = r''
    dt_array = np.load(data_path)
    for i in range(dt_array.shape[0]):
        dt_single = dt_array[i]
        dt_single = dt_single[dt_single>0]
        dt_list.extend(dt_single.tolist())
    mean, std = np.mean(dt_list), np.std(dt_list)
    print(f'mean:{mean}, std: {std}')
    

def statistic_data_time_interval_mimic():
    dt_list = []
    for mode in ['train', 'test', 'val']:
        data_path = f''
        for file_name in os.listdir(data_path):
            file_path = os.path.join(data_path, file_name)
            dt_array = np.load(file_path)
            dt_array = dt_array * 60
            dt_list.extend(dt_array.tolist())
    dt_array = np.array(dt_list)
    mean_value, std_value = np.mean(dt_array), np.std(dt_array)
    print(f'mean_value: {mean_value}, std_value: {std_value}')
    

def get_max_length():
    # test 6281
    # train 44018

    max_len = 0
    for file_name in os.listdir(r''):
        length = len(pd.read_csv(os.path.join(r'', file_name)))
        if length > max_len:
            print(file_name)
            max_len = length
    print(max_len)

def get_missing_rate():
    data_path = r''
    data = np.load(data_path, allow_pickle=True)
    dt = np.load(data_path.replace('dataset_50.npy', 'dt_50.npy'), allow_pickle=True)
    mask = data[:, 1, ...]
    length_list = dict(zip([i for i in range(33)], [0 for j in range(33)]))
    value_num_list = dict(zip([i for i in range(33)], [0 for j in range(33)]))
    for i in range(dt.shape[0]):
        if not 0 in dt[i]:
            length = len(dt[i])
        else:
            length = np.min(np.where(dt[i]==0))
        mask_array = mask[i]
        for j in range(mask_array.shape[0]):
            length_list[j] += length
            value_num_list[j] += np.sum(mask_array[j][:length] == 1)
    print([1 - value_num_list[i]/length_list[i] for i in range(33)])

def get_missing_rate_mimic():
    data_dir = r''
    length = 0
    value_dict = None
    for root, _, file_list in os.walk(data_dir):
        file_list = list(filter(lambda x: x.endswith('mask.csv'), file_list))
        for file_name in file_list:
            mask_df = pd.read_csv(os.path.join(root, file_name))
            length_series = len(mask_df)
            length += length_series
            if not value_dict:
                value_dict = dict(zip(mask_df.columns, [0 for i in mask_df.columns]))

            mask_df = mask_df.sum().to_dict()
            for key in mask_df:
                value_dict[key] += mask_df[key]
    
    for key in value_dict:
        print(key, end=':')
        print(1-value_dict[key] / length)

def filter_the_list_file():
    data_dir = r''
    for mode in ['train', 'test', 'val']:
        list_df = pd.read_csv(os.path.join(data_dir, f'{mode}_listfile.csv'))
        data_fold = 'test' if mode == 'test' else 'train'
        ts_list = os.listdir(os.path.join(data_dir, data_fold))
        list_df = list_df[list_df['stay'].isin(ts_list)]
        list_df.to_csv(os.path.join(data_dir, f'{mode}_listfile_filtered.csv'), index=False)

def gen_dt_list():
    index_json = open(r'', mode='r')
    index_dict = json.load(index_json)
    train_list = os.listdir(os.path.join(r''))
    save_path = r''
    all_dt = []
    avg_sample_rate = 0
    for file_name in tqdm(index_dict.keys()):
        if file_name in train_list:
            file_path = os.path.join(r'', file_name)
        else:
            file_path = os.path.join(r'', file_name)
        ts_df = pd.read_csv(file_path)
        # idx_list = index_dict[file_name]
        idx_list = list(range(len(ts_df)))
        avg_sample_rate += len(idx_list) / len(ts_df)
        idx_list.sort()
        ts_df = ts_df.iloc[idx_list]
        dt_list = list(ts_df['Hours'].diff())
        all_dt.extend(dt_list[1:])
    df_plt = pd.DataFrame({'DT': all_dt})
    df_plt.to_csv(os.path.join(save_path, 'sampled_100_dt.csv'))
    statistic_dict = {'BIN': [], 'COUNT': []}
    print(avg_sample_rate/len(index_dict))
    # sns.displot(df_plt, x='DT')
    bins = np.arange(0, 51, 1).tolist()
    for i in range(len(bins)-1):
        center = round((bins[i] + bins[i+1]) / 2, 2)
        statistic_dict['BIN'].append(center)
        count = len(df_plt[(bins[i]<= df_plt['DT']) & (df_plt['DT'] < bins[i+1])])
        statistic_dict['COUNT'].append(count)

    plt.savefig(os.path.join(save_path, 'sampled_100.png'))
    pd.DataFrame(statistic_dict).to_csv(os.path.join(save_path, 'sampled_100_statistic.csv'))


def statistic_dt_MIMIC():
    dt_path = r''
    save_path =r''
    df_plt = pd.read_csv(dt_path)
    statistic_dict = {'BIN': [], 'COUNT': []}
    bins = np.arange(0, 51, 1).tolist()
    for i in range(len(bins)-1):
        center = round((bins[i] + bins[i+1]) / 2, 2)
        statistic_dict['BIN'].append(center)
        count = df_plt[(bins[i]<= df_plt['DT']) & (df_plt['DT'] < bins[i+1])]['COUNT'].sum()
        statistic_dict['COUNT'].append(count)

    pd.DataFrame(statistic_dict).to_csv(save_path)


def statistic_dt_2012():
    dt_path = r''
    dt_array = np.load(dt_path, allow_pickle=True)
    dt_array[:, 0] = -1
    dt_array = np.reshape(dt_array, -1)
    dt_array = dt_array[dt_array>0]
    dt_array = dt_array

    pd.DataFrame({'DT': dt_array.tolist()}).to_csv(r'')
    save_path =r''
    statistic_dict = {'BIN': [], 'COUNT': []}
    bins = np.arange(0, 125, 5).tolist()
    for i in range(len(bins)-1):
        center = round((bins[i] + bins[i+1]) / 2, 2)
        statistic_dict['BIN'].append(center)
        count =  len(dt_array[(dt_array>=bins[i]) & (dt_array<bins[i+1])])
        statistic_dict['COUNT'].append(count)

    pd.DataFrame(statistic_dict).to_csv(save_path)

def statistic_miss_rate_2012():
    data_path = r''
    dt_path = r''
    data_array = np.load(data_path, allow_pickle=True)
    mask_array = data_array[:, 1]
    dt_array = np.load(dt_path, allow_pickle=True)
    len_dict = dict(zip([i for i in range(33)], [list() for i in range(33)]))
    num_dict =  dict(zip([i for i in range(33)], [list() for i in range(33)]))
    miss_dict =  dict(zip([i for i in range(33)], [list() for i in range(33)]))
    for i in range(len(dt_array)):
        try:
            seq_len = np.min(np.where(dt_array[i]==0))
        except:
            print(i)
            continue
        for j in range(33):
            num = sum(mask_array[i,j])
            assert num <= seq_len
            len_dict[j].append(seq_len)
            num_dict[j].append(num)
    for k in range(33):
        for l in range(len(num_dict[0])):
            miss_dict[k].append(1-num_dict[k][l]/len_dict[k][l])

    pd.DataFrame(miss_dict).to_csv(r'')
    
def statistic_miss_rate_MIMIC():
    data_path = r''
    
    miss_dict = {
        "Capillary refill rate": [],
        "Diastolic blood pressure": [],
        "Fraction inspired oxygen": [],
        "Glascow coma scale eye opening": [],
        "Glascow coma scale motor response": [],
        "Glascow coma scale total": [],
        "Glascow coma scale verbal response": [],
        "Glucose": [],
        "Heart Rate": [],
        "Height": [],
        "Mean blood pressure": [],
        "Oxygen saturation": [],
        "Respiratory rate": [],
        "Systolic blood pressure": [],
        "Temperature": [],
        "Weight": [],
        "pH": []

    }
    for file_name in tqdm(pd.read_csv(os.path.join(data_path, 'train_listfile.csv'))['stay']):
        data_df = pd.read_csv(os.path.join(data_path, 'train', file_name))
        length = len(data_df)
        for col_name in miss_dict.keys():
            num = data_df[col_name].count()
            miss_dict[col_name].append(1 - (num / length))

    pd.DataFrame(miss_dict).to_csv(r'')
    

def statistic_dt():
    root_dir = r''
    dt_list = []
    for file in tqdm(os.listdir(root_dir)):
        data = np.load(os.path.join(root_dir, file))
        dt_list.extend(data.tolist())
    dt_int_list = np.array(list(map(lambda x: round(x, 2), dt_list)))
    dts, counts = np.unique(dt_int_list, return_counts=True)
    pd.DataFrame({'DT': dts, 'COUNT': counts}).to_csv('./count_100.csv', index=False)
    df_plt = pd.DataFrame({'DT': dt_list})
    sns.displot(df_plt, x='DT')
    plt.show()


def random_sample():
    # inverse proportional sampling
    sample_rate = 10
    prob_scale = 10

    root_dir = r''
    save_dir = r''
    count_dict = pd.read_csv(r'').to_dict(orient='list')

    count_list = count_dict['COUNT']
    count_list = list(map(lambda x: sum(count_list)/x, count_list))
    count_list = list(map(lambda x: x/sum(count_list), count_list))
    count_dict = dict(zip(count_dict['DT'], count_list))

    file_all = []
    for root, _, file_list in os.walk(root_dir):
        file_list = list(filter(lambda x: x.endswith('_timeseries.csv'), file_list))
        file_list = list(map(lambda x: os.path.join(root, x), file_list))
        file_all.extend(file_list)
    idx_selected_dict = {}
    df_dict = {}
    while True:
        sampled_file_num = 0
        avg_sample_rate = 0
        for file_path in tqdm(file_all):
            file_name = os.path.split(file_path)[-1]
            if file_name not in df_dict:
                ts_df = pd.read_csv(file_path)
                df_dict[file_name] = ts_df
            ts_df = df_dict[file_name]
            if file_name not in idx_selected_dict:
                idx_selected_dict[file_name] = [0]
            avg_sample_rate += (len(idx_selected_dict[file_name]) / len(ts_df))
            if len(idx_selected_dict[file_name]) >= (len(ts_df) * sample_rate / 100):
                sampled_file_num += 1
                continue
            for idx, row in ts_df.iterrows():
                if idx in idx_selected_dict[file_name]:
                    continue

                previous_idxes = list(filter(lambda x: x<idx, idx_selected_dict[file_name]))
                previous_idxes.sort()
                previous_time_stamp = float(ts_df['Hours'][previous_idxes[-1]])

                dt_tmp = round(row['Hours'] - previous_time_stamp, 2)
                if dt_tmp not in count_dict:
                    try:
                        select_prob = count_dict[round(dt_tmp)]
                    except:
                        select_prob = 0
                else:
                    select_prob = count_dict[dt_tmp]
                if random.uniform(0, 1) <= select_prob * prob_scale:
                    idx_selected_dict[file_name].append(idx)

        avg_sample_rate = avg_sample_rate / len(file_all)
        print(avg_sample_rate)
        json.dump(idx_selected_dict, open(os.path.join(save_dir, 'last.json'), mode='w', encoding='utf8'))
        
        if avg_sample_rate*100 >= sample_rate:
            json.dump(idx_selected_dict, open(os.path.join(save_dir, f'sampled_{str(avg_sample_rate*100)}.json'), mode='w', encoding='utf8'))
        if 0 <= round(avg_sample_rate * 100, 1) % 5 <= 1:
            json.dump(idx_selected_dict, open(os.path.join(save_dir, f'{str(round(sampled_file_num / len(file_all)))}.json'), mode='w', encoding='utf8'))

        print('Sample rate: {}, Sample num: {}'.format(avg_sample_rate, sampled_file_num))
        if sampled_file_num >= len(file_all):
            break


if __name__ == '__main__':

    pass