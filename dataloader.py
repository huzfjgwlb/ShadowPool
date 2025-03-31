import torch
from datasets.celeba import get_celeba_dataset
from datasets.adult import get_adult_dataset
from datasets.census import get_census_dataset
from datasets.bankmk import get_bankmk_dataset
import numpy as np
import random
from torch.utils.data import Subset



def get_data_ratio(args, dataset_name, prop_name, t_value, select_num=60000):
    dataset_mapping = {
        'adult': get_adult_dataset,
        'census': get_census_dataset,
        'bankmk': get_bankmk_dataset,
    }
    
    if dataset_name == 'adult':
        args.input_dim = 111,
        args.num_classes = 2
    elif dataset_name == 'census':
        args.input_dim = 511
        args.num_classes = 2
    else:
        raise NotImplementedError()

    dataset_name = dataset_name.lower()
    
    if dataset_name in dataset_mapping:
        data_func = dataset_mapping[dataset_name]
        data, prop_data = data_func(property=prop_name)
        
        for i in range(len(data)):
            data[i] = data[i] + (prop_data[i],)

    elif dataset_name.lower() == 'celeba':
        num_classes, dataset = get_celeba_dataset('celeba', attr='attr', prop_name=prop_name, root='../data/')
        data_len = 200000
        data, _ = torch.utils.data.random_split(dataset, [data_len, len(dataset)-data_len])
        
    else:
        print("Dataset error!")
        return

    data = list(data)
    random.shuffle(data)

    prop_data = np.array([p for (_, _, p) in data])
    prop_index = np.where(prop_data == 1)[0]
    non_prop_index = np.where(prop_data == 0)[0]

    print(f'All:{len(data)} Prop:{len(prop_index)} Non-prop:{len(non_prop_index)}')

    # create training and test dataset
    prop_len = int(select_num * t_value)
    nonprop_len = int(select_num * (1 - t_value))
    if prop_len > len(prop_index) or nonprop_len > len(non_prop_index):
        print(f'Error - {prop_len}:{len(prop_index)} {nonprop_len}:{len(non_prop_index)}')
        return
    else:
        print(f'Assign - prop_len:{prop_len} nonprop_len:{nonprop_len} {dataset_name}-{prop_name}:{t_value:.2f}')

    p_index = np.random.choice(prop_index, prop_len, replace=False)
    np_index = np.random.choice(non_prop_index, nonprop_len, replace=False)
    _index = np.concatenate((p_index, np_index))
    select_data = [data[i] for i in _index]
    sampled_dataset = Subset(select_data, range(len(select_data)))

    return sampled_dataset
    
if __name__ == "__main__":
    query = {
        'adult': ['sex', 'race', 'workclass'],
        'census': ['sex', 'race', 'education'],
        'bankmk': ['month', 'marital', 'contact'],
    } 

    # census-sex: 30% vs 50%
    # get_data_ratio('census', 'sex', 0.30, 60000)
    # get_data_ratio('census', 'sex', 0.50, 60000)

    # census-race: 5% vs 15%
    # get_data_ratio('census', 'race', 0.05, 60000)
    # get_data_ratio('census', 'race', 0.15, 60000)

    # census-education: 5% vs 15%
    get_data_ratio('census', 'education', 0.05, 60000)
    get_data_ratio('census', 'education', 0.15, 60000)



    # parser.add_argument('--ratio_num', type=int, default=20000, help='num of dataset')
    # parser.add_argument('--t_value', type=float, default=0.10, help='the fraction of a target property')