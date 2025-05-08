import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
from ConFSMoE_submission.models import *
# from torchvision.transforms import Compose, ToTensor, Normalize
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import random
import pickle
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# device = 'cpu'


class CMU_MOSEI(Dataset):
    def __init__(self, datapath, missing_ratio=0.1, is_train=True, available_mod=None, experiment_setting='II'):
        super().__init__()
        
        if is_train or experiment_setting == 'II':
            self.available_mod = 'AIN'
        else:
            self.available_mod = available_mod
            
        self.experiment_setting = experiment_setting
        self.datapath = datapath 
        self.is_train = is_train 
        data_flag = 'train' if is_train else 'test'
        with open(os.path.join(datapath, f'aligned_50.pkl'), 'rb') as f:
            self.data = pickle.load(f)[data_flag]
            
        self.audio = self.data['audio']
        self.vision = self.data['vision']
        self.text = self.data['text']
        self.labels = self.data['classification_labels']
        self.length_data = len(self.audio)
        self.num_mod = 3
        self.ms_mask = self.generate_missing_mask(self.length_data, self.num_mod, missing_ratio).astype(bool)
        self.generate_missing_data()


    def generate_missing_data(self):
        self.audio[self.ms_mask[:, 0]] = 0
        self.vision[self.ms_mask[:, 1]] = 0
        self.text[self.ms_mask[:, 2]] = 0
        
        
    def mod_pool(self):        
        audio_pool = torch.tensor(np.stack([i for i, mask in zip(self.audio, self.ms_mask[:, 0]) if mask == 0]))
        vision_pool = torch.tensor(np.stack([i for i, mask in zip(self.vision, self.ms_mask[:, 1]) if mask == 0]))
        text_pool = torch.tensor(np.stack([i for i, mask in zip(self.text, self.ms_mask[:, 2]) if mask == 0]))
        return {'audio':audio_pool, 'vision':vision_pool, 'text':text_pool}



    def generate_missing_mask(self, n_samples=200, n_modalities=3, missing_ratio=0.1):
        mask = np.zeros((n_samples, n_modalities), dtype=int)
        if self.is_train or self.experiment_setting == 'II': # Experiment setting III: if we are in training mode, then we take 50% data off and test it with entirely modality drops.
            total_elements = n_samples * n_modalities
            missing_count = int(total_elements * missing_ratio)
            reserved_positions = {(i, np.random.choice(n_modalities)) for i in range(n_samples)}
            available_positions = [(i, j) for i in range(n_samples) for j in range(n_modalities)
                                if (i, j) not in reserved_positions]
            missing_count = min(missing_count, len(available_positions))
            selected_positions = random.sample(available_positions, missing_count)
            for i, j in selected_positions:
                mask[i, j] = 1
        
        # otherwise, we drop the entire modality.
        elif 'A' not in self.available_mod:
            mask[:, 0] = 1

        elif 'I' not in self.available_mod:
            mask[:, 1] = 1 
        
        elif 'N' not in self.available_mod:
            mask[:, 2] = 1
            
        return mask


    def __len__(self):
        return self.length_data
    
    def __getitem__(self, idx):
        au_embed = self.audio[idx]
        im_embed = self.vision[idx]
        text_embed = self.text[idx]
        label = self.labels[idx]
        mask = self.ms_mask[idx]
        return au_embed, im_embed, text_embed, mask, label




def CMU_MOSEI_collate_fn(batch):
    au_embed, im_embed, text_embed, mask, label = zip(*batch)
    au_embed, im_embed, text_embed, label =  torch.tensor(au_embed).to(torch.float32), torch.tensor(im_embed).to(torch.float32), torch.tensor(text_embed).to(torch.float32), torch.tensor(label).to(torch.float32)
    
    output = {
        'audio':au_embed,
        'vision':im_embed,
        'text':text_embed
    }
    mask = np.vstack(mask).astype(bool)
    mod_mask = {
        'audio': mask[:, 0],
        'vision':mask[:, 1],
        'text':mask[:, 2]
    }
    
    
    return output, mod_mask, label



def load_and_preprocess_data_CMU_MOSEI(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_dict = {}
    datapath, missing_ratio = args.datapath, args.missing_ratio
    
    train_dataset = CMU_MOSEI(datapath, missing_ratio, is_train=True, available_mod=args.modality, experiment_setting=args.experiment_setting)
    test_dataset = CMU_MOSEI(datapath, missing_ratio, is_train=False, available_mod=args.modality, experiment_setting=args.experiment_setting)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=CMU_MOSEI_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=CMU_MOSEI_collate_fn)
    encoder_dict['audio'] = nn.Linear(74, args.hidden_dim).to(device) 
    encoder_dict['text'] = nn.Linear(768, args.hidden_dim).to(device)
    encoder_dict['vision'] = nn.Linear(35, args.hidden_dim).to(device) 
    
    return train_dataset, encoder_dict, train_loader, test_loader






class CMU_MOSI(Dataset):
    def __init__(self, datapath, missing_ratio=0.1, is_train=True, available_mod='AIN', experiment_setting='II'):
        super().__init__()
        
        if is_train or experiment_setting == 'II':
            self.available_mod = 'AIN'
        else:
            self.available_mod = available_mod
            
        self.experiment_setting = experiment_setting
        self.datapath = datapath 
        self.is_train = is_train 
        data_flag = 'train' if is_train else 'test'
        with open(os.path.join(datapath, f'aligned_50.pkl'), 'rb') as f:
            self.data = pickle.load(f)[data_flag]
            
        self.audio = self.data['audio']
        self.vision = self.data['vision']
        self.text = self.data['text']
        self.labels = self.data['classification_labels']
        self.length_data = len(self.audio)
        self.num_mod = 3
        self.ms_mask = self.generate_missing_mask(self.length_data, self.num_mod, missing_ratio).astype(bool)
        self.generate_missing_data()


    def generate_missing_data(self):
        self.audio[self.ms_mask[:, 0]] = 0
        self.vision[self.ms_mask[:, 1]] = 0
        self.text[self.ms_mask[:, 2]] = 0
        
        
    def mod_pool(self):        
        audio_pool = torch.tensor(np.stack([i for i, mask in zip(self.audio, self.ms_mask[:, 0]) if mask == 0]))
        vision_pool = torch.tensor(np.stack([i for i, mask in zip(self.vision, self.ms_mask[:, 1]) if mask == 0]))
        text_pool = torch.tensor(np.stack([i for i, mask in zip(self.text, self.ms_mask[:, 2]) if mask == 0]))
        return {'audio':audio_pool, 'vision':vision_pool, 'text':text_pool}
    
    
    def generate_missing_mask(self, n_samples=200, n_modalities=3, missing_ratio=0.1):
        mask = np.zeros((n_samples, n_modalities), dtype=int)
        if self.is_train or self.experiment_setting == 'II': # Experiment setting III: if we are in training mode, then we take 50% data off and test it with entirely modality drops.
            total_elements = n_samples * n_modalities
            missing_count = int(total_elements * missing_ratio)
            reserved_positions = {(i, np.random.choice(n_modalities)) for i in range(n_samples)}
            available_positions = [(i, j) for i in range(n_samples) for j in range(n_modalities)
                                if (i, j) not in reserved_positions]
            missing_count = min(missing_count, len(available_positions))
            selected_positions = random.sample(available_positions, missing_count)
            for i, j in selected_positions:
                mask[i, j] = 1
        
        # otherwise, we drop the entire modality.
        elif 'A' not in self.available_mod:
            mask[:, 0] = 1

        elif 'I' not in self.available_mod:
            mask[:, 1] = 1 
        
        elif 'N' not in self.available_mod:
            mask[:, 2] = 1
            
        return mask


    def __len__(self):
        return self.length_data
    
    
    def __getitem__(self, idx):
        au_embed = self.audio[idx]
        im_embed = self.vision[idx]
        text_embed = self.text[idx]
        label = self.labels[idx]
        mask = self.ms_mask[idx]
        return au_embed, im_embed, text_embed, mask, label



def CMU_MOSI_collate_fn(batch):
    au_embed, im_embed, text_embed, mask, label = zip(*batch)
    au_embed, im_embed, text_embed, label =  torch.tensor(au_embed).to(torch.float32), torch.tensor(im_embed).to(torch.float32), torch.tensor(text_embed).to(torch.float32), torch.tensor(label).to(torch.float32)
    
    output = {
        'audio':au_embed,
        'vision':im_embed,
        'text':text_embed
    }
    mask = np.vstack(mask).astype(bool)
    mod_mask = {
        'audio': mask[:, 0],
        'vision':mask[:, 1],
        'text':mask[:, 2]
    }
    
    
    return output, mod_mask, label



def load_and_preprocess_data_CMU_MOSI(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    encoder_dict = {}
    datapath, missing_ratio = args.datapath, args.missing_ratio
    
    train_dataset = CMU_MOSI(datapath, missing_ratio, is_train=True, available_mod=args.modality, experiment_setting=args.experiment_setting)
    test_dataset = CMU_MOSI(datapath, missing_ratio, is_train=False, available_mod=args.modality, experiment_setting=args.experiment_setting)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=CMU_MOSI_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=CMU_MOSI_collate_fn)
    encoder_dict['audio'] = nn.Linear(5, args.hidden_dim).to(device) 
    encoder_dict['text'] = nn.Linear(768, args.hidden_dim).to(device)
    encoder_dict['vision'] = nn.Linear(20, args.hidden_dim).to(device) 
    
    return train_dataset, encoder_dict, train_loader, test_loader



class MIMIC4Dataset(Dataset):
    def __init__(self, raw_data, data_list, availabel_mod=None, is_train=True, mean_ecg=None, std_ecg=None, task='in-hospital-mortality'):
        super().__init__()
        self.raw_data = raw_data 
        self.data_list = data_list
        self.task = task
        if is_train:
            self.available_mod = 'INTE'
        else:
            self.available_mod = availabel_mod

        self.mean_ecg = mean_ecg
        self.std_ecg = std_ecg

    def __len__(self):
        return len(self.raw_data)
    
    
    
    def __getitem__(self, index):
        x_filename = self.data_list.loc[index, 'stay']
        x_id = x_filename.split('_')[0]
        note = self.raw_data[x_id]['note']
        image = self.raw_data[x_id]['image']
        ecg = self.raw_data[x_id]['ecg']
        irg_ts = self.raw_data[x_id]['irg_ts']
        label = self.raw_data[x_id]['label']
        nonetype = type(None)
        
        if not isinstance(note, nonetype) and 'N' in self.available_mod:
            note = note.squeeze(0)
            note_miss = False 
        else:
            note = torch.zeros((128, 768))
            note_miss = True 
        
        
        if not isinstance(image, nonetype)  and 'I' in self.available_mod:
            image = image.squeeze(0)
            image_miss = False
        else:
            image = torch.zeros((1024, 49))
            image_miss = True
        
        
        if not isinstance(ecg, nonetype)  and 'E' in self.available_mod:
            ecg = ecg.astype(np.float32)
            nan_mask = np.isnan(ecg)
            rows, cols = np.where(nan_mask)
            ecg[rows, cols] = self.mean_ecg[cols]
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
            ecg = (ecg - self.mean_ecg) / self.std_ecg
            ecg = torch.tensor(ecg)
            ecg_miss = False 
        else:
            ecg = torch.zeros((5000, 12))
            ecg_miss = True
        
        
        # because irg_ts came from
        if not isinstance(irg_ts, nonetype)  and 'T' in self.available_mod:
            irg_ts = torch.tensor(irg_ts)
            irg_miss = False
        else:
            irg_ts = torch.zeros((48, 76))
            irg_miss = True
        
        if self.task == 'length-of-stay':
            label = (label > 3*24).astype(np.int32)
        else:
            label = label.astype(np.float32)

        # best_exp = self.get_best_expert_idx(irts_data, note_embedding)
        # best_exp_idx = self.combination_to_index[best_exp]
        return image, note, irg_ts, ecg, image_miss, note_miss, irg_miss, ecg_miss, label
    
    
    def mod_pool(self):        
        sample_ids = list(self.raw_data.keys())
        image_pool = [] 
        note_pool = []
        ts_pool = []
        ecg_pool = []
        nonetype = type(None)
        for key in sample_ids:
            image = self.raw_data[key]['image']
            note = self.raw_data[key]['note']
            irg_ts = self.raw_data[key]['irg_ts']
            ecg = self.raw_data[key]['ecg']

            if not isinstance(image, nonetype):
                image_pool.append(image.squeeze().to(torch.float32))
            
            if not isinstance(note, nonetype):
                note_pool.append(note.squeeze().to(torch.float32))
            
            if not isinstance(irg_ts, nonetype):
                ts_pool.append(torch.tensor(irg_ts).to(torch.float32))
                
            if not isinstance(ecg, nonetype):
                ecg_pool.append(ecg)
        
        
        image_pool = torch.stack(image_pool)
        note_pool = pad_sequence(note_pool, batch_first=True, padding_value=0)
        ts_pool = pad_sequence(ts_pool, batch_first=True, padding_value=0)[:, :48, :]
        ecg_pool = np.stack(ecg_pool)
        
        ecg_pool = ecg_pool.astype(np.float32)
        nan_mask = np.isnan(ecg_pool)
        ecg_pool[nan_mask] = np.take(self.mean_ecg, np.where(nan_mask)[2])
        ecg_pool = np.nan_to_num(ecg_pool, nan=0.0, posinf=0.0, neginf=0.0)
        
        ecg_pool = torch.tensor(ecg_pool)
        
        output = {
            'ecg':ecg_pool,
            'image':image_pool,
            'irg_ts':ts_pool,
            'note':note_pool,
        }
        
        return output



def mimic4_collate_fn(batch):
    image, note, irg_ts, ecg, image_miss, note_miss, irg_miss, ecg_miss, labels = zip(*batch)
    irg_ts = pad_sequence(irg_ts, batch_first=True, padding_value=0)
    note = pad_sequence(note, batch_first=True, padding_value=0)
    image = torch.stack(image)
    ecg = torch.stack(ecg)
    labels = torch.tensor(labels).to(torch.float32)
    out_mask = {
            'ecg':ecg_miss,
            'image':image_miss,
            'irg_ts':irg_miss,
            'note':note_miss,
        }        
    output = {
            'ecg':ecg,
            'image':image,
            'irg_ts':irg_ts,
            'note':note,
        }
    return output, out_mask, labels




def load_and_preprocess_data_mimic4(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_dict = {}
    
    data_dir = '/home/nathan/Missing_Modality/Trust_MoE/data/mimic4'
    train_data_path = os.path.join(data_dir, args.task, 'train_complete_data.pkl')
    test_data_path = os.path.join(data_dir, args.task, 'test_complete_data.pkl')
    
    
    train_list = pd.read_csv(os.path.join(data_dir, args.task, 'train_listfile.csv'))
    test_list = pd.read_csv(os.path.join(data_dir, args.task, 'test_listfile.csv'))
    
    if args.task == 'length-of-stay':
        train_list = train_list.loc[train_list['period_length'] == 48].reset_index(drop=True)
        test_list = test_list.loc[test_list['period_length'] == 48].reset_index(drop=True)
    
    
    with open(train_data_path, 'rb') as train_file:
        train_data = pickle.load(train_file)
        
    with open(test_data_path, 'rb') as test_file:
        test_data = pickle.load(test_file)
    
    ## loop over the train set and get the mean and std for the ecg training data
    ## Just to normalize it, irg_ts is normalized while preprocessing, thanks to MedFuse
    ## Note and image is feat already from the pretrained encoder
    ## ECG is raw data and needs normalization for each column 
    ecg_list = []
    for each in train_data:
        x = train_data[each]['ecg']
        nonetype = type(None)
        if not isinstance(x, nonetype):
            ecg_list.append(x)
    
    
    
    ecg_list = np.stack(ecg_list)
    ecg_list = ecg_list.astype(np.float32)
    
    mean_ecg = np.nanmean(ecg_list, axis=(0, 1))
    nan_mask = np.isnan(ecg_list)
    ecg_list[nan_mask] = np.take(mean_ecg, np.where(nan_mask)[2])
    ecg_list = np.nan_to_num(ecg_list, nan=0.0, posinf=0.0, neginf=0.0)
    std_ecg = np.nanstd(ecg_list, axis=(0, 1))
    
    irg_data_inp = 76
    encoder_dict['note'] = nn.Linear(768, args.hidden_dim).to(device) # all hid_state are 64
    encoder_dict['irg_ts'] = IRG_TS_Encoder(irg_data_inp, args.num_patches, args.hidden_dim).to(device)
    encoder_dict['image'] = Image_Encoder(num_channel=1024, in_feat=7*7, d_hid=args.hidden_dim).to(device) # an image sample is in shape [1024, 7, 7] --> I wanna make it as [128, 128]  
    encoder_dict['ecg'] = ECG_Encoder(patch_size=64, feat_size=12, d_hid=args.hidden_dim).to(device)
    
    
    
    train_dataset = MIMIC4Dataset(train_data, train_list, is_train=True, mean_ecg=mean_ecg, std_ecg=std_ecg, task=args.task)
    test_dataset = MIMIC4Dataset(test_data, test_list, is_train=False, availabel_mod=args.modality, mean_ecg=mean_ecg, std_ecg=std_ecg, task=args.task)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=mimic4_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=mimic4_collate_fn)
    
    return train_dataset, encoder_dict, train_loader, test_loader







class MIMIC3Dataset(Dataset):
    def __init__(self, raw_data, combination_to_index):
        self.raw_data = raw_data
        self.combination_to_index = combination_to_index
        
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample = self.raw_data[idx]
        irts_data = torch.tensor(sample['reg_ts'])
        note_embedding = torch.tensor(sample['text_embedding'])
        
        
        if irts_data.shape == (1,1):
            irts_miss = True
            irts_data = torch.zeros((48, 34))
        else:
            irts_miss = False
        
        
        if note_embedding.shape == (1,1):
            note_miss = True
            note_embedding = torch.zeros((128, 768)) 
        else:
            note_miss = False
        
        
        miss_mask = {
            'irg_ts':irts_miss,
            'text':note_miss
        }
        
        label = sample['label']
        best_exp = self.get_best_expert_idx(irts_data, note_embedding)
        best_exp_idx = self.combination_to_index[best_exp]
        
        return irts_data, note_embedding, best_exp_idx, miss_mask, label


    def get_best_expert_idx(self, ts, txt):
        ts_mis = 'T' if (ts!=0).any() else ''
        nt_mis = 'N' if (txt!=0).any() else ''
        comb = 'NT' if ts_mis and nt_mis else ''
        best_exp_idx = sorted([ts_mis, nt_mis, comb], key=lambda x:len(x))[-1]
        return best_exp_idx


    def mod_pool(self, is_train=True):
        ts_pool = [torch.tensor(each['reg_ts']) for each in self.raw_data if (each['reg_ts'] != 0).any()]
        ts_pool = pad_sequence(ts_pool, batch_first=True, padding_value=0)
        text_pool = [each['text_embedding'] for each in self.raw_data if (each['text_embedding'] != 0).all()]
        for idx, each in enumerate(text_pool):
            seq_len, embed_size = each.shape
            if each.shape[0] < 256:
                text_mod = np.zeros(shape=(256, embed_size))
                text_mod[:seq_len, :] = each
                text_pool[idx] = text_mod
        text_pool = torch.tensor(np.stack(text_pool, axis=0))
        
        return {'irg_ts':ts_pool, 'text':text_pool}


def mimic3_collate_fn(batch):
    irg_ts, notes, best_exp_idx, miss_mask, labels = zip(*batch)
    # best_exp_idx = np.array(best_exp_idx)
    out_mask = {}
    keys = list(miss_mask[0].keys())
    for k in keys:
        k_list = []
        for each in miss_mask:
            k_list.append(each[k])
        out_mask[k] = k_list 
    
    irg_ts = pad_sequence(irg_ts, batch_first=True, padding_value=0)
    notes = pad_sequence(notes, batch_first=True, padding_value=0)
    labels = torch.tensor(labels).to(torch.float32)
    
    output = {
        'irg_ts':irg_ts,
        'text':notes
    }
    
    return output, out_mask, labels



def load_and_preprocess_data_mimic3(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_dict = {}
    
    data_dir = '/home/nathan/Missing_Modality/Data/mimic-iii-processed' #dont modify
    train_data_path = os.path.join(data_dir, args.task, 'train_embedding.pkl')
    test_data_path = os.path.join(data_dir, args.task, 'test_embedding.pkl')
    
    with open(train_data_path, 'rb') as train_file:
        train_data = pickle.load(train_file)
        
    with open(test_data_path, 'rb') as test_file:
        test_data = pickle.load(test_file)
    
    irg_data_inp = test_data[0]['reg_ts'].shape[-1]
    encoder_dict['text'] = nn.Linear(768, args.hidden_dim).to(device) # all hid_state are 64
    encoder_dict['irg_ts'] = IRG_TS_Encoder(irg_data_inp, args.num_patches, args.hidden_dim).to(device)
    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index

    train_dataset = MIMIC3Dataset(train_data, combination_to_index)
    test_dataset = MIMIC3Dataset(test_data, combination_to_index)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=mimic3_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=mimic3_collate_fn)
    
    
    return train_dataset, encoder_dict, train_loader, test_loader



# Updated: full modality index is 0.
def get_modality_combinations(modalities):
    all_combinations = []
    for i in range(len(modalities), 0, -1):
        comb = list(combinations(modalities, i))
        all_combinations.extend(comb)
    
    # Create a mapping dictionary
    combination_to_index = {''.join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)}
    return combination_to_index
