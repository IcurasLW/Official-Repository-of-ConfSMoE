import os
import torch
import numpy as np
import argparse
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from copy import deepcopy
from tqdm import tqdm, trange
from utils import seed_everything, setup_logger
from data import *
from ConFSMoE_submission.models import ConfSMoE
import warnings
import torch.nn.functional as F
import logging 
import os 


warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")


# Utility function to convert string to bool
def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')


# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='ConfNet')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--note', type=str, default='conf_score')
    parser.add_argument('--data', type=str, default='CMU_MOSI')
    parser.add_argument('--modality', type=str, default='AIN') # I G C B for ADNI, L N C for MIMIC4, TN for MIMIC3
    parser.add_argument('--initial_filling', type=str, default='mean') # None mean
    parser.add_argument('--task', type=str, default='in-hospital-mortality'), # in-hospital-mortality, phenotyping, length-of-stay##########
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=2) # Number of Routers
    parser.add_argument('--num_patches', type=int, default=1) # Number of Patches for Input Token
    parser.add_argument('--num_experts', type=int, default=4) # Number of Experts
    parser.add_argument('--num_layers_enc', type=int, default=1) # Number of MLP layers for encoders
    parser.add_argument('--num_layers_fus', type=int, default=1) # Number of MLP layers for fusion model
    parser.add_argument('--num_layers_pred', type=int, default=1) # Number of MLP layers for prediction head
    parser.add_argument('--num_heads', type=int, default=4) # Number of heads
    parser.add_argument('--num_workers', type=int, default=8) # Number of workers for DataLoader
    parser.add_argument('--pin_memory', type=str2bool, default=True) # Pin memory in DataLoader
    parser.add_argument('--use_common_ids', type=str2bool, default=False) # Use common ids across modalities    
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--save', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--n_runs', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--label', type=int, default=3) #### 2: in-hospital-mortality, 25: phenotyping, 2: length-of-stay
    parser.add_argument('--multilabel', type=bool, default=False) #### True: phenotyping, False: other tasks
    parser.add_argument('--TokenLevelConf', type=str2bool, default='False')
    parser.add_argument('--datapath', type=str, default='/home/nathan/Missing_Modality/Trust_MoE/data/CMU-MOSI/Processed/')
    parser.add_argument('--missing_ratio', type=float, default=0.5)
    parser.add_argument('--experiment_setting', type=str, default='II')
    return parser.parse_known_args()



def run_epoch(args, train_dataset, loader, encoder_dict, modality_dict, fusion_model, criterion, device, is_train=False, optimizer=None, gate_loss_weight=0.0):
    all_preds = []
    all_labels = []
    all_probs = []
    task_losses = []
    gate_losses = []
    
    if is_train:
        fusion_model.train()
        for encoder in encoder_dict.values():
            encoder.train()
    else:
        fusion_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()


    mod_pool = train_dataset.mod_pool()
    exp_count = {exp:0 for exp in range(args.num_experts)}
    exp_conf = {exp:[] for exp in range(args.num_experts)}
    
    for batch_samples, miss_mask, batch_labels in tqdm(loader):
        batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
        batch_labels = batch_labels.to(device, non_blocking=True)
        
        fusion_input = []
        mod_mask = []
        for mod_name, mod in batch_samples.items():
            batch_size = mod.shape[0]
            mask = miss_mask[mod_name]
            mask = np.array(mask)  # mask, 0: no missing, 1: missing

            mod_embd = encoder_dict[mod_name](mod[~mask])
            seq_len, embed_size = mod_embd.shape[1], mod_embd.shape[2]
            seq_len = args.seq_len
            mod_out = torch.zeros(size=(batch_size, seq_len, embed_size)).to(device)
            mod_out[~mask] = mod_embd[:, :seq_len, :] # encode those existing data
            if (mask).sum() > 0:
                # Randomly select the sample as a embedding template
                mod_pool_i = mod_pool[mod_name]
                num_miss = np.count_nonzero(mask)
                sele_k = torch.randint(0, len(mod_pool_i), (num_miss, 10))
                sele_k = sele_k.to(torch.long)
                miss_embed = mod_pool_i[sele_k].to(device).to(torch.float32)
                miss_embed = encoder_dict[mod_name](miss_embed.mean(dim=1))
                mod_out[mask] = miss_embed[:, :seq_len, :]
            fusion_input.append(mod_out)
            mod_mask.append(mask)
        
        outputs, conf_loss = fusion_model(*fusion_input, mod_mask=miss_mask, y_label=batch_labels, is_train=is_train)
        
        if is_train:
            optimizer.zero_grad()
            batch_labels = batch_labels.type(torch.LongTensor) # <---- Here (casting)
            batch_labels = batch_labels.to(device)
            task_loss = criterion(outputs, batch_labels)
            task_losses.append(task_loss.item())
            loss = task_loss + conf_loss
            loss.backward()
            optimizer.step()
        else:
            if 'pheno' in args.task:
                preds = (F.sigmoid(outputs) > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(F.sigmoid(outputs).detach().cpu().numpy())
            else:
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
    
    if is_train:
        return np.mean(task_losses), exp_count, exp_conf
    else:
        all_probs = np.vstack(all_probs)
        return all_preds, all_labels, all_probs



def train_and_evaluate(args, seed):
    seed_everything(seed)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    if args.data == 'mimic3':
        modality_dict = {'clinical': 0, 'notes': 1}
        train_dataset, encoder_dict, train_loader, test_loader = load_and_preprocess_data_mimic3(args)
        
    elif args.data == 'mimic4':
        modality_dict = {'image': 0, 'note': 1, 'irg_ts':2, 'ecg':3}
        train_dataset, encoder_dict, train_loader, test_loader = load_and_preprocess_data_mimic4(args)
        
    elif args.data == 'CMU_MOSI':
        modality_dict = {'audio':0, 'vision':1, 'text':2}
        train_dataset, encoder_dict, train_loader, test_loader = load_and_preprocess_data_CMU_MOSI(args)
    
    elif args.data == 'CMU_MOSEI':
        modality_dict = {'audio':0, 'vision':1, 'text':2}
        train_dataset, encoder_dict, train_loader, test_loader = load_and_preprocess_data_CMU_MOSEI(args)
        
    args.n_full_modalities = len(modality_dict)
    fusion_model = ConfSMoE(args.n_full_modalities, args.seq_len, args.hidden_dim, 
                            args.label, args.num_layers_fus, args.num_layers_pred, args.num_experts, 
                            args.num_routers, args.top_k, args.num_heads, args.dropout, args.multilabel, TokenLevelConf=args.TokenLevelConf).to(device)
    
    params = list(fusion_model.parameters()) + [param for encoder in encoder_dict.values() for param in encoder.parameters()]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if args.multilabel:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)
    
    best_test_f1 = 0.0
    for epoch in range(args.train_epochs):
        fusion_model.train()
        for encoder in encoder_dict.values():
            encoder.train()
            
        task_losses, exp_count, exp_conf = run_epoch(args, train_dataset, train_loader, encoder_dict, modality_dict, fusion_model, criterion, device, optimizer=optimizer, gate_loss_weight=args.gate_loss_weight, is_train=True)
        
        ## Validation
        fusion_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

        with torch.no_grad():
            test_preds, test_labels, test_probs = run_epoch(args, train_dataset, test_loader, encoder_dict, modality_dict, fusion_model, criterion, device, is_train=False)
        
        if args.label == 2:
            test_acc = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds)
            test_auc = roc_auc_score(test_labels, test_probs[:, 1])
        elif args.multilabel:
            test_acc = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average='macro')
            test_auc = roc_auc_score(test_labels, test_probs, average='macro')
        else:
            test_acc = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average='macro')
            test_auc = roc_auc_score(test_labels, test_probs, average='macro', multi_class='ovr')

        if test_f1 > best_test_f1:
            print(f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {test_acc*100:.2f}, Val F1: {test_f1*100:.2f}, Val AUC: {test_auc*100:.2f}")
            best_test_acc = test_acc
            best_val_f1 = test_f1
            best_val_auc = test_auc
            
        logging.info(f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}] Task Loss: {task_losses:.2f} / Val Acc: {test_acc*100:.2f}, Val F1: {test_f1*100:.2f}, Val AUC: {test_auc*100:.2f}")
        torch.cuda.empty_cache()
    
    
    return best_test_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc



def main():
    args, _ = parse_args()
    logger = setup_logger('./logs', f'{args.data}', f'{args.modality}.txt')
    
    if args.data == 'CMU_MOSI' or args.data == 'CMU_MOSEI':
        logging.basicConfig(level=logging.INFO, filename=f'./ConfNet_{args.data}_{args.modality}_{args.missing_ratio}_output.txt')
    else:
        logging.basicConfig(level=logging.INFO, filename=f'./ConfNet_{args.data}_{args.modality}_{args.task}output.txt')
        
    logging.info(args)
    
    
    seeds = np.arange(args.n_runs) # [0, 1, 2]
    val_accs = []
    val_f1s = []
    val_aucs = []
    test_accs = []
    test_f1s = []
    test_aucs = []
    
    log_summary = "======================================================================================\n"
    
    model_kwargs = {
        "model": 'FlexMoE',
        "modality": args.modality,
        "initial_filling": args.initial_filling,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "warm_up_epochs": args.warm_up_epochs,
        "num_experts": args.num_experts,
        "num_routers": args.num_routers,
        "top_k": args.top_k,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_patches": args.num_patches,
        "gate_loss_weight": args.gate_loss_weight,
        "multilable":args.multilabel
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print('Modality:', args.modality)

    for seed in seeds:
        train_and_evaluate(args, seed)

if __name__ == '__main__':
    main()