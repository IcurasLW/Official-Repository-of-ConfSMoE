import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ConFSMoE_submission.moe_module import *
from itertools import combinations


device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Image_Encoder(nn.Module):
    def __init__(self, num_channel, in_feat, d_hid):
        super().__init__()
        self.channel_ln = nn.Linear(num_channel, 64)
        self.feat_ln = nn.Linear(in_feat, d_hid)
        self.channel_layernorm = nn.LayerNorm(num_channel)
        self.feat_layernorm = nn.LayerNorm(in_feat)

    def forward(self, x):
        '''
        
        x: [B, 1024, 49]
        
        '''
        
        out = self.channel_ln(self.channel_layernorm(x.permute(0, 2, 1)))
        out = self.feat_ln(self.feat_layernorm(out.permute(0, 2, 1))) # end up with [B, 64, 128]
        return out



class ECG_Encoder(nn.Module):
    def __init__(self, patch_size: int, d_hid: int, feat_size):
        super().__init__()
        self.patch_size = patch_size
        self.layernorm = nn.LayerNorm(patch_size*feat_size)
        self.unfold = nn.Unfold(kernel_size=(1, patch_size), stride=patch_size)
        self.projection = nn.Linear(patch_size*feat_size, d_hid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, S, E]  (batch, time, channels)
        """
        B, S, E = x.shape
        num_patch = math.ceil(S / self.patch_size)
        pad_size  = num_patch * self.patch_size - S
        if pad_size:
            x = F.pad(x, (0, 0, 0, pad_size))
        x = x.permute(0, 2, 1).unsqueeze(2)  
        x = self.unfold(x)              
        x = x.permute(0, 2, 1)
        x = self.layernorm(x)
        x_emb = self.projection(x)                     
        return x_emb


class IRG_TS_Encoder(nn.Module):
    def __init__(self, feature_size, patch_size, embed_dim, dropout=0.25):
        super().__init__()
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.ts_proj = nn.Linear(patch_size*self.feature_size, embed_dim)
        self.irg_proj = nn.Linear(patch_size*self.feature_size, embed_dim)
        self.layernorm = nn.LayerNorm(patch_size*self.feature_size)
        
    def forward(self, x):
        batch_size, seq_len, fea_size = x.shape
        n_patch = seq_len // self.patch_size
        pad_size = seq_len - (self.patch_size*n_patch)
        
        x = F.pad(x, (0, pad_size))
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size).reshape(batch_size, n_patch, fea_size*self.patch_size)
        ts_x = torch.tensor(x).float()
        ts_x = self.layernorm(ts_x)
        ts_x = self.ts_proj(ts_x)
        out = ts_x
        return out



class ConfSMoE(nn.Module):
    def __init__(self, num_modalities, seq_len, hidden_dim, output_dim, num_layers, num_layers_pred, num_experts, num_routers, top_k, num_heads=2, dropout=0.5, multilabel=False, TokenLevelConf=True):
        super(ConfSMoE, self).__init__()
        layers = []
        _sparse = True
        ## This is the layer --> Attention --> router --> MoE
        layers.append(TransformerEncoderLayer(num_experts, hidden_dim, num_head=num_heads, dropout=dropout, mlp_sparse=_sparse, top_k=top_k, multilabel=multilabel, num_labels=output_dim, TokenLevelConf=TokenLevelConf)) 
        for j in range(num_layers - 1):
            _sparse = not _sparse
            layers.append(TransformerEncoderLayer(num_experts, hidden_dim, num_head=num_heads, dropout=dropout, mlp_sparse=_sparse, top_k=top_k, multilabel=multilabel, num_labels=output_dim))
        layers.append(MLP(hidden_dim*num_modalities, hidden_dim, hidden_dim, num_layers_pred, activation=nn.ReLU(), dropout=0.5)) # Readout layer
        
        self.network = nn.Sequential(*layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, np.sum([seq_len]*num_modalities), hidden_dim))
        self.combination_to_index = self._create_combination_index(num_modalities)
        self.norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        
        
    def forward(self, *inputs, mod_mask, y_label=None, is_train=True):
        # global tracker
        
        chunk_size = [input.shape[1] for input in inputs]
        batch_size  = inputs[0].shape[0]
        x = torch.cat(inputs, dim=1)
        
        if self.pos_embed != None:
            x = x + self.pos_embed
        
        mod_mask = torch.tensor(list(mod_mask.values()))
        miss_mod_idx = torch.nonzero(torch.any(mod_mask, dim=0)).flatten()
        full_mod_idx = torch.nonzero(torch.all(~mod_mask, dim=0)).flatten()
        out = torch.zeros(batch_size, self.output_dim).to(device)
        
        conf_loss = 0
        out_conf_1 = []
        out_conf_2 = []
        out_prob_GT_1 = []
        out_prob_GT_2 = []
        # Full modality do modality fusion, concatenate
        if full_mod_idx.shape[0] > 0:
            full_mod_mask = mod_mask[:, full_mod_idx]
            full_sample = x[full_mod_idx]
            full_sample = torch.split(full_sample, chunk_size, dim=1)
            
            for i in range(len(self.network) - 1):   
                full_sample,  out_conf_1, out_prob_GT_1 = self.network[i](full_sample, ms_mask=full_mod_mask, task='fuse', y_label=y_label[full_mod_idx], is_train=is_train)
            full_sample = [item.mean(dim=1) for item in full_sample]
            full_sample = torch.cat(full_sample, dim=1)
            full_sample = self.network[-1](full_sample)
            full_sample = self.readout(self.norm(full_sample))
            out[full_mod_idx] = full_sample
        
        conf_loss = 0
        # missing modality do modality imputation --> fusion, concatenate
        if miss_mod_idx.shape[0] > 0:
            ### do imputat and fusion for missing modality part
            miss_sample = x[miss_mod_idx]
            miss_sample = torch.split(miss_sample, chunk_size, dim=1)
            ms_mask = mod_mask[:, miss_mod_idx]
            miss_sample, out_conf_2, out_prob_GT_2 = self.network[0](miss_sample, ms_mask=ms_mask, task='imput_fuse', y_label=y_label[miss_mod_idx], is_train=is_train) # 先进 self-attention MOE, I am trying to get the confidence of each expert.
            miss_sample = [item.mean(dim=1) for item in miss_sample]
            miss_sample = torch.cat(miss_sample, dim=1)
            miss_sample = self.network[-1](miss_sample)
            miss_sample = self.readout(self.norm(miss_sample))
            out[miss_mod_idx] = miss_sample  
        
        out_conf = torch.concat(out_conf_1 + out_conf_2, dim=0)
        out_prob_GT = torch.concat(out_prob_GT_1 + out_prob_GT_2, dim=0)
        conf_loss = F.mse_loss(out_conf, out_prob_GT)
        
        return out, conf_loss


    def gate_loss(self):
        g_loss = []
        for mn, mm in self.named_modules():
            # print(mn)
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is None:
                        print(f"[WARN] The gate loss if {mn}, modality: {i} is emtpy, check weather call <get_loss> twice.")
                    else:
                        g_loss.append(i_loss)
        return sum(g_loss)


    def _create_combination_index(self, num_modalities):
        combinations_list = []
        for r in range(1, num_modalities + 1):
            combinations_list.extend(combinations(range(num_modalities), r))
        combination_to_index = {tuple(sorted(comb)): idx for idx, comb in enumerate(combinations_list)}
        return combination_to_index


    def assign_expert(self, combination):
        index = self.combination_to_index.get(tuple(sorted(combination)))
        return index


    def set_full_modality(self, is_full_modality):
        for layer in self.network:
            if hasattr(layer, 'set_full_modality'):
                layer.set_full_modality(is_full_modality)




class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, kv, attn_mask=None):
        # attn_mask: (B, N+1, N+1) input-dependent
        eps = 1e-6

        Bx, Nx, Cx = x.shape
        B, N, C = kv.shape
        q = self.q(x).reshape(Bx, Nx, self.num_heads, Cx//self.num_heads)
        q = q.permute(0, 2, 1, 3)
        kv = self.kv(kv)
        kv = kv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N+1, C/H) @ (B, H, C/H, N+1) -> (B, H, N+1, N+1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(Bx, Nx, -1)  # (B, H, N+1, N+1) * (B, H, N+1, C/H) -> (B, H, N+1, C/H) -> (B, N+1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                num_experts,
                d_model, 
                num_head, 
                dropout=0.1, 
                activation=nn.GELU, 
                mlp_sparse = False, 
                self_attn = True,
                multilabel=True,
                top_k=2,
                num_labels=2,
                TokenLevelConf=True,
                **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.attn = Attention(d_model, num_heads=num_head, qkv_bias=False, attn_drop=dropout, proj_drop=dropout)
        self.mlp_sparse = mlp_sparse
        self.self_attn = self_attn
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.mlp = MoEFeedForward(num_expert=num_experts, d_model=d_model, top_k=top_k, multilabel=multilabel, num_labels=num_labels, TokenLevelConf=TokenLevelConf)
        
        
    def forward(self, x, attn_mask = None, ms_mask=None, task='fuse', y_label=None, is_train=True):
        # batch_size, seq_len, embed_size = x[0].shape
        # n_topk_exp = len(x)
        chunk_size = [item.shape[1] for item in x]
        x = torch.cat(x, dim=1)
        x = self.norm1(x)
        kv = x
        x = self.attn(x, kv, attn_mask)
        x = x + self.dropout1(x)
        x = torch.split(x, chunk_size, dim=1)
        x = [item for item in x]
        out_conf = []
        out_prob_GT = []
        
        if task == 'fuse': # The modality has no missing data
            for i in range(len(chunk_size)):
                output, conf_score, prob_GT = self.mlp(self.norm2(x[i]), ms_mask[i], y_label=y_label, is_train=is_train)
                out_conf.append(conf_score)
                out_prob_GT.append(prob_GT)
                
                conf_score = conf_score.view(output.shape[0], output.shape[1], -1)
                output = torch.einsum('bske,bsk->bse', output, conf_score) # fusion use the conf score, yep
                x[i] = self.layer_norm_1(x[i] + output)
            
            
        elif task == 'imput_fuse':
            # conf_scores = []
            exp_out = []
            for i in range(len(chunk_size)):
                if (~ms_mask[i]).any(): # it is True if any of data in this particular modality exists
                    # the current modality is not missing
                    # the MoE layer only forward those existing modality, the missing modality is pooled from the mod_pool
                    output, conf_score, prob_GT = self.mlp(self.norm2(x[i]), ms_mask[i], y_label=y_label, is_train=is_train)
                    out_conf.append(conf_score)
                    out_prob_GT.append(prob_GT)
                    conf_score = conf_score.view(output.shape[0], output.shape[1], -1)
                    exp_out.append(output)
                    output = torch.einsum('bske,bsk->bse', output, conf_score) # fusion use the conf score, yep
                    x[i] = self.layer_norm_2(x[i] + output)
                else:
                    exp_out.append(x[i].unsqueeze(2).repeat(1,1,self.top_k,1)) # make the same shape as other existing mod
            
            
            imput_data, imp_sam_idx = self.imput_moddality(exp_out, ms_mask)
            for i in range(len(chunk_size)):
                if imp_sam_idx[i] == None:
                    continue 
                # BUG: needs to use out-of-place operaction to fix single missing
                temp = x[i].clone()
                imput_data_i = self.layer_norm_2(x[i][imp_sam_idx[i]] + imput_data[i])
                temp[imp_sam_idx[i]] = imput_data_i
                x[i] = temp
                
        return x, out_conf, out_prob_GT
    
    
    def imput_moddality(self, x, mask):
        n_mod = len(x)
        batch_size, seq_len, n_exp, embed_size = x[0].shape
        x = torch.stack(x) # dim=0 is two modality
        output = []
        out_idx = []
        for mod in range(n_mod):
            cur_mask = mask[mod] 
            
            if (~cur_mask).all():
                # if there is nothing to be imputed
                output.append(None)
                out_idx.append(None)
                continue
            
            oth_mask = torch.cat((mask[:mod], mask[mod+1:]), dim=0)
            ms_idx = torch.nonzero(cur_mask).squeeze(-1).to(device)
            ms_sample = x[:, ms_idx]
            ms_idx = ms_idx.to('cpu')
            oth_mask = oth_mask[:, ms_idx]
            
            if ms_idx.dim() == 0:
                num_ms = 1
            else:
                num_ms = len(ms_idx)
            
            mis_embed = ms_sample[mod].mean(dim=2)
            exit_embed = torch.cat((ms_sample[:mod], ms_sample[mod+1:]), dim=0)
            exit_embed[oth_mask] = 0 # replace missing embed as zero
            
            exit_embed = exit_embed.permute(1, 0, 2, 3, 4)
            exit_embed = exit_embed.reshape(exit_embed.shape[0], -1, exit_embed.shape[-1])
            mis_embed = self.q_linear(mis_embed)
            exit_embed = self.k_linear(exit_embed)
            attn_logits = torch.matmul(mis_embed, exit_embed.transpose(-2, -1)) / embed_size**0.5
            shape_logits = attn_logits.shape
            attn_logits = attn_logits.view(shape_logits[0], shape_logits[1], int(shape_logits[2]/self.top_k), -1)
            attn_weights = F.softmax(attn_logits, dim=-2)
            
            k = (n_mod - 1)*(seq_len) // 4
            topk_weight, topk_indices = torch.topk(attn_weights, k, dim=-2)
            exit_embed = exit_embed.view(num_ms, -1, self.top_k, embed_size) # [14, 100, 2, 128]
            exit_embed = exit_embed.unsqueeze(1).repeat(1, seq_len, 1, 1, 1) # [5, 10, 10, 2, 64]
            
            topk_indices = topk_indices.unsqueeze(-1).repeat(1, 1, 1, 1, embed_size) # [14, 50, 12, 2, 12]
            topk_value = torch.gather(exit_embed, 2, topk_indices) # [14, 50, 2, 2, 128]
            out = torch.einsum('bskne,bskn->bse', topk_value, topk_weight)
            output.append(out)
            out_idx.append(ms_idx)
            
        return output, out_idx
        