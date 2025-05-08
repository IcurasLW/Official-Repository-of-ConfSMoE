import torch
import torch.nn.functional as F
import torch.nn as nn
from fmoe.transformer import _Expert
from fmoe.layers import FMoE, _fmoe_general_global_forward, mark_module_parallel_comm
from fmoe.functions import ensure_comm, Slice, AllGather
from fmoe.gates import NaiveGate
import tree
from fmoe.gates import NoisyGate


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConfNework(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_size = in_feature
        self.out_size = out_feature
        self.linear = nn.Sequential(
                            nn.LayerNorm(in_feature),
                            nn.Linear(in_feature, out_feature),
                            nn.Sigmoid()
                            )
        
    def forward(self, x):
        x = self.linear(x)
        return x



class Expert_NN(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.ln1 = nn.Linear(in_feature, out_feature)
        self.act = nn.GELU()
        self.ln2 = nn.Linear(in_feature, out_feature)

    def forward(self, x):
        x = self.ln1(x)
        x = self.act(x)
        x = self.ln2(x)
        return x 
        
        
class FixedFMoE(nn.Module):
    def __init__(self, num_expert=32, d_model=1024, world_size=1, mp_group=None,
                 slice_group=None, moe_group=None, top_k=2, gate=NaiveGate, 
                 expert=None, gate_hook=None, mask=None, mask_dict=None, multilabel=False, num_labels=2):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        self.slice_group = slice_group
        
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()
            
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert_NN(d_model, d_model) for _ in range(num_expert)])
        self.prob_layer = nn.Linear(d_model, num_labels)
        self.conf_net = nn.ModuleList([ConfNework(d_model, 1) for _ in range(num_expert)])
        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group
        self.multilabel = multilabel
        
        
    def forward(self, moe_inp, y_label=None, is_train=True, task='fuse'):
        
        
        # moe_inp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_inp))
        # assert all([batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]), "MoE inputs must have the same batch size"
        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)
            tree.map_structure(ensure_comm_func, moe_inp)

        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)
            moe_inp = tree.map_structure(slice_func, moe_inp)
        gate_top_k_idx, gate_score = self.gate(moe_inp) ##### AddtionalNoisyGate
        gate_top_k_idx = gate_top_k_idx.view(moe_inp.shape[0], self.top_k)
        self.gate.set_topk_indicates(gate_top_k_idx)
        moe_out = torch.zeros(moe_inp.shape[0], self.top_k, moe_inp.shape[1]).to(device)
        conf_score = torch.zeros(moe_inp.shape[0], self.top_k, 1).to(device)
        gate_score = gate_score.view(-1, 1, self.top_k)
        for i in range(self.num_expert):
            token_idx = torch.nonzero(gate_top_k_idx.view(-1) == i) % moe_inp.shape[0]
            token_position = (gate_top_k_idx == i).nonzero(as_tuple=True) # token idx send to exper i
            exp_embed = self.experts[i](moe_inp[token_idx]).squeeze()
            moe_out[token_position] = exp_embed
            conf_score[token_position] = self.conf_net[i](exp_embed)

        conf_score = conf_score.squeeze()
        ## learn the conf network
        if is_train:
            if self.multilabel:
                prob_GT = F.sigmoid(self.prob_layer(moe_out))
            else:
                prob_GT = F.softmax(self.prob_layer(moe_out), dim=-1) # [Num_token, num_expert, label_conf]
            return moe_out, conf_score, prob_GT
        else:
            return moe_out, conf_score, torch.tensor(0)
        



class AddtionalNoisyGate(NoisyGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(d_model, num_expert, world_size, top_k)
        self.topk_logits = []
        self.indicates = None
        self.is_full_modality = False

    def set_topk_logit(self, logit):
        self.topk_logits.append(logit)
    
    def get_topk_logit(self, clear = True):
        topk_logit = self.topk_logits
        if clear:
            self.topk_logits = None
        return topk_logit

    def set_topk_indicates(self, indicate):
        self.indicates = indicate
        
    def get_topk_indicate(self, clear = True):
        topk_indicate = self.indicates
        if clear:
            self.indicates = None
        return topk_indicate
    
    def set_loss(self, loss):
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss
    
    def set_full_modality(self, is_full_modality):
        self.is_full_modality = is_full_modality

    def forward(self, inp):
        clean_logits = inp @ self.w_gate
        raw_noise_stddev = inp @ self.w_noise
        noise_stddev = (
            self.softplus(raw_noise_stddev) + self.noise_epsilon
        ) * self.training
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
        loss = 0

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        
        # self.set_topk_logit(top_k_indices.detach())

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.top_k < self.tot_expert and self.training:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            )
        else:
            load = self._gates_to_load(gates)
        
        
        load = load.sum(0) if self.training else load
        importance = gates.sum(0) if self.training else gates.sum(0)
        importance = importance.unsqueeze(-1)
        load = load.unsqueeze(-1)
        loss_1 = self.cv_squared(importance)
        loss_2 = self.cv_squared(load).to(device).item()
        loss = loss + loss_1 + loss_2
        
        # self.set_loss(loss)
        
        return (
            top_k_indices.contiguous().view(-1),
            top_k_gates.contiguous().unsqueeze(1),
        )



class MoEFeedForward(nn.Module):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        gate=AddtionalNoisyGate, # NaiveGate
        world_size=1,
        top_k=2,
        multilabel=False,
        num_labels=2,
        TokenLevelConf=True,
        **kwargs
    ):

        super().__init__()
        self.top_k = top_k
        self.multilabel = multilabel
        self.num_labels = num_labels
        self.world_size = world_size
        self.num_expert = num_expert 
        self.d_model = d_model
        
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert_NN(d_model, d_model) for _ in range(num_expert)])
        self.prob_layer = nn.Linear(d_model, num_labels)
        self.conf_net = nn.ModuleList([ConfNework(d_model, 1) for _ in range(num_expert)])
        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.multilabel = multilabel
        self.find_token_conf = TokenLevelConf


    def forward(self, inp, mask, y_label=None, is_train=True):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        missing_inp = inp[mask] # missing modality info
        inp = inp[~mask] # existing modality # send those modality no missing
        select_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        y_label = y_label[~mask]
        output = torch.zeros(original_shape[0], original_shape[1], self.top_k, original_shape[-1]).to(device)
        # Fuse-by-gate
        gate_top_k_idx, gate_score = self.gate(inp) ##### AddtionalNoisyGate
        gate_top_k_idx = gate_top_k_idx.view(inp.shape[0], self.top_k)
        # self.gate.set_topk_indicates(gate_top_k_idx)
        
        moe_out = torch.zeros(inp.shape[0], self.top_k, inp.shape[1]).to(device)
        out_conf_score = torch.zeros(original_shape[0], original_shape[1], self.top_k, 1).to(device) # conf_score have to be the same shape as output
        conf_score = torch.zeros(select_shape[0] * select_shape[1], self.top_k, 1).to(device)
        out_prob_GT = torch.zeros(original_shape[0], select_shape[1], self.top_k, 1).to(device)
        gate_score = gate_score.view(-1, self.top_k, 1)
        
        for i in range(self.num_expert):
            token_idx = torch.nonzero(gate_top_k_idx.view(-1) == i) % inp.shape[0]
            if token_idx.shape[0] == 0:
                continue
            
            token_position = (gate_top_k_idx == i).nonzero(as_tuple=True) # token idx send to exper i
            exp_embed = self.experts[i](inp[token_idx]).squeeze(1)
            moe_out[token_position] = exp_embed
            
            
            if not self.find_token_conf:
                conf_score[token_position] = self.conf_net[i](exp_embed).mean()
            else:
                conf_score[token_position] = self.conf_net[i](exp_embed)
            
            
        conf_score = conf_score.view(select_shape[0], select_shape[1], self.top_k, 1)
        moe_out = moe_out.view(select_shape[0], select_shape[1], self.top_k, -1)
        output[~mask] = moe_out
        out_conf_score[~mask] = conf_score
        
        # in case there is a sample with missing modality in input sample
        if not torch.nonzero(mask).all():
            output[mask] = missing_inp.unsqueeze(2).repeat(1,1,2,1)
            
        # Get the predictive confidence of Ground True
        if is_train:
            if self.multilabel:
                prob_GT = F.sigmoid(self.prob_layer(moe_out))
                y_expand = y_label.unsqueeze(1).unsqueeze(1).expand(-1,  select_shape[1], self.top_k, -1) #[batch, seq_len, topk, n_label] # because we need to select the one with True probablity
                prob_GT = prob_GT.view(select_shape[0], select_shape[1], self.top_k, -1) # [batch, seq_len, topk, n_labels]
                prob_GT = torch.where(
                                y_expand == 1,
                                torch.where(prob_GT < 0.5, 1 - prob_GT, prob_GT),
                                torch.where(prob_GT < 0.5, prob_GT, 1 - prob_GT)
                                ).mean(-1).view(-1, self.top_k)
                prob_GT = prob_GT.view(select_shape[0], select_shape[1], self.top_k, 1)
            else:
                prob_GT = F.softmax(self.prob_layer(moe_out), dim=-1) # [Num_token, num_expert, label_conf]
                y_expand = y_label.unsqueeze(-1).unsqueeze(-1).expand(-1, select_shape[1], self.top_k).to(torch.int64) #[batch, seq_len, topk]# because we need to select the one with True probablity
                prob_GT = prob_GT.view(select_shape[0], select_shape[1], self.top_k, -1) # [batch, seq_len, topk, labels]
                prob_GT = torch.gather(prob_GT, dim=3, index=y_expand.unsqueeze(-1)).view(-1, self.top_k) #[batch * seq_len, topk]
                prob_GT = prob_GT.view(select_shape[0], select_shape[1], self.top_k, 1)
        else:
            prob_GT = torch.zeros(select_shape[0], select_shape[1], self.top_k, 1).to(device)
        
        out_prob_GT[~mask] = prob_GT
        return output, out_conf_score, out_prob_GT