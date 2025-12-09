# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp
from Embed import DataEmbedding, get_1d_sincos_pos_embed_from_grid
import copy

selected_ids_list = []


class Memory(nn.Module):
    """ Memory prompt
    """

    def __init__(self, num_memory, memory_dim, args=None):
        super().__init__()

        self.args = args

        self.num_memory = num_memory
        self.memory_dim = memory_dim

        self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C
        self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C

        self.memMatrix.requires_grad = True
        self.keyMatrix.requires_grad = True

        self.x_proj = nn.Linear(memory_dim, memory_dim)

        self.initialize_weights()

        print("model initialized memory")

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product

        assert x.shape[-1] == self.memMatrix.shape[-1] == self.keyMatrix.shape[-1], "dimension mismatch"

        x_query = torch.tanh(self.x_proj(x))

        att_weight = F.linear(input=x_query, weight=self.keyMatrix)  # [N,C] by [M,C]^T --> [N,M]

        att_weight = F.softmax(att_weight, dim=-1)  # NxM

        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]

        return dict(out=out, att_weight=att_weight)


class MemoryCond(nn.Module):
    """
    记忆模块：专门学习 cond -> theory_cal 的物理映射
    输入:
      cond:      (B, T, C_cond)  已做过准确映射的低维特征（比如 Hata 参数）
      theory:    (B, T, 1)       对应的理论值(标量)
    输出:
      prompt_D:  (B, T, D)       注入主干的 prompt
      aux:       包含 aux 损失与中间量
    """
    def __init__(self, num_memory, key_dim, val_dim, D_prompt, args=None):
        super().__init__()
        self.M = num_memory
        self.Ck = key_dim     # key 维
        self.Cv = val_dim     # value 维
        self.D  = D_prompt    # 输出到主干的 prompt 维（=主干隐藏维）

        # memory keys/vals
        self.keyMatrix = nn.Parameter(torch.zeros(self.M, self.Ck))  # [M,Ck]
        self.memMatrix = nn.Parameter(torch.zeros(self.M, self.Cv))  # [M,Cv]
        nn.init.trunc_normal_(self.keyMatrix, std=0.02)
        nn.init.trunc_normal_(self.memMatrix, std=0.02)

        # 查询映射（把 cond -> key 空间）；若 C_cond==Ck 也建议保留做数值对齐
        # 允许 C_cond != Ck
        self.q_map = nn.Linear(key_dim, self.Ck, bias=True)  # 提前创建

        # 值侧两个头：
        # 1) 预测理论值：Cv -> 1（监督：theory_cal）
        self.val_to_scalar = nn.Linear(self.Cv, 1)

        # 2) 输出给主干的 prompt：Cv -> D
        self.val_to_prompt = nn.Linear(self.Cv, self.D)
      

        # 损失权重
        self.aux_weight = getattr(args, "aux_weight", 0.1) if args else 0.1

        # 可选：平滑（时间维）
        self.smooth = getattr(args, "mem_smooth", 0.0) if args else 0.0
        if self.smooth > 0:
            self.temporal_smooth = nn.Conv1d(self.Cv, self.Cv, kernel_size=3, padding=1, groups=self.Cv)


    @staticmethod
    def _zscore(x, dim=-1, eps=1e-6):
        mu = x.mean(dim=dim, keepdim=True); sd = x.std(dim=dim, keepdim=True)
        return (x - mu) / (sd + eps)

    def forward(self, cond, theory=None, use_detach_for_main=True, return_aux=True):
        """
        cond:   (B, T, C_cond)
        theory: (B, T, 1)
        """
        B, T, C_cond = cond.shape
        

        # 查询到 key 空间
        q = cond.reshape(B*T, C_cond)          # (B*T, C_cond)
        q_proj = self.q_map(q)                 # (B*T, Ck)

        # 归一化 + 温度缩放 的注意力
        qn = F.normalize(q_proj, dim=-1)       # (B*T, Ck)
        kn = F.normalize(self.keyMatrix, dim=-1)  # (M, Ck)
        logits = (qn @ kn.t()) / (self.Ck ** 0.5)   # (B*T, M)
        att = torch.softmax(logits, dim=-1)    # (B*T, M)

        # 取值向量（未投到 D 前）
        val = att @ self.memMatrix             # (B*T, Cv)
        val = val.view(B, T, self.Cv)          # (B, T, Cv)
        att = att.view(B, T, self.M)           # (B, T, M)


        # 面向主干的 prompt（映射到 D）
        prompt_D_raw = self.val_to_prompt(val)        # (B, T, D)

        aux = None
        if return_aux and (theory is not None):
            # 用值向量回归理论标量
            theory_hat = self.val_to_scalar(val)      # (B, T, 1)
            # 形状一致性（可选）：把二者 z-score 后做 cosine
            Vz = self._zscore(val)                    # (B, T, Cv)
            # 将理论值也升到 Cv 维再比较（简单做法：重复；也可以加一个 1->Cv 的投影头）
            Th_cv = theory.expand(-1, -1, self.Cv)    # (B, T, Cv)
            Tz = self._zscore(Th_cv)                  # (B, T, Cv)
            cos = F.cosine_similarity(Vz, Tz, dim=-1).mean()
            L_cos = 1.0 - cos

            # 数值贴合：预测理论值 vs 真值
            L_val = F.smooth_l1_loss(theory_hat, theory)

            aux_loss = L_val #+ 0.1 * L_cos  # 比重可调

            aux = {
                "loss_phys": aux_loss,
                "theory_hat": theory_hat,
                "val_raw": val,             # 训练记忆参数的载体
            }

        # 给主干的 prompt 是否隔离梯度
        prompt_D = prompt_D_raw.detach() if use_detach_for_main else prompt_D_raw

        return {"prompt_D": prompt_D, "att": att, "aux": aux}





def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # print(bsz, seq_len, h)
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.pretraining_tp = pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            # print(self.up_proj.weight.size(), self.down_proj.weight.size())
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=-1)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, embed_dim, mlp_ratio=4, num_experts=16, num_experts_per_tok=2, n_shared_experts=2,
                 pretraining_tp=2):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList(
            [MoeMLP(hidden_size=embed_dim, intermediate_size=mlp_ratio * embed_dim, pretraining_tp=pretraining_tp) for _
             in range(num_experts)])
        self.gate = MoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.n_shared_experts = n_shared_experts

        if self.n_shared_experts is not None:
            intermediate_size = embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size=embed_dim, intermediate_size=intermediate_size,
                                         pretraining_tp=pretraining_tp)

    def forward(self, hidden_states, t):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        # if t[0] == 0 :
        #     # print(topk_idx.tolist(), print(len(topk_idx.tolist())))
        #     global selected_ids_list
        #     selected_ids_list.append(topk_idx.tolist())

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).float()
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size1, num_heads, mlp_ratio=4, num_experts=8, num_experts_per_tok=4, n_shared_experts=4,
                 pretraining_tp=2, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size1, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size1, num_heads=num_heads, qkv_bias=True, attn_drop=0, proj_drop=0,
                              **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size1, elementwise_affine=False, eps=1e-6)
        # --------------
        self.time_layer = get_torch_trans(heads=num_heads, layers=1, channels=hidden_size1)
        # ------------

        mlp_hidden_dim = int(hidden_size1 * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size1, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.moe = SparseMoeBlock(hidden_size1, mlp_ratio, num_experts, num_experts_per_tok, n_shared_experts,
                                  pretraining_tp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size1, 6 * hidden_size1, bias=True)
        )
        self.hide = hidden_size1

    # def forward(self, x, c, prompt_period0,  prompt_period1, prompt_period2):
    def forward(self, x, c, t):
        # c = c.type_as(self.adaLN_modulation.weight)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))


        x = x + gate_mlp * self.moe(modulate(self.norm2(x), shift_mlp, scale_mlp), t)

        # x = self.time_layer((x+c).permute(1,0,2)).permute(1,0,2)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PositionalEncodingSinCos(torch.nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 4096):
        super().__init__()
        assert hidden_size % 2 == 0, "hidden_size 必须为偶数以适配 sin/cos 拼接"
        self.hidden_size = hidden_size
        self.max_len = max_len
        # 预计算到 max_len，注册为 buffer（不可训练）
        pe = self._build_sincos(max_len, hidden_size)  # [max_len, hidden_size]
        self.register_buffer("pe", pe, persistent=True)  # [L, D]

    @staticmethod
    def _build_sincos(L: int, D: int) -> torch.Tensor:
        # 等价于 numpy 版本，但全 torch、可直接放到目标 device
        pos = torch.arange(L, dtype=torch.float32).unsqueeze(1)  # [L,1]
        div = torch.arange(D // 2, dtype=torch.float32) / (D // 2)  # [D/2]
        omega = 1.0 / (10000 ** div)  # [D/2]
        angles = pos @ omega.unsqueeze(0)  # [L, D/2]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [L, D]
        return emb  # float32

    def forward(self, batch_size: int, T: int, device=None, dtype=None):
        # 若需要更长的序列，动态扩容一次（仅当 T>max_len 时）
        if T > self.pe.size(0):
            with torch.no_grad():
                new_pe = self._build_sincos(T, self.hidden_size).to(self.pe.device)
            self.pe = new_pe  # 会自动留在 buffer 上
            self.max_len = T

        pe_t = self.pe[:T]  # [T, D]
        if device is not None:
            pe_t = pe_t.to(device)
        if dtype is not None:
            pe_t = pe_t.to(dtype)

        # 返回 [B, T, D]，使用 expand 避免拷贝
        return pe_t.unsqueeze(0).expand(batch_size, -1, -1)


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            args=None,
            patch_size=1,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4,
            learn_sigma=False,
    ):
        super().__init__()
        self.save_counter = 0
        self.learn_sigma = learn_sigma
        self.in_channels = patch_size
        self.out_channels = patch_size
        self.num_heads = num_heads
        self.args = args
        self.hidden_size = hidden_size
        self.Embedding = DataEmbedding(1, self.hidden_size, args=self.args, size1=7, size2=24)
        self.Embedding_half = DataEmbedding(1, self.hidden_size, args=self.args, size1=7, size2=48)
        self.Embedding_rsrp = DataEmbedding(1, self.hidden_size, args=self.args, size1=1, size2=96)
        self.Embedding_rsrp2 = DataEmbedding(1, self.hidden_size, args=self.args, size1=1, size2=96)

        self.enc_memory_freq0 = Memory(num_memory=args.num_memory, memory_dim=hidden_size, args=self.args)

        self.enc_memory_freq1 = Memory(num_memory=args.num_memory, memory_dim=hidden_size, args=self.args)

        self.enc_memory_freq2 = Memory(num_memory=args.num_memory, memory_dim=hidden_size, args=self.args)

        self.enc_memory_skg = Memory(num_memory=args.num_memory, memory_dim=hidden_size, args=self.args)
        
        self.enc_memory_cond = MemoryCond(num_memory = getattr(self.args, "mem_slots", hidden_size),key_dim= getattr(self.args, "mem_key_dim", hidden_size),val_dim    = getattr(self.args, "mem_val_dim", hidden_size),D_prompt=hidden_size,args= self.args)


        self.t_embedder = TimestepEmbedder(hidden_size)


        self.pos_enc = PositionalEncodingSinCos(hidden_size=self.hidden_size, max_len=168)

        self.proj_x = Conv1d_with_init(1, hidden_size, kernel_size=1)
        self.proj_mask = Conv1d_with_init(1, hidden_size, kernel_size=1)
        self.proj_obs = Conv1d_with_init(1, hidden_size, kernel_size=1)

        self.proj_perid1 = Conv1d_with_init(1, hidden_size, kernel_size=1)
        self.proj_perid2 = Conv1d_with_init(1, hidden_size, kernel_size=1)
        self.proj_perid3 = Conv1d_with_init(1, hidden_size, kernel_size=1)

        self.x_obs_projection = Conv1d_with_init(3, hidden_size, kernel_size=1)

        self.cond_pro = nn.Linear(256, hidden_size)
        self.cond_pro2 = nn.Linear(256, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights_trivial()
        self.mha0 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)


    def initialize_weights_trivial(self):


        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.weekday_embed.weight.data, std=0.02)
        w = self.Embedding.temporal_embedding.timeconv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.trunc_normal_(self.Embedding_half.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding_half.temporal_embedding.weekday_embed.weight.data, std=0.02)
        w2 = self.Embedding_half.temporal_embedding.timeconv.weight.data
        torch.nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        
        torch.nn.init.trunc_normal_(self.Embedding_rsrp.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding_rsrp.temporal_embedding.weekday_embed.weight.data, std=0.02)
        w3 = self.Embedding_rsrp.temporal_embedding.timeconv.weight.data
        torch.nn.init.xavier_uniform_(w3.view([w3.shape[0], -1]))
        
        torch.nn.init.trunc_normal_(self.Embedding_rsrp2.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding_rsrp2.temporal_embedding.weekday_embed.weight.data, std=0.02)
        w4 = self.Embedding_rsrp2.temporal_embedding.timeconv.weight.data
        torch.nn.init.xavier_uniform_(w4.view([w4.shape[0], -1]))

        # # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:  # Check if elementwise_affine is True
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        T = self.args.seq_len
        t = T // self.args.t_patch_size
        sigma_split = 2 if self.learn_sigma else 1

        x = x.reshape(x.shape[0], t, self.args.t_patch_size, sigma_split).permute(0, 3, 1, 2)
        imgs = x.reshape(x.shape[0], sigma_split, T)
        return imgs

    def get_weights_sincos(self, num_t_patch):

        pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_t_patch, self.hidden_size)
        )

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(pos_embed_temporal.shape[-1],
                                                             np.arange(num_t_patch, dtype=np.float32))

        pos_embed_temporal.data.copy_(torch.from_numpy(pos_temporal_emb).float().unsqueeze(0))

        pos_embed_temporal.requires_grad = False

        return pos_embed_temporal, copy.deepcopy(pos_embed_temporal)

    def pos_embed_enc(self, batch, input_size):

        pos_embed_temporal, _ = self.get_weights_sincos(input_size)

        pos_embed = pos_embed_temporal

        pos_embed = pos_embed.expand(batch, -1, -1)

        return pos_embed

    def forward(self, x, cond, mask_origin, t, datatype, theory_cal, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        noise = t
        N, _, T = x.shape
        if '30m' in datatype:
            TimeEmb = self.Embedding_half(y)
        elif 'RSRP1' in datatype:
            TimeEmb = self.Embedding_rsrp(y)
        elif 'RSRP2' in datatype:
            TimeEmb = self.Embedding_rsrp2(y)
        else:
            TimeEmb = self.Embedding(y)
        T = T // self.args.t_patch_size
        input_size = T
        pos_embed_sort = self.pos_enc(batch_size=N, T=input_size, device=x.device)  # [N, T, D]

        #####-----------------------------------------------------------------------####
        x_obs = x[:, 0].unsqueeze(1).to(dtype=torch.float32)  #
        x_noise_mask = x[:, 1].unsqueeze(1).to(dtype=torch.float32)  #


        x_mask_emb0 = F.leaky_relu(self.proj_x(x_noise_mask.to(dtype=torch.float32)).permute(0, 2, 1))
        # if noise[0] == 99:
        #     torch.save(x_obs.cpu(), f"./Interpretablility/data_tensor_input1_{self.save_counter}.pt")
        #     self.save_counter += 1


        _, L, C = x_mask_emb0.shape
        assert x_mask_emb0.shape == pos_embed_sort.shape



        x_mask_emb_comb = F.leaky_relu(
            self.x_obs_projection(torch.cat([x_noise_mask, x_obs, mask_origin], dim=1))).permute(0, 2, 1)
        # if noise[0] == 0:
        #     torch.save(x_mask_emb_comb.cpu(), f"./Interpretablility/data_tensor_output0_{self.save_counter-1}.pt")
        if 'RSRP' in datatype:
            cond1 = cond.to(dtype=torch.float32)
            cond1 = self.cond_pro(cond1)
            Y, _ = self.mha0(cond1, x_mask_emb_comb, x_mask_emb_comb)
            x_mask_emb_comb = x_mask_emb_comb + Y
            theory_cal = theory_cal.to(dtype=torch.float32)
            mem_out = self.enc_memory_cond(cond = cond1,theory = theory_cal,use_detach_for_main = True,return_aux = True)
            prompt_all = mem_out["prompt_D"]      # (B, T, D)
            aux = mem_out["aux"]                  # 包含 loss_phys
            # if t[0] == 1:
            #     torch.save(cond1.cpu(), f"./Interpretablility/data_tensor_input_{self.save_counter}.pt")
            #     torch.save(prompt_all.cpu(), f"./Interpretablility/data_tensor_memory_{self.save_counter}.pt")
            #     self.save_counter += 1

        else:
            cond1 = cond.to(dtype=torch.float32).expand(N, T, self.args.feature_size)
            cond1 = self.cond_pro2(cond1)
            prompt_periodical1 = self.periodicity_extractor(x_obs + x_noise_mask,
                                                            0)  # (B, x_channels, L) -> (B, prompt_channels, L)
            prompt_period1 = F.leaky_relu(self.proj_perid1(prompt_periodical1).permute(0, 2, 1))
            prompt_period1, attention_w1 = self.enc_memory_freq0(prompt_period1)['out'], \
            self.enc_memory_freq0(prompt_period1)['att_weight']
            # if t[0] == 0 :
            #     # print(topk_idx.tolist(), print(len(topk_idx.tolist())))
            #     global selected_ids_list
            #     selected_ids_list.append(attention_w1.tolist())

            prompt_periodical2 = self.periodicity_extractor(x_obs + x_noise_mask,
                                                            6)  # (B, x_channels, L) -> (B, prompt_channels, L)
            prompt_period2 = F.leaky_relu(self.proj_perid2(prompt_periodical2).permute(0, 2, 1))
            prompt_period2, attention_w2 = self.enc_memory_freq1(prompt_period2)['out'], \
            self.enc_memory_freq1(prompt_period2)['att_weight']

            # if t[0] == 0 :
            #     # print(topk_idx.tolist(), print(len(topk_idx.tolist())))
            #     global selected_ids_list
            #     selected_ids_list.append(attention_w2.tolist())

            prompt_periodical3 = self.periodicity_extractor(x_obs + x_noise_mask,
                                                            12)  # (B, x_channels, L) -> (B, prompt_channels, L)
            prompt_period3 = F.leaky_relu(self.proj_perid3(prompt_periodical3).permute(0, 2, 1))
            prompt_period3, attention_w3 = self.enc_memory_freq2(prompt_period3)['out'], \
            self.enc_memory_freq2(prompt_period3)['att_weight']
            # if t[0] == 0 :
            #     # print(topk_idx.tolist(), print(len(topk_idx.tolist())))
            #     global selected_ids_list
            #     selected_ids_list.append(attention_w3.tolist())
            
            prompt_skg = self.enc_memory_skg(cond1)['out']
            prompt_all = prompt_period1 + prompt_period2 + prompt_period3 + prompt_skg
            aux = 0.0

            # if t[0] == 1:
            #     torch.save(cond1.cpu(), f"./Interpretablility/data_tensor_input_{self.save_counter}.pt")
            #     torch.save(prompt_skg.cpu(), f"./Interpretablility/data_tensor_memory_{self.save_counter}.pt")
            #     self.save_counter += 1



        t = self.t_embedder(t)  # (N, D)
        x_mask_emb = x_mask_emb_comb + pos_embed_sort + prompt_all
        # if noise[0] == 0:
        #     torch.save(x_mask_emb.cpu(), f"./Interpretablility/data_tensor_output1_{self.save_counter-1}.pt")
        c = t.unsqueeze(1).repeat(1, x_mask_emb.shape[1],
                                  1) + TimeEmb  # + prompt_period1+ prompt_period2+  prompt_period3 + cond_emb

        #####-----------------------------------------------------------------------####
        for block in self.blocks:
            x_mask_emb = block(x_mask_emb, c, noise)  # (N, T, D)
        # if noise[0] == 0:
        #     torch.save(x_mask_emb.cpu(), f"./Interpretablility/data_tensor_output2_{self.save_counter-1}.pt")

        x = self.final_layer(x_mask_emb, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)


        return x, mask_origin, aux 

    def periodicity_extractor(self, x, k):
        X = torch.fft.rfft(x, dim=-1)


        X_real = X.real.clone()
        X_real[..., 0] = float('-inf')


        _, top_indices = torch.topk(X_real, k=k + 6, dim=-1)


        selected_indices = top_indices[..., k:k + 6]


        mask = torch.zeros_like(X, dtype=torch.bool)
        mask.scatter_(-1, selected_indices, True)

        X_filtered = X * mask.float()

        out = torch.fft.irfft(X_filtered, dim=-1).real.to(x.dtype)
        return out


#################################################################################
#                                   DiT Configs                                  #
#################################################################################



def DiT_S_8(args=None, **kwargs):
    return DiT(args=args, num_heads=8, **kwargs)


DiT_models = { 'DiT-S/8': DiT_S_8,}


