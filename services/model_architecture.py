import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    TARGET_FRAMES,
    D_MODEL,
    N_HEADS,
    DROPOUT,
    ATTN_DROPOUT,
    TCN_DROPOUT,
    AUX_DROPOUT,
    FUSION_DROPOUT,
    GATE_FLOOR,
    AUX_DIM,
)
from services.video_processor import A_HAND42, A_UPPER49

class MaskedAttentionPooling(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x, mask):
        score = self.score(x).squeeze(-1)
        score = score.masked_fill(mask <= 0, -1e4)
        w = torch.softmax(score, dim=1)
        w = torch.nan_to_num(w, nan=0.0)
        return torch.sum(x * w.unsqueeze(-1), dim=1), w


class GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, A, dropout=0.1):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x, node_mask=None):
        neigh = torch.einsum("ij,btjc->btic", self.A, x)
        y = self.lin(neigh)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = y + self.res(x)
        if node_mask is not None:
            y = y * node_mask.unsqueeze(-1).float()
        return y


class GraphFrameEncoder(nn.Module):
    def __init__(self, in_ch, d_model, A, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(GraphConvLayer(ch, d_model, A, dropout=dropout))
            ch = d_model
        self.layers = nn.ModuleList(layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, node_mask, frame_mask):
        for layer in self.layers:
            x = layer(x, node_mask)
        m = node_mask.float()
        denom = m.sum(dim=2, keepdim=True).clamp_min(1.0)
        pooled = (x * m.unsqueeze(-1)).sum(dim=2) / denom
        pooled = self.out_norm(pooled)
        return pooled * frame_mask.unsqueeze(-1).float()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers=2, dropout=0.1, ff_mult=4):
        super().__init__()
        self.pos = PositionalEncoding(d_model, max_len=TARGET_FRAMES + 8)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, frame_mask):
        x = self.pos(x)
        key_padding_mask = frame_mask <= 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x * frame_mask.unsqueeze(-1).float()


class MultiScaleTCNBlock(nn.Module):
    def __init__(self, dim, kernel_sizes=(3, 5, 7), dilations=(1, 2, 4), dropout=0.15):
        super().__init__()
        self.branches = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            pad = (k - 1) * d // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=k, padding=pad, dilation=d),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.mix = nn.Sequential(
            nn.Conv1d(dim * len(self.branches), dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 2, dim, kernel_size=1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, frame_mask):
        res = x
        y = x.transpose(1, 2)
        outs = [branch(y) for branch in self.branches]
        y = torch.cat(outs, dim=1)
        y = self.mix(y)
        y = y + res.transpose(1, 2)
        y = y + self.ffn(y)
        y = y.transpose(1, 2)
        y = self.norm(y)
        return y * frame_mask.unsqueeze(-1).float()


class StaticHandshapeTransformerBranch(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.hand_gcn = GraphFrameEncoder(4, d_model, A_HAND42, num_layers=2, dropout=DROPOUT)
        self.temporal = TemporalTransformerEncoder(d_model, N_HEADS, num_layers=2, dropout=ATTN_DROPOUT)
        self.pool = MaskedAttentionPooling(d_model, dropout=DROPOUT)

    def forward(self, hloc, hdom, hloc_m, hdom_m, frame_mask):
        x = torch.cat([hloc, hdom], dim=-1)
        m = ((hloc_m > 0) | (hdom_m > 0)).to(torch.uint8)
        seq = self.hand_gcn(x, m, frame_mask)
        seq = self.temporal(seq, frame_mask)
        return self.pool(seq, frame_mask)


class MultiScaleTemporalMotionBranch(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        in_dim = 42 * 2 + 42 * 2 + 49 * 2 + 49 * 2
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(TCN_DROPOUT),
        )
        self.blocks = nn.ModuleList([
            MultiScaleTCNBlock(d_model, kernel_sizes=(3, 5, 7), dilations=(1, 2, 4), dropout=TCN_DROPOUT),
            MultiScaleTCNBlock(d_model, kernel_sizes=(3, 5, 7), dilations=(2, 4, 8), dropout=TCN_DROPOUT),
            MultiScaleTCNBlock(d_model, kernel_sizes=(3, 5, 7), dilations=(1, 4, 8), dropout=TCN_DROPOUT),
        ])
        self.pool = MaskedAttentionPooling(d_model, dropout=TCN_DROPOUT)

    def forward(self, hloc_v, hdom_v, uno_v, ual_v, frame_mask):
        B, T = frame_mask.shape
        x = torch.cat([
            hloc_v.reshape(B, T, -1),
            hdom_v.reshape(B, T, -1),
            uno_v.reshape(B, T, -1),
            ual_v.reshape(B, T, -1),
        ], dim=-1)
        x = self.proj(x) * frame_mask.unsqueeze(-1).float()
        for block in self.blocks:
            x = block(x, frame_mask)
        return self.pool(x, frame_mask)


class HandBodyCrossAttentionBranch(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.hand_encoder = GraphFrameEncoder(8, d_model, A_HAND42, num_layers=2, dropout=DROPOUT)
        self.body_encoder = GraphFrameEncoder(8, d_model, A_UPPER49, num_layers=2, dropout=DROPOUT)
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, N_HEADS, dropout=ATTN_DROPOUT, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(DROPOUT),
        )
        self.temporal = TemporalTransformerEncoder(d_model, N_HEADS, num_layers=1, dropout=ATTN_DROPOUT)
        self.pool = MaskedAttentionPooling(d_model, dropout=DROPOUT)

    def forward(self, batch):
        fm = batch["frame_mask"].float()
        h = torch.cat([
            batch["X_hand_local"], batch["X_hand_dominant"],
            batch["X_hand_local_velocity"], batch["X_hand_dominant_velocity"],
        ], dim=-1)
        hm = ((batch["hand_local_mask"] > 0) | (batch["hand_dominant_mask"] > 0)).to(torch.uint8)
        b = torch.cat([
            batch["X_upper_noalign"], batch["X_upper_noalign_velocity"],
            batch["X_upper_aligned"], batch["X_upper_aligned_velocity"],
        ], dim=-1)
        bm = ((batch["upper_noalign_mask"] > 0) | (batch["upper_aligned_mask"] > 0)).to(torch.uint8)
        hand_seq = self.hand_encoder(h, hm, fm)
        body_seq = self.body_encoder(b, bm, fm)
        q = self.q_norm(hand_seq)
        kv = self.kv_norm(body_seq)
        key_padding_mask = fm <= 0
        cross, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        x = hand_seq + cross
        x = x + self.ffn(x)
        x = self.temporal(x, fm)
        return self.pool(x, fm)


class AuxReliabilityBranch(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(AUX_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(AUX_DROPOUT),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(AUX_DROPOUT),
        )
        self.temporal = MultiScaleTCNBlock(d_model, kernel_sizes=(3, 5, 7), dilations=(1, 2, 4), dropout=AUX_DROPOUT)
        self.pool = MaskedAttentionPooling(d_model, dropout=AUX_DROPOUT)

    def forward(self, aux, aux_mask, frame_mask):
        aux = aux * aux_mask.float()
        x = self.frame_mlp(aux)
        x = x * frame_mask.unsqueeze(-1).float()
        x = self.temporal(x, frame_mask.float())
        return self.pool(x, frame_mask.float())


class FourWayGatedFusion(nn.Module):
    def __init__(self, d_model=D_MODEL, num_branches=4, gate_floor=GATE_FLOOR):
        super().__init__()
        self.num_branches = num_branches
        self.gate_floor = gate_floor
        self.gate_net = nn.Sequential(
            nn.LayerNorm(d_model * num_branches),
            nn.Linear(d_model * num_branches, d_model),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(d_model, num_branches),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, feats):
        cat = torch.cat(feats, dim=-1)
        raw = torch.softmax(self.gate_net(cat), dim=-1)
        gates = self.gate_floor + (1.0 - self.num_branches * self.gate_floor) * raw
        stacked = torch.stack(feats, dim=1)
        fused = torch.sum(stacked * gates.unsqueeze(-1), dim=1)
        return self.out_norm(fused), gates


class HFSMCAHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.static_branch = StaticHandshapeTransformerBranch(D_MODEL)
        self.motion_branch = MultiScaleTemporalMotionBranch(D_MODEL)
        self.cross_branch = HandBodyCrossAttentionBranch(D_MODEL)
        self.aux_branch = AuxReliabilityBranch(D_MODEL)
        self.fusion = FourWayGatedFusion(D_MODEL, num_branches=4, gate_floor=GATE_FLOOR)
        self.classifier = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(D_MODEL, num_classes),
        )

    def forward(self, batch, return_gates=False, return_aux=False):
        fm = batch["frame_mask"].float()
        static_feat, static_attn = self.static_branch(
            batch["X_hand_local"], batch["X_hand_dominant"],
            batch["hand_local_mask"], batch["hand_dominant_mask"], fm,
        )
        motion_feat, motion_attn = self.motion_branch(
            batch["X_hand_local_velocity"], batch["X_hand_dominant_velocity"],
            batch["X_upper_noalign_velocity"], batch["X_upper_aligned_velocity"], fm,
        )
        cross_feat, cross_attn = self.cross_branch(batch)
        aux_feat, aux_attn = self.aux_branch(batch["X_aux_10A"], batch["X_aux_10A_mask"], fm)
        fused, gates = self.fusion([static_feat, motion_feat, cross_feat, aux_feat])
        logits = self.classifier(fused)
        if return_aux:
            return logits, {
                "gates": gates,
                "static_attn": static_attn,
                "motion_attn": motion_attn,
                "cross_attn": cross_attn,
                "aux_attn": aux_attn,
            }
        if return_gates:
            return logits, gates
        return logits
