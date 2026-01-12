# -*- coding: iso-8859-1 -*-
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gmp
from torch_geometric.nn import GATConv, global_add_pool as gap
from torch_geometric.utils import to_dense_batch
import math
import torch.nn.init as init
from .model_transformer import *

# torch.cuda.set_device(0)  # ����Ĭ���豸Ϊ�ڶ��� GPU

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_length):
        super(PositionalEncoding, self).__init__()
        self.embedding_size = embedding_size
        self.max_length = max_length
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_length, embedding_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x * math.sqrt(self.embedding_size)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        x = x + pe
        return x

class comModel(nn.Module):
    def __init__(self, args):
        super(comModel, self).__init__()
        self.num_features_smi = args['num_features_smi']
        self.num_features_ecfp = args['num_features_ecfp']
        self.num_features_x = args['num_features_x']
        self.dropout = args['dropout']
        self.num_layer = args['num_layer']
        self.num_heads = args['num_heads']
        self.hidden_dim = args['hidden_dim']
        self.output_dim = args['output_dim']
        self.n_output = args['n_output']
        
        self.encoder = make_model(self.num_features_smi, self.num_layer, self.hidden_dim, self.output_dim, self.num_heads, self.dropout)
        self.ep_gru = nn.GRU(self.num_features_ecfp, self.hidden_dim, self.num_layer, batch_first=True, bidirectional=True)
       
        self.smi_norm = nn.LayerNorm(self.hidden_dim)
        self.ep_norm1 = nn.LayerNorm(self.hidden_dim*2)
        self.ep_norm2 = nn.LayerNorm(self.output_dim)

        self.gcn_conv1 = GCNConv(self.num_features_x, self.num_features_x)
        self.gcn_conv2 = GCNConv(self.num_features_x, self.num_features_x * 2)
 
        self.num_attention_heads = 2
        self.ep_attention_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim)
        ) for _ in range(self.num_attention_heads)])
        self.ep_fc_layers = nn.ModuleList([nn.Linear(self.hidden_dim * 2, self.output_dim) for _ in range(self.num_attention_heads)])

        self.dropout = nn.Dropout()

        self.smi_fc1 = nn.Linear(self.num_features_smi, self.n_output)
        self.smi_fc2 = nn.Linear(self.hidden_dim, self.n_output)
        self.ep_fc1 = nn.Linear(self.output_dim, self.n_output)
        self.smi_ep_fc1 = nn.Linear(self.output_dim, self.num_features_x*2)
        self.gc_fc = nn.Linear(2*self.num_features_x, self.n_output)
    
    def forward(
        self,
        encodedSmi: torch.Tensor,
        encodedSmi_mask: torch.Tensor,
        ecfp: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            encodedSmi (torch.Tensor): Tokenized SMILES indices, shape (B, L)
            encodedSmi_mask (torch.Tensor): SMILES mask, shape (B, L), 1 for valid, 0 for pad
            ecfp (torch.Tensor): ECFP tensor, shape (B, T, F) or (B, F)
            x (torch.Tensor): Graph node features, shape (N, num_features_x)
            edge_index (torch.Tensor): Graph edge index, shape (2, E)
            batch (torch.Tensor): Batch assignment vector, shape (N,)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                smi_out: (B,)
                ep_out: (B,) or (B, T) depending on your head (kept compatible with your code)
                gc_out: (B,)
        """
        # ==========================================================
        # SMILES branch (FIXED): (B, L, H) -> pooled (B, H) -> regression
        # ==========================================================
        smi_mask = encodedSmi_mask.unsqueeze(1)  # keep as your transformer expects
        smi_encoded = self.encoder(encodedSmi, smi_mask)  # expected (B, L, H)

        # mask-aware mean pooling over L
        token_mask = encodedSmi_mask.unsqueeze(-1).type_as(smi_encoded)  # (B, L, 1)
        smi_sum = (smi_encoded * token_mask).sum(dim=1)                  # (B, H)
        denom = token_mask.sum(dim=1).clamp_min(1.0)                     # (B, 1)
        smi_pooled = smi_sum / denom                                     # (B, H)

        # Optional normalization if you have it; safe even if not used
        if hasattr(self, "smi_norm"):
            smi_pooled = self.smi_norm(smi_pooled)

        # IMPORTANT: smi_fc2 must be Linear(hidden_dim, n_output)
        # If your current __init__ has smi_fc2 = Linear(hidden_dim, 1), this works.
        smi_out = self.smi_fc2(smi_pooled).squeeze(-1)  # (B,)

        # ==========================================================
        # ECFP branch (kept as-is, but add 2D -> 3D safety)
        # ==========================================================
        if ecfp.dim() == 2:
            ecfp = ecfp.unsqueeze(1)  # (B, 1, F)
        ep_gru, _ = self.ep_gru(ecfp)

        ep_attended_out = 0.0
        for i in range(self.num_attention_heads):
            ep_attention_weights = torch.softmax(self.ep_attention_layers[i](ep_gru), dim=1)
            ep_linear = self.ep_fc_layers[i](ep_gru)
            ep_attended_out = ep_attended_out + ep_attention_weights * ep_linear
        ep_attended_out = ep_attended_out / float(self.num_attention_heads)

        # your original code used ep_fc1 then squeeze; keep behavior
        ep_out = self.ep_fc1(ep_attended_out).squeeze()

        # ==========================================================
        # Graph branch (kept as-is)
        # ==========================================================
        x = self.gcn_conv1(x, edge_index)
        x = F.relu(x)
        x = self.gcn_conv2(x, edge_index)
        x = F.relu(x)
        gc = gap(x, batch)
        gc_out = self.gc_fc(gc).squeeze()

        return smi_out, ep_out, gc_out

    def get_embeddings(
        self,
        encodedSmi: torch.Tensor,
        encodedSmi_mask: torch.Tensor,
        ecfp: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get embeddings for the input data.

        Args:
            encodedSmi (torch.Tensor): Tokenized SMILES indices, shape (B, L)
            encodedSmi_mask (torch.Tensor): SMILES mask, shape (B, L), 1 for valid, 0 for pad
            ecfp (torch.Tensor): ECFP tensor, shape (B, T, F) or (B, F)
            x (torch.Tensor): Graph node features, shape (N, num_features_x)
            edge_index (torch.Tensor): Graph edge index, shape (2, E)
            batch (torch.Tensor): Batch assignment vector, shape (N,)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                smi_embedding: (B, hidden_dim)
                ep_embedding: (B, output_dim)
                gc_embedding: (B, 2*num_features_x)
        """
        # ==========================================================
        # SMILES embedding: (B, L, H) -> pooled (B, H)
        # ==========================================================
        smi_mask = encodedSmi_mask.unsqueeze(1)
        smi_encoded = self.encoder(encodedSmi, smi_mask)  # (B, L, H)

        token_mask = encodedSmi_mask.unsqueeze(-1).type_as(smi_encoded)  # (B, L, 1)
        smi_sum = (smi_encoded * token_mask).sum(dim=1)                  # (B, H)
        denom = token_mask.sum(dim=1).clamp_min(1.0)                     # (B, 1)
        smi_embedding = smi_sum / denom                                   # (B, H)

        if hasattr(self, "smi_norm"):
            smi_embedding = self.smi_norm(smi_embedding)

        # ==========================================================
        # ECFP embedding: attention output pooled over T -> (B, output_dim)
        # ==========================================================
        if ecfp.dim() == 2:
            ecfp = ecfp.unsqueeze(1)  # (B, 1, F)
        ep_gru, _ = self.ep_gru(ecfp)  # (B, T, 2H)

        ep_attended_out = 0.0
        for i in range(self.num_attention_heads):
            ep_attention_weights = torch.softmax(self.ep_attention_layers[i](ep_gru), dim=1)  # (B, T, output_dim)
            ep_linear = self.ep_fc_layers[i](ep_gru)                                          # (B, T, output_dim)
            ep_attended_out = ep_attended_out + ep_attention_weights * ep_linear
        ep_attended_out = ep_attended_out / float(self.num_attention_heads)                   # (B, T, output_dim)

        # time pooling
        ep_embedding = ep_attended_out.mean(dim=1)                                             # (B, output_dim)

        # ==========================================================
        # Graph embedding: (B, 2*num_features_x)
        # ==========================================================
        x = self.gcn_conv1(x, edge_index)
        x = F.relu(x)
        x = self.gcn_conv2(x, edge_index)
        x = F.relu(x)
        gc_embedding = gap(x, batch)

        return smi_embedding, ep_embedding, gc_embedding




