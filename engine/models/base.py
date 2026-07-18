from __future__ import annotations
import types
import torch
import torch.nn as nn
from typing import Any

class GLUSpatialFusion(nn.Module):
    """Late-fusion: gate = sigmoid(W_g [e; s]); h = gate * e + (1-gate) * s."""

    def __init__(self, spatial_dim: int, hidden_size: int):
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(spatial_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2), # Add spatial dropout
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, text_emb: torch.Tensor, spatial_feat: torch.Tensor) -> torch.Tensor:
        s    = self.spatial_mlp(spatial_feat)
        gate = torch.sigmoid(self.gate(torch.cat([text_emb, s], dim=-1)))
        return gate * text_emb + (1 - gate) * s


def custom_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.FloatTensor | None = None,
    past_key_values: Any = None,
    **kwargs,
) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.attention_head_size)

    # get all proj
    query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
    key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
    value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

    # Compute attention logits
    scaling = query_layer.size(-1) ** -0.5
    attn_weights = torch.matmul(query_layer, key_layer.transpose(2, 3)) * scaling

    # Retrieve spatial_matrix_temp
    spatial_matrix = None
    if hasattr(self, "encoder_ref") and hasattr(self.encoder_ref, "spatial_matrix_temp"):
        spatial_matrix = self.encoder_ref.spatial_matrix_temp

    if spatial_matrix is not None:
        B, num_heads, N, _ = attn_weights.shape
        # Ensure spatial_matrix shape is (B, N, N, 2)
        if spatial_matrix.dim() == 3:
            spatial_matrix = spatial_matrix.unsqueeze(0)
            
        SM_B, SM_N, _, _ = spatial_matrix.shape
        
        # Cast and move device
        spatial_matrix = spatial_matrix.to(device=attn_weights.device, dtype=attn_weights.dtype)
        
        if SM_B != B:
            if SM_B == 1:
                spatial_matrix = spatial_matrix.expand(B, -1, -1, -1)
                SM_B = B
            else:
                spatial_matrix = spatial_matrix[:B]
                SM_B = spatial_matrix.shape[0]
                
        if SM_N != N:
            if SM_N > N:
                spatial_matrix = spatial_matrix[:, :N, :N, :]
            else:
                padded = torch.zeros(SM_B, N, N, 2, device=spatial_matrix.device, dtype=spatial_matrix.dtype)
                padded[:, :SM_N, :SM_N, :] = spatial_matrix
                spatial_matrix = padded
                
        # Project using spatial_bias_proj: (B, N, N, 2) -> (B, N, N, num_heads)
        bias = self.spatial_bias_proj(spatial_matrix)
        # Permute to (B, num_heads, N, N) to match attn_weights
        bias = bias.permute(0, 3, 1, 2)
        
        # Add directly as an additive bias to attention logit score matrix
        attn_weights = attn_weights + bias

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout.p if self.training else 0.0, training=self.training)

    attn_output = torch.matmul(attn_weights, value_layer)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    
    return attn_output, attn_weights


def custom_segmenter_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.FloatTensor | None = None,
    past_key_values: Any = None,
    **kwargs,
) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.attention_head_size)

    # get all proj
    query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
    key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
    value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

    # Compute attention logits
    scaling = query_layer.size(-1) ** -0.5
    attn_weights = torch.matmul(query_layer, key_layer.transpose(2, 3)) * scaling

    # Retrieve spatial_matrix_temp
    spatial_matrix = None
    if hasattr(self, "encoder_ref") and hasattr(self.encoder_ref, "spatial_matrix_temp"):
        spatial_matrix = self.encoder_ref.spatial_matrix_temp

    if spatial_matrix is not None:
        B, num_heads, N, _ = attn_weights.shape
        # Ensure spatial_matrix shape is (B, N, N, 2)
        if spatial_matrix.dim() == 3:
            spatial_matrix = spatial_matrix.unsqueeze(0)
            
        SM_B, SM_N, _, _ = spatial_matrix.shape
        
        # Cast and move device
        spatial_matrix = spatial_matrix.to(device=attn_weights.device, dtype=attn_weights.dtype)
        
        if SM_B != B:
            if SM_B == 1:
                spatial_matrix = spatial_matrix.expand(B, -1, -1, -1)
                SM_B = B
            else:
                spatial_matrix = spatial_matrix[:B]
                SM_B = spatial_matrix.shape[0]
                
        if SM_N != N:
            if SM_N > N:
                spatial_matrix = spatial_matrix[:, :N, :N, :]
            else:
                padded = torch.zeros(SM_B, N, N, 2, device=spatial_matrix.device, dtype=spatial_matrix.dtype)
                padded[:, :SM_N, :SM_N, :] = spatial_matrix
                spatial_matrix = padded
                
        # Project using spatial_bias_proj: (B, N, N, 2) -> (B, N, N, num_heads)
        bias = self.spatial_bias_proj(spatial_matrix)
        # Permute to (B, num_heads, N, N) to match attn_weights
        bias = bias.permute(0, 3, 1, 2)
        
        # Amplify attention bias: 35% spatial weight contribution
        attn_weights = 0.65 * attn_weights + 0.35 * bias

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout.p if self.training else 0.0, training=self.training)

    attn_output = torch.matmul(attn_weights, value_layer)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    
    return attn_output, attn_weights
