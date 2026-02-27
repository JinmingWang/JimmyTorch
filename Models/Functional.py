import torch
from contextlib import nullcontext

def extendAs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Extend tensor a to the same number of dimensions as tensor b by adding singleton dimensions.
    :param a: Tensor to be extended.
    :param b: Tensor to match shape with.
    :return: Extended tensor a.
    """
    if a.ndim < b.ndim:
        return a.view(a.shape + (1,) * (b.ndim - a.ndim))



class ReplaceMultiHeadAttention(torch.nn.Module):
    # This is to replace the torch.nn.MultiheadAttention so we can correctly compute computational costs
    def __init__(self, multi_head_attention: torch.nn.MultiheadAttention):
        super().__init__()
        self.num_heads = multi_head_attention.num_heads
        self.embed_dim = multi_head_attention.embed_dim
        self.kdim = multi_head_attention.kdim
        self.vdim = multi_head_attention.vdim
        self.dropout = multi_head_attention.dropout
        self.head_dim = multi_head_attention.head_dim
        self.batch_first = True

        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = torch.nn.Linear(self.kdim, self.embed_dim)
        self.v_proj = torch.nn.Linear(self.vdim, self.embed_dim)
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, q, k, v, *args, **kwargs):
        B, L_q, _ = q.shape
        B, L_k, _ = k.shape
        B, L_v, _ = v.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2).flatten(0, 1)     # (B * num_heads, L_q, head_dim)
        k = k.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2).flatten(0, 1)     # (B * num_heads, L_k, head_dim)
        v = v.view(B, L_v, self.num_heads, self.head_dim).transpose(1, 2).flatten(0, 1)     # (B * num_heads, L_v, head_dim)

        score = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn = torch.nn.functional.softmax(score, dim=-1)
        attn = torch.nn.functional.dropout(attn, p=self.dropout, training=self.training)

        out = attn @ v    # (B * num_heads, L_q, head_dim)
        out = out.unflatten(0, (B, -1)).transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)  # (B, L_q, embed_dim)
        out = self.out_proj(out)
        return out, attn


class ReplaceTransformerEncoderLayer(torch.nn.Module):
    # This is to replace the torch.nn.TransformerEncoderLayer so we can correctly compute computational costs
    def __init__(self, encoder_layer: torch.nn.TransformerEncoderLayer):
        super().__init__()
        self.self_attn = ReplaceMultiHeadAttention(encoder_layer.self_attn)
        self.linear1 = encoder_layer.linear1
        self.linear2 = encoder_layer.linear2
        self.norm1 = encoder_layer.norm1
        self.norm2 = encoder_layer.norm2
        self.dropout1 = encoder_layer.dropout1
        self.dropout2 = encoder_layer.dropout2
        self.activation = encoder_layer.activation
        self.batch_first = True
        self.embed_dim = encoder_layer.self_attn.embed_dim

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.norm1(src)
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        src = self.norm2(src)
        src = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout2(src)
        return src

def getAutoCast(data_sample: torch.Tensor, mixed_precision: bool):
    if mixed_precision:
        return torch.autocast(device_type=data_sample.device.type, dtype=torch.float16)
    else:
        return nullcontext()