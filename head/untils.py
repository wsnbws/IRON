import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, **kargs) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# ===== Weight Initialization Utilities =====

def init_conv_weights(module, mode='fan_out', nonlinearity='relu'):
    """Initialize convolution layer weights."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_bn_weights(module):
    """Initialize batch normalization weights."""
    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def init_linear_weights(module, init_type='xavier'):
    """Initialize linear layer weights."""
    if isinstance(module, nn.Linear):
        if init_type == 'xavier':
            nn.init.xavier_uniform_(module.weight)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight)
        elif init_type == 'normal':
            nn.init.normal_(module.weight, std=0.02)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_embedding_weights(module, std=0.02):
    """Initialize embedding layer weights."""
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=std)


def init_conv_module_weights(conv_module):
    """Initialize ConvModule weights (handles both conv and bn)."""
    if hasattr(conv_module, 'conv'):
        init_conv_weights(conv_module.conv)
    if hasattr(conv_module, 'bn') and conv_module.bn is not None:
        init_bn_weights(conv_module.bn)


def init_psp_weights(psp_module):
    """Initialize PSP module weights."""
    for module in psp_module.modules():
        init_conv_weights(module)
        init_bn_weights(module)


def init_mlp_weights(mlp_module):
    """Initialize MLP weights."""
    for module in mlp_module.modules():
        init_linear_weights(module, init_type='xavier')


def init_attention_weights(attention_module):
    """Initialize attention module weights."""
    for module in attention_module.modules():
        init_linear_weights(module, init_type='xavier')
        init_conv_weights(module)


def init_encoder_weights(encoder_module):
    """Initialize encoder module weights."""
    for module in encoder_module.modules():
        init_conv_weights(module)
        init_bn_weights(module)
        init_linear_weights(module, init_type='xavier')


def init_predictor_weights(predictor_module):
    """Initialize point predictor weights."""
    for module in predictor_module.modules():
        init_linear_weights(module, init_type='xavier')
        init_conv_weights(module)


def init_sam_weights(prompt_encoder, mask_decoder):
    """Initialize SAM decoder weights."""
    # Initialize prompt encoder
    for module in prompt_encoder.modules():
        init_linear_weights(module, init_type='xavier')
        init_conv_weights(module)
        init_embedding_weights(module)
        
    # Initialize mask decoder  
    for module in mask_decoder.modules():
        init_linear_weights(module, init_type='xavier')
        init_conv_weights(module)
        init_bn_weights(module)