import jittor as jt
import jittor.nn as nn
from jittor import init
import math
from functools import partial
import warnings
from jrs.utils.registry import BACKBONES

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------
import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from tqdm import tqdm

HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))
        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def _is_legacy_zip_format(filename):
    if zipfile.is_zipfile(filename):
        infolist = zipfile.ZipFile(filename).infolist()
        return len(infolist) == 1 and not infolist[0].is_dir()
    return False

def _legacy_zip_load(filename, model_dir, map_location):
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
        f.extractall(model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)
    return jt.load(extracted_file)


model_urls = {
    "stripnet_t": "https://drive.google.com/uc?export=download&id=1Le4QoQPMUlFEssq7BFXGmaYfHoMktvju",
    "stripnet_s": "https://drive.google.com/uc?export=download&id=1_c2aXANKHl0cIBb370LNIkCyDmQpA3_o"
}

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 10, 'input_size': (3, 1024, 1024), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


# 如果需要自定义的 DropPath，实现一个简单的版本
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if self.drop_prob == 0. or not self.is_training():
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + jt.rand(shape)
        random_tensor.floor_()
        output = x.divide(keep_prob) * random_tensor
        return output

# 构建归一化层
def build_norm_layer(norm_cfg, num_features):
    norm_type = norm_cfg.get('type', 'BN')
    if norm_type == 'BN':
        return nn.BatchNorm(num_features)
    elif norm_type == 'LN':
        return nn.LayerNorm(num_features)
    else:
        raise ValueError(f'Unsupported norm type {norm_type}')

# 双重元组函数
def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer
        self.fc2 = nn.Conv(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv(dim, dim, 3, padding=1, groups=dim)

    def execute(self, x):
        x = self.dwconv(x)
        return x

class StripBlock(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv(dim, dim, 5, padding=2, groups=dim)
        self.strip_conv1 = nn.Conv(dim, dim, (k1, k2), padding=(k1 // 2, k2 // 2), groups=dim)
        self.strip_conv2 = nn.Conv(dim, dim, (k2, k1), padding=(k2 // 2, k1 // 2), groups=dim)
        self.conv1 = nn.Conv(dim, dim, 1)

    def execute(self, x):
        attn = self.conv0(x)
        attn = self.strip_conv1(attn)
        attn = self.strip_conv2(attn)
        attn = self.conv1(attn)
        return x * attn

class Attention(nn.Module):
    def __init__(self, d_model, k1, k2):
        super().__init__()
        self.proj_1 = nn.Conv(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripBlock(d_model, k1, k2)
        self.proj_2 = nn.Conv(d_model, d_model, 1)

    def execute(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., k1=1, k2=19, drop=0., drop_path=0., act_layer=nn.GELU(), norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)
            self.norm2 = build_norm_layer(norm_cfg, dim)
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, k1, k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = jt.Var(layer_scale_init_value * jt.ones((dim,)))
        self.layer_scale_2 = jt.Var(layer_scale_init_value * jt.ones((dim,)))

    def execute(self, x):
        x = x + self.drop_path(self.layer_scale_1.view(1, -1, 1, 1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.view(1, -1, 1, 1) * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                            padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def execute(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W

class StripNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], k1s=[1, 1, 1, 1], k2s=[19, 19, 19, 19],
                 drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, out_indices = (0, 1, 2), pretrained=None, init_cfg=None,
                 norm_cfg=None):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages
        self.out_indices = out_indices

        dpr = [float(x) for x in jt.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                norm_cfg=norm_cfg
            )

            blocks = []
            for j in range(depths[i]):
                block = Block(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    k1=k1s[i],
                    k2=k2s[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    norm_cfg=norm_cfg
                )
                blocks.append(block)
            blocks = nn.Sequential(*blocks)
            cur += depths[i]

            norm = norm_layer(embed_dims[i])

            setattr(self, f"patch_embed{i+1}", patch_embed)
            setattr(self, f"block{i+1}", blocks)
            setattr(self, f"norm{i+1}", norm)

    def execute(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i+1}")
            blocks = getattr(self, f"block{i+1}")
            norm = getattr(self, f"norm{i+1}")
            x, H, W = patch_embed(x)
            x = blocks(x)
            x = x.reshape(B, -1, H*W).transpose(0, 2, 1)
            x = norm(x)
            x = x.transpose(0, 2, 1).reshape(B, -1, H, W)
            if i in self.out_indices:
                outs.append(x)
        return outs

# 初始化参数函数
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, value=1.0)
        elif isinstance(m, nn.Conv):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.gauss_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.zero_()

def trunc_normal_init(module, mean=0., std=1.):
    values = jt.normal(mean, std, module.weight.shape)
    module.weight.assign(values)
    if module.bias is not None:
        module.bias.zero_()

def constant_init(module, value):
    module.weight.assign(jt.full(module.weight.shape, value))
    if module.bias is not None:
        module.bias.zero_()






from torch.hub import load_state_dict_from_url
def load_param(url, model):
    checkpoint = load_state_dict_from_url(
        url=url, map_location="cpu", check_hash=True
    )
    del checkpoint["state_dict"]["head.weight"]
    del checkpoint["state_dict"]["head.bias"]
    model.load_state_dict(checkpoint["state_dict"])
    return model

@BACKBONES.register_module()
def StripNet_T(pretrained=False, **kwargs):
    model = StripNet(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4], 
        norm_layer=nn.LayerNorm, depths=[3, 3, 5, 2],
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model.load_state_dict
        model = load_param(model_urls['stripnet_t'], model)
    return model


@BACKBONES.register_module()
def StripNet_S(pretrained=False, **kwargs):
    model = StripNet(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=nn.LayerNorm, depths=[2, 2, 4, 2],
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model.load_state_dict
        model = load_param(model_urls['stripnet_s'], model)
    return model
 

