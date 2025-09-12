import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import rnn

import torch.backends.cuda

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from core.models.e2e.bandit.attend import Attend

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack

from hyper_connections import get_init_and_expand_reduce_stream_functions

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack

from hyper_connections import get_init_and_expand_reduce_stream_functions

# helper functions

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# attention

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        rotary_embed = None,
        flash = True,
        learned_value_residual_mix = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash = flash, dropout = dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_value_residual_mix = nn.Linear(dim, heads) if learned_value_residual_mix else None

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x, value_residual = None):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        orig_v = v

        if exists(self.to_value_residual_mix):
            mix = self.to_value_residual_mix(x)
            mix = rearrange(mix, 'b n h -> b h n 1').sigmoid()

            assert exists(value_residual)
            v = v.lerp(value_residual, mix)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), orig_v

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        norm_output = True,
        rotary_embed = None,
        flash_attn = True,
        add_value_residual = False,
        num_residual_streams = 1,
        num_residual_fracs = 1
    ):
        super().__init__()
        self.layers = ModuleList([])

        init_hyper_conn, *_ = get_init_and_expand_reduce_stream_functions(num_residual_streams, num_fracs = num_residual_fracs)

        for _ in range(depth):
            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_embed = rotary_embed, flash = flash_attn, learned_value_residual_mix = add_value_residual)),
                init_hyper_conn(dim = dim, branch = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, value_residual = None):

        first_values = None

        for attn, ff in self.layers:
            x, next_values = attn(x, value_residual = value_residual)

            first_values = default(first_values, next_values)

            x = ff(x)

        return self.norm(x), first_values

class TimeFrequencyModellingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class ResidualRNN(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            rnn_dim: int,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            use_batch_trick: bool = True,
            use_layer_norm: bool = True,
    ) -> None:
        # n_group is the size of the 2nd dim
        super().__init__()

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(emb_dim)
        else:
            self.norm = nn.GroupNorm(num_groups=emb_dim, num_channels=emb_dim)

        self.rnn = rnn.__dict__[rnn_type](
                input_size=emb_dim,
                hidden_size=rnn_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
        )

        self.fc = nn.Linear(
                in_features=rnn_dim * (2 if bidirectional else 1),
                out_features=emb_dim
        )

        self.use_batch_trick = use_batch_trick
        if not self.use_batch_trick:
            warnings.warn("NOT USING BATCH TRICK IS EXTREMELY SLOW!!")

    def forward(self, z):
        # z = (batch, n_uncrossed, n_across, emb_dim)

        z0 = torch.clone(z)

        # print(z.device)

        if self.use_layer_norm:
            z = self.norm(z)  # (batch, n_uncrossed, n_across, emb_dim)
        else:
            z = torch.permute(
                    z, (0, 3, 1, 2)
            )  # (batch, emb_dim, n_uncrossed, n_across)

            z = self.norm(z)  # (batch, emb_dim, n_uncrossed, n_across)

            z = torch.permute(
                    z, (0, 2, 3, 1)
            )  # (batch, n_uncrossed, n_across, emb_dim)

        batch, n_uncrossed, n_across, emb_dim = z.shape

        if self.use_batch_trick:
            z = torch.reshape(z, (batch * n_uncrossed, n_across, emb_dim))

            z = self.rnn(z.contiguous())[0]  # (batch * n_uncrossed, n_across, dir_rnn_dim)

            z = torch.reshape(z, (batch, n_uncrossed, n_across, -1))
            # (batch, n_uncrossed, n_across, dir_rnn_dim)
        else:
            # Note: this is EXTREMELY SLOW
            zlist = []
            for i in range(n_uncrossed):
                zi = self.rnn(z[:, i, :, :])[0]  # (batch, n_across, emb_dim)
                zlist.append(zi)

            z = torch.stack(
                    zlist,
                    dim=1
            )  # (batch, n_uncrossed, n_across, dir_rnn_dim)

        z = self.fc(z)  # (batch, n_uncrossed, n_across, emb_dim)

        z = z + z0

        return z

class SeqBandModellingModule(TimeFrequencyModellingModule):
    def __init__(
            self,
            n_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            time_transformer_depth = 1,
            freq_transformer_depth = 1,
            parallel_mode=False,
            dim_head = 64,
            heads = 8,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            flash_attn = True,
            num_residual_streams = 1,
            num_residual_fracs = 1, 
    ) -> None:
        super().__init__()
        
        _, self.expand_stream, self.reduce_stream = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)
        
        self.layers = nn.ModuleList([])

        transformer_kwargs = dict(
            dim = emb_dim,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            flash_attn = flash_attn,
            num_residual_streams = num_residual_streams,
            num_residual_fracs = num_residual_fracs,
            norm_output = False,
        )
        
        time_rotary_embed = RotaryEmbedding(dim = dim_head)
        freq_rotary_embed = RotaryEmbedding(dim = dim_head)
        for layer_index in range(n_modules):
            is_first = layer_index == 0

            self.layers.append(nn.ModuleList([
                Transformer(depth = time_transformer_depth, rotary_embed = time_rotary_embed, add_value_residual = not is_first, **transformer_kwargs),
                Transformer(depth = freq_transformer_depth, rotary_embed = freq_rotary_embed, add_value_residual = not is_first, **transformer_kwargs)
            ]))

        self.final_norm = RMSNorm(emb_dim)

        self.parallel_mode = parallel_mode

    def forward(self, x):
        # x = (batch, n_bands, n_time, emb_dim)
        x = rearrange(x, 'b f t d -> b t f d')
        # value residuals

        time_v_residual = None
        freq_v_residual = None

        # maybe expand residual streams

        x = self.expand_stream(x)

        # axial / hierarchical attention

        for time_transformer, freq_transformer in self.layers:

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            x, next_time_v_residual = time_transformer(x, value_residual = time_v_residual)

            time_v_residual = default(time_v_residual, next_time_v_residual)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            x, next_freq_v_residual = freq_transformer(x, value_residual = freq_v_residual)

            freq_v_residual = default(freq_v_residual, next_freq_v_residual)

            x, = unpack(x, ps, '* f d')

        # maybe reduce residual streams

        x = self.reduce_stream(x)

        x = self.final_norm(x)
        
        x = rearrange(x, 'b t f d -> b f t d')
        return x