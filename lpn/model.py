"""Core Liquid Probabilistic Network architecture."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ._cfc import CfCCell


@dataclass(frozen=True)
class LPNConfig:
    history_points: int = 151
    future_points: int = 150
    width: int = 320
    heads: int = 5
    embedding_hidden: int = 96
    history_ffn_hidden: int = 640
    gated_ffn_hidden: int = 640
    future_kernels: tuple = (3, 11)
    dilations: tuple = (1, 2, 4, 8, 16)
    dropout: float = 0.08
    sample_seconds: float = 0.2

    def __post_init__(self):
        if self.width != 320:
            raise ValueError("This model uses a 320-unit CfC history layer")
        if self.width % self.heads or self.width % 2:
            raise ValueError("Width must be even and divisible by the head count")
        if self.history_points < 1 or self.future_points < 1:
            raise ValueError("Sequence lengths must be positive")
        if len(self.future_kernels) != 2:
            raise ValueError("The future mixer requires two convolution branches")
        if any(k < 1 or k % 2 == 0 for k in self.future_kernels):
            raise ValueError("Convolution kernels must be positive odd integers")
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError("Dilations must be positive")
        if not 0 <= self.dropout < 1 or self.sample_seconds <= 0:
            raise ValueError("Invalid dropout or observation interval")


def positional_encoding(length, width):
    positions = torch.arange(length).float()[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.) / width))
    encoding = torch.empty(length, width)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding



class _LiquidSettings:
    units = 320
    mode = "default"
    backbone_activation = "lecun_tanh"
    backbone_units = 320
    backbone_layers = 1

    def __init__(self, sample_seconds):
        self.sample_seconds = sample_seconds



class LiquidHistoryBlock(nn.Module):
    def __init__(self, config, liquid):
        super().__init__()
        if config.width != liquid.units:
            raise ValueError('CfC output width must match the decoder memory width')
        self.sample_seconds = liquid.sample_seconds
        self.temporal_norm = nn.LayerNorm(config.width)
        self.cell = CfCCell(config.width, liquid.units, mode=liquid.mode,
                            backbone_activation=liquid.backbone_activation,
                            backbone_units=liquid.backbone_units,
                            backbone_layers=liquid.backbone_layers,
                            backbone_dropout=0.0)
        self.feed_norm = nn.LayerNorm(config.width)
        self.feed = nn.Sequential(nn.Linear(config.width, config.history_ffn_hidden),
                                  nn.GELU(), nn.Dropout(config.dropout),
                                  nn.Linear(config.history_ffn_hidden, config.width))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, history):
        observed = self.temporal_norm(history)
        state = observed.new_zeros((observed.shape[0], observed.shape[2]))
        sequence = []
        for point in observed.unbind(dim=1):
            output, state = self.cell(point, state, self.sample_seconds)
            sequence.append(output)
        hidden = history + self.dropout(torch.stack(sequence, dim=1))
        return hidden + self.dropout(self.feed(self.feed_norm(hidden)))


class MotionFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, hidden = config.width, config.embedding_hidden
        self.history_projection = nn.Sequential(nn.Linear(7, hidden), nn.GELU(),
                                                nn.Linear(hidden, width))
        self.context_projection = nn.Sequential(nn.Linear(8, hidden), nn.GELU(),
                                                nn.Linear(hidden, width))
        self.register_buffer('positions', positional_encoding(config.history_points + 1, width))
        self.layers = nn.ModuleList([
            LiquidHistoryBlock(config, _LiquidSettings(config.sample_seconds))
        ])
        self.norm = nn.LayerNorm(width)

    def forward(self, history, context):
        context_token = self.context_projection(context)
        hidden = self.history_projection(history) + self.positions[1:history.shape[1] + 1]
        hidden = hidden + context_token[:, None]
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(torch.cat([context_token[:, None], hidden], dim=1))


class DualScaleFutureMixer(nn.Module):
    def __init__(self, width, kernels, dilation):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, kernel, dilation=dilation, groups=width,
                          padding=dilation * (kernel - 1) // 2, padding_mode='replicate'),
                nn.Conv1d(width, width // 2, 1),
            )
            for kernel in kernels
        ])

    def forward(self, hidden):
        channels = hidden.transpose(1, 2)
        mixed = torch.cat([branch(channels) for branch in self.branches], dim=1)
        return F.gelu(mixed.transpose(1, 2))


class GatedFeedForward(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.content = nn.Linear(width, hidden)
        self.gate = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        gated = self.content(hidden) * torch.sigmoid(self.gate(hidden))
        return self.output(self.dropout(gated))


class ConditionalConvolutionDecoderLayer(nn.Module):
    def __init__(self, config, dilation):
        super().__init__()
        width = config.width
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_mixer = DualScaleFutureMixer(width, config.future_kernels, dilation)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, config.heads,
                                                     dropout=config.dropout, batch_first=True)
        self.feed_norm = nn.LayerNorm(width)
        self.condition_affine = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        nn.init.zeros_(self.condition_affine[-1].weight)
        nn.init.zeros_(self.condition_affine[-1].bias)
        self.feed = GatedFeedForward(width, config.gated_ffn_hidden, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, memory, condition):
        hidden = hidden + self.dropout(self.temporal_mixer(self.temporal_norm(hidden)))
        query = self.cross_norm(hidden)
        attended = self.cross_attention(query, memory, memory, need_weights=False)[0]
        hidden = hidden + self.dropout(attended)
        scale, shift = self.condition_affine(condition).chunk(2, dim=-1)
        feed_input = self.feed_norm(hidden) * (1 + scale[:, None]) + shift[:, None]
        return hidden + self.dropout(self.feed(feed_input))


class LPN(nn.Module):
    """Conditional vector field for normalized seven-channel future states."""

    def __init__(self, config=None):
        super().__init__()
        self.config = LPNConfig() if config is None else config
        config = self.config
        self.history_points = config.history_points
        self.future_points = config.future_points
        self.motion_features = MotionFeatureExtractor(config)
        self.future_projection = nn.Linear(7, config.width)
        self.register_buffer("future_positions", positional_encoding(config.future_points, config.width))
        self.decoder_layers = nn.ModuleList([
            ConditionalConvolutionDecoderLayer(config, dilation)
            for dilation in config.dilations
        ])
        self.output_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, 7)

    def encode(self, history, context):
        if history.shape[1:] != (self.history_points, 7):
            raise ValueError("History must have shape [batch, history_points, 7]")
        if context.shape != (len(history), 8):
            raise ValueError("Context must have shape [batch, 8]")
        return self.motion_features(history, context)

    def vector_field(self, state, memory):
        if state.shape[1:] != (self.future_points, 7):
            raise ValueError("Future state must have shape [batch, future_points, 7]")
        if memory.shape != (len(state), self.history_points + 1, self.config.width):
            raise ValueError("Encoded history has an unexpected shape")
        hidden = self.future_projection(state) + self.future_positions
        condition = memory.mean(dim=1)
        for layer in self.decoder_layers:
            hidden = layer(hidden, memory, condition)
        return self.output(self.output_norm(hidden))

    def forward(self, history, context, state):
        return self.vector_field(state, self.encode(history, context))
