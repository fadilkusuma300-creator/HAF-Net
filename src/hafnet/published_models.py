from __future__ import annotations

"""Published deep-tabular comparison architectures used by the benchmark.

The implementations in this module keep the defining architectural mechanisms of
TabNet, FT-Transformer, and Trompt while exposing a small common PyTorch
interface for the two all-numerical financial benchmarks. TabTransformer is
provided separately as an explicitly named numerical-input adaptation because
the original model contextualizes categorical embeddings and the benchmark
contains no categorical columns.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax transformation (Martins & Astudillo, 2016)."""
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(shifted, descending=True, dim=dim)
    z_cumsum = z_sorted.cumsum(dim) - 1
    shape = [1] * logits.dim()
    shape[dim] = logits.size(dim)
    ranks = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype).view(shape)
    support = ranks * z_sorted > z_cumsum
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = z_cumsum.gather(dim, k.long() - 1) / k.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


class _GLUBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, 2 * d_out, bias=False)
        self.bn = nn.BatchNorm1d(2 * d_out, momentum=0.02)
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.fc(x))
        value, gate = x.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class _FeatureTransformer(nn.Module):
    """TabNet feature transformer with shared and step-specific GLU blocks."""

    def __init__(self, d_in: int, d_hidden: int, shared: nn.ModuleList) -> None:
        super().__init__()
        self.shared = shared
        self.independent = nn.ModuleList([_GLUBlock(d_hidden, d_hidden), _GLUBlock(d_hidden, d_hidden)])
        self.input_projection = _GLUBlock(d_in, d_hidden) if d_in != d_hidden else None

    @staticmethod
    def _residual(x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return (x + update) * math.sqrt(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_projection is not None:
            x = self.input_projection(x)
        for block in self.shared:
            x = self._residual(x, block(x))
        for block in self.independent:
            x = self._residual(x, block(x))
        return x


class _AttentiveTransformer(nn.Module):
    def __init__(self, d_attn: int, n_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_attn, n_features, bias=False)
        self.bn = nn.BatchNorm1d(n_features, momentum=0.02)

    def forward(self, attention_features: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        logits = self.bn(self.fc(attention_features))
        return sparsemax(logits * prior, dim=-1)


class TabNet(nn.Module):
    """TabNet architecture for continuous inputs.

    This implementation retains sequential attentive masks, sparsemax feature
    selection, GLU feature transformers, a prior scale, and decision-step
    aggregation from Arik & Pfister (AAAI 2021).
    """

    def __init__(
        self,
        n_features: int,
        d_hidden: int = 64,
        n_steps: int = 4,
        gamma: float = 1.5,
    ) -> None:
        super().__init__()
        if n_steps < 2:
            raise ValueError("TabNet requires at least two decision steps.")
        self.n_features = n_features
        self.d_hidden = d_hidden
        self.n_steps = n_steps
        self.gamma = float(gamma)
        self.input_bn = nn.BatchNorm1d(n_features)
        shared = nn.ModuleList([_GLUBlock(d_hidden, d_hidden), _GLUBlock(d_hidden, d_hidden)])
        self.initial = _FeatureTransformer(n_features, d_hidden, shared)
        self.transformers = nn.ModuleList(
            [_FeatureTransformer(n_features, d_hidden, shared) for _ in range(n_steps)]
        )
        self.attentive = nn.ModuleList(
            [_AttentiveTransformer(d_hidden, n_features) for _ in range(n_steps - 1)]
        )
        self.head = nn.Linear(d_hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        prior = torch.ones_like(x)
        initial = self.initial(x)
        attention_features = initial
        aggregated = torch.zeros(x.shape[0], self.d_hidden, dtype=x.dtype, device=x.device)
        mask = torch.ones_like(x)
        for step in range(self.n_steps):
            if step > 0:
                mask = self.attentive[step - 1](attention_features, prior)
                prior = prior * (self.gamma - mask)
            transformed = self.transformers[step](x * mask)
            decision = F.relu(transformed)
            aggregated = aggregated + decision
            attention_features = transformed
        return self.head(aggregated).squeeze(-1)


class _ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class _FTBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, first: bool) -> None:
        super().__init__()
        self.attn_norm = None if first else nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        d_hidden = max(4, int(d_model * 4 / 3))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_hidden),
            _ReGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, query_cls_only: bool = False) -> torch.Tensor:
        identity = x
        q_input = x if self.attn_norm is None else self.attn_norm(x)
        q = q_input[:, :1] if query_cls_only else q_input
        attn_out, _ = self.attn(q, q_input, q_input, need_weights=False)
        if query_cls_only:
            x = identity[:, :1] + self.attn_drop(attn_out)
        else:
            x = identity + self.attn_drop(attn_out)
        x = x + self.ffn_drop(self.ffn(self.ffn_norm(x)))
        return x


class FTTransformer(nn.Module):
    """FT-Transformer for continuous features following Gorishniy et al. (2021)."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        dropout: float = 0.1,
        layers: int = 3,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("FT-Transformer embedding width must be divisible by the number of heads.")
        bound = d_model ** -0.5
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self.cls = nn.Parameter(torch.empty(d_model))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.cls, -bound, bound)
        self.blocks = nn.ModuleList(
            [_FTBlock(d_model, heads, dropout, first=(i == 0)) for i in range(layers)]
        )
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.view(1, 1, -1).expand(x.shape[0], 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, query_cls_only=(i == len(self.blocks) - 1))
        return self.output(tokens[:, 0]).squeeze(-1)


class TabTransformerNumericAdaptation(nn.Module):
    """Numerical-input adaptation of TabTransformer.

    The published TabTransformer contextualizes categorical embeddings. Both
    financial benchmarks used here contain continuous features only, so this
    explicitly labelled adaptation tokenizes each continuous feature before the
    contextual Transformer while retaining the published contextual-attention
    principle. It is not presented as a reproduction of the original categorical
    architecture.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        dropout: float = 0.1,
        layers: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("TabTransformer width must be divisible by the number of heads.")
        self.value_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.value_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.column_embedding = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.mlp = nn.Sequential(
            nn.Linear(n_features * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.value_weight.unsqueeze(0) + self.value_bias.unsqueeze(0)
        tokens = tokens + self.column_embedding.unsqueeze(0)
        contextual = self.transformer(tokens)
        return self.mlp(contextual.flatten(1)).squeeze(-1)


class _NumericFeatureEncoder(nn.Module):
    def __init__(self, n_features: int, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, channels))
        self.bias = nn.Parameter(torch.empty(n_features, channels))
        self.norm = nn.LayerNorm(channels)
        bound = channels ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)))


class _TromptCell(nn.Module):
    def __init__(self, channels: int, n_features: int, n_prompts: int) -> None:
        super().__init__()
        self.channels = channels
        self.n_features = n_features
        self.n_prompts = n_prompts
        self.embedding_column = nn.Parameter(torch.empty(n_features, channels))
        self.embedding_prompt = nn.Parameter(torch.empty(n_prompts, channels))
        self.lin = nn.Linear(2 * channels, channels)
        self.prompt_feature_scale = nn.Parameter(torch.empty(n_prompts))
        groups = 2 if n_prompts % 2 == 0 else 1
        self.group_norm = nn.GroupNorm(groups, n_prompts)
        self.column_norm = nn.LayerNorm(channels)
        self.prompt_norm = nn.LayerNorm(channels)
        nn.init.normal_(self.embedding_column, std=0.01)
        nn.init.normal_(self.embedding_prompt, std=0.01)
        nn.init.normal_(self.prompt_feature_scale, std=0.01)

    def forward(self, x: torch.Tensor, prompt_state: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        e_prompt = self.prompt_norm(self.embedding_prompt).unsqueeze(0).expand(batch, -1, -1)
        e_prompt = e_prompt + prompt_state + self.lin(torch.cat([e_prompt, prompt_state], dim=-1))
        e_column = self.column_norm(self.embedding_column).unsqueeze(0).expand(batch, -1, -1)
        importance = F.softmax(e_prompt @ e_column.transpose(1, 2), dim=-1).unsqueeze(-1)
        z = torch.einsum("bfc,p->bpfc", x, self.prompt_feature_scale)
        z = F.relu(z)
        x_expanded = x.unsqueeze(1).expand(-1, self.n_prompts, -1, -1)
        features = self.group_norm(z) + x_expanded
        return (features * importance).sum(dim=2)


class _TromptDecoder(nn.Module):
    def __init__(self, channels: int, n_prompts: int) -> None:
        super().__init__()
        self.attn = nn.Linear(channels, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Linear(channels, 1),
        )
        self.n_prompts = n_prompts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attn(x), dim=1)
        pooled = (weights * x).sum(dim=1)
        return self.mlp(pooled).squeeze(-1)


class Trompt(nn.Module):
    """Trompt architecture following Chen et al. (ICML 2023).

    The model uses layer-specific numerical feature encoders, prompt and column
    embeddings, prompt-to-column importance matrices, repeated Trompt cells,
    a shared prompt decoder, and layer-wise predictions.
    """

    def __init__(
        self,
        n_features: int,
        channels: int = 128,
        n_prompts: int = 128,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.prompt = nn.Parameter(torch.empty(n_prompts, channels))
        nn.init.normal_(self.prompt, std=0.01)
        self.encoders = nn.ModuleList([_NumericFeatureEncoder(n_features, channels) for _ in range(n_layers)])
        self.cells = nn.ModuleList([_TromptCell(channels, n_features, n_prompts) for _ in range(n_layers)])
        self.decoder = _TromptDecoder(channels, n_prompts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prompt_state = self.prompt.unsqueeze(0).expand(x.shape[0], -1, -1)
        outputs = []
        for encoder, cell in zip(self.encoders, self.cells):
            features = encoder(x)
            prompt_state = cell(features, prompt_state)
            outputs.append(self.decoder(prompt_state))
        return torch.stack(outputs, dim=1)

    @staticmethod
    def inference_logits(layerwise_logits: torch.Tensor) -> torch.Tensor:
        return layerwise_logits.mean(dim=1)
