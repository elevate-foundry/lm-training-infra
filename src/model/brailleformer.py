"""BrailleFormer: A novel architecture inspired by Braille cell structure.

Tokens are grouped into cells (like Braille's 2x3 dot matrix). Within each cell,
higher-order tensor contractions capture joint configurations. Cells are arranged
on a 2D grid with multi-directional attention (L→R, R→L, top→down, down→top).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BrailleFormerConfig:
    vocab_size: int = 256      # Character-level
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 1024
    cell_size: int = 6         # Tokens per cell (like Braille's 6 dots)
    tensor_rank: int = 16      # CP decomposition rank for trilinear interaction
    dropout: float = 0.1


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Stage 1: Cell Encoder — higher-order tensor contraction within each cell
# ---------------------------------------------------------------------------

class TrilinearCellEncoder(nn.Module):
    """Encodes a cell of k tokens into a single representation using
    CP-decomposed trilinear tensor contraction.

    For tokens t1..tk in a cell, computes all 3rd-order interactions:
        cell = Σ_r (A_r · t_i)(B_r · t_j)(C_r · t_k)
    summed over all (i,j,k) triples within the cell.

    CP decomposition keeps this O(k^3 * R * d) instead of O(k^3 * d^3).
    """

    def __init__(self, d_model: int, cell_size: int, rank: int):
        super().__init__()
        self.cell_size = cell_size
        self.rank = rank

        # CP factors: three projection matrices, each (d_model -> rank)
        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(d_model, rank, bias=False)
        self.C = nn.Linear(d_model, rank, bias=False)

        # Project the trilinear output back to d_model
        self.out_proj = nn.Linear(rank, d_model, bias=False)

        # Also keep a residual path: simple mean-pool + linear
        self.residual_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, cell_size, d_model)
        Returns:
            cell_repr: (batch, n_cells, d_model)
        """
        B, N, K, D = x.shape

        # CP-decomposed trilinear interaction
        a = self.A(x)  # (B, N, K, R)
        b = self.B(x)  # (B, N, K, R)
        c = self.C(x)  # (B, N, K, R)

        # Sum over all tokens in the cell for each factor, then element-wise product
        # This computes Σ_{i,j,k} (A·t_i)(B·t_j)(C·t_k) via factored sums
        a_sum = a.sum(dim=2)  # (B, N, R)
        b_sum = b.sum(dim=2)  # (B, N, R)
        c_sum = c.sum(dim=2)  # (B, N, R)
        trilinear = a_sum * b_sum * c_sum  # (B, N, R)

        trilinear_out = self.out_proj(trilinear)  # (B, N, D)

        # Residual path: mean pooling
        residual = self.residual_proj(x.mean(dim=2))  # (B, N, D)

        # Gated fusion
        combined = torch.cat([trilinear_out, residual], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        cell_repr = gate * trilinear_out + (1 - gate) * residual

        return self.norm(cell_repr)


# ---------------------------------------------------------------------------
# Stage 2: Multi-directional grid attention
# ---------------------------------------------------------------------------

class DirectionalAttention(nn.Module):
    """Causal attention along a single direction on a 2D grid of cells."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            causal_mask: (seq_len, seq_len) boolean mask, True = attend
        """
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build float mask for SDPA
        attn_mask = torch.where(causal_mask, 0.0, float("-inf"))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(y)


class MultiDirectionalGridAttention(nn.Module):
    """Four directional attention streams over a 2D grid, fused via learned gating."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Four direction streams
        self.attn_lr = DirectionalAttention(d_model, n_heads, dropout)
        self.attn_rl = DirectionalAttention(d_model, n_heads, dropout)
        self.attn_td = DirectionalAttention(d_model, n_heads, dropout)
        self.attn_bu = DirectionalAttention(d_model, n_heads, dropout)

        # Gated fusion
        self.gate = nn.Linear(d_model * 4, d_model * 4)
        self.out_proj = nn.Linear(d_model * 4, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def _build_masks(self, H: int, W: int, device: torch.device):
        """Build causal masks for 4 directions on an HxW grid."""
        N = H * W

        def grid_pos(idx):
            return idx // W, idx % W

        # Left-to-right: attend to cells in same row with col <= current col,
        # and all cells in previous rows
        lr_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        rl_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        td_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        bu_mask = torch.zeros(N, N, dtype=torch.bool, device=device)

        for i in range(N):
            ri, ci = grid_pos(i)
            for j in range(N):
                rj, cj = grid_pos(j)
                # L→R: raster order (row-major, left to right)
                lr_mask[i, j] = (ri > rj) or (ri == rj and ci >= cj)
                # R→L: reverse raster
                rl_mask[i, j] = (ri < rj) or (ri == rj and ci <= cj)
                # Top→Down: column-major, top to bottom
                td_mask[i, j] = (ci > cj) or (ci == cj and ri >= rj)
                # Bottom→Up: reverse column-major
                bu_mask[i, j] = (ci < cj) or (ci == cj and ri <= rj)

        return lr_mask, rl_mask, td_mask, bu_mask

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, d_model) — flattened grid in row-major order
            grid_h, grid_w: grid dimensions
        """
        masks = self._build_masks(grid_h, grid_w, x.device)

        # Reorder for each direction
        h_lr = self.attn_lr(x, masks[0])
        h_rl = self.attn_rl(x, masks[1])
        h_td = self.attn_td(x, masks[2])
        h_bu = self.attn_bu(x, masks[3])

        # Gated fusion
        concat = torch.cat([h_lr, h_rl, h_td, h_bu], dim=-1)  # (B, N, 4D)
        gate = torch.sigmoid(self.gate(concat))
        fused = self.out_proj(gate * concat)

        return self.norm(fused + x)


class GridTransformerBlock(nn.Module):
    def __init__(self, config: BrailleFormerConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = MultiDirectionalGridAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.ln2 = RMSNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.SiLU(),
            nn.Linear(config.d_ff, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        x = self.attn(self.ln1(x), grid_h, grid_w)
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Stage 3: Cell Decoder — expand cell representations back to token-level
# ---------------------------------------------------------------------------

class CellDecoder(nn.Module):
    """Broadcast cell representations back to per-token predictions."""

    def __init__(self, d_model: int, cell_size: int):
        super().__init__()
        self.cell_size = cell_size
        # Learned position-within-cell embeddings
        self.pos_embed = nn.Parameter(torch.randn(cell_size, d_model) * 0.02)
        self.refine = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.norm = RMSNorm(d_model)

    def forward(self, cell_repr: torch.Tensor, token_residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cell_repr: (batch, n_cells, d_model)
            token_residual: (batch, n_cells, cell_size, d_model) — original token embeddings
        Returns:
            (batch, seq_len, d_model)
        """
        B, N, D = cell_repr.shape
        # Broadcast cell repr to each token position within the cell
        expanded = cell_repr.unsqueeze(2).expand(B, N, self.cell_size, D)
        # Add position-within-cell information
        expanded = expanded + self.pos_embed.unsqueeze(0).unsqueeze(0)
        # Add residual from original token embeddings
        refined = self.refine(expanded + token_residual)
        output = self.norm(refined + expanded)
        return output.reshape(B, N * self.cell_size, D)


# ---------------------------------------------------------------------------
# Full BrailleFormer
# ---------------------------------------------------------------------------

class BrailleFormer(nn.Module):
    def __init__(self, config: BrailleFormerConfig):
        super().__init__()
        self.config = config
        self.cell_size = config.cell_size

        # Token embedding (allocate extra positions for cell_size padding)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        max_pos = config.max_seq_len + config.cell_size
        self.pos_emb = nn.Embedding(max_pos, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Stage 1: Cell encoder
        self.cell_encoder = TrilinearCellEncoder(
            config.d_model, config.cell_size, config.tensor_rank
        )

        # Stage 2: Grid transformer blocks
        self.blocks = nn.ModuleList([
            GridTransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Stage 3: Cell decoder
        self.cell_decoder = CellDecoder(config.d_model, config.cell_size)

        # Output head
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _compute_grid_dims(self, n_cells: int) -> tuple[int, int]:
        """Find grid dimensions closest to square."""
        h = int(math.sqrt(n_cells))
        while n_cells % h != 0 and h > 1:
            h -= 1
        w = n_cells // h
        return h, w

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        k = self.cell_size

        # Pad sequence to multiple of cell_size
        pad_len = (k - T % k) % k
        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            if targets is not None:
                targets = F.pad(targets, (0, pad_len), value=-100)

        T_padded = input_ids.shape[1]
        n_cells = T_padded // k

        # Token embeddings
        positions = torch.arange(T_padded, device=input_ids.device).unsqueeze(0)
        tok = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))

        # Reshape into cells: (B, n_cells, cell_size, d_model)
        tok_cells = tok.reshape(B, n_cells, k, self.config.d_model)

        # Stage 1: Encode cells
        cell_repr = self.cell_encoder(tok_cells)  # (B, n_cells, d_model)

        # Stage 2: Grid attention
        grid_h, grid_w = self._compute_grid_dims(n_cells)
        for block in self.blocks:
            cell_repr = block(cell_repr, grid_h, grid_w)

        # Stage 3: Decode back to tokens
        token_out = self.cell_decoder(cell_repr, tok_cells)  # (B, T_padded, d_model)

        # Output
        logits = self.lm_head(self.ln_f(token_out))

        # Remove padding from logits
        if pad_len > 0:
            logits = logits[:, :T, :]
            if targets is not None:
                targets = targets[:, :T]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
