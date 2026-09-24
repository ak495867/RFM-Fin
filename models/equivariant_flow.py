"""
===============================================================================
RFM-Fin: Permutation-Equivariant Riemannian Flow Architecture
===============================================================================
Implements the Asset-Cross-Attention Transformer (Equivariant DiT):
  - Strictly invariant to asset permutation P in S_N:
      v_R(R P^T, P M P^T, t, C P^T) = v_R(R, M, t, C) P^T
      v_M(R P^T, P M P^T, t, C P^T) = P v_M(R, M, t, C) P^T
  - Strictly symmetric tangent velocity: v_M = v_M^T
  - Riemannian Log-Euclidean flow matching with 5-step geodesic Euler sampler
===============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def sym_matrix_log(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes Log(A): S_{++}^N -> Sym(N) via spectral decomposition."""
    A_sym = 0.5 * (A + A.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)
    clamped_evals = torch.clamp(eigenvalues, min=eps)
    log_evals = torch.log(clamped_evals)
    return eigenvectors @ torch.diag_embed(log_evals) @ eigenvectors.transpose(-1, -2)


def sym_matrix_exp(M: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
    """Computes Exp(M): Sym(N) -> S_{++}^N strictly preserving positive definiteness."""
    M_sym = 0.5 * (M + M.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(M_sym)
    clamped_evals = torch.clamp(eigenvalues, max=max_val)
    exp_evals = torch.exp(clamped_evals)
    return eigenvectors @ torch.diag_embed(exp_evals) @ eigenvectors.transpose(-1, -2)


class SinusoidalTimeEmbedding(nn.Module):
    """Fourier time embedding for continuous flow time t in [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1)
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EquivariantTransformerBlock(nn.Module):
    """
    Permutation-Equivariant Transformer Block across asset tokens.
    Self-attention with zero positional encodings ensures exact S_N equivariance.
    Conditioned on flow time via Adaptive LayerNorm (AdaLN).
    """
    def __init__(self, d_model: int, num_heads: int = 4, d_ff: int = 256):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        # AdaLN modulation generator: produces scale and shift for norm1 and norm2
        self.time_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model)
        )

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable metric bias scalar for injecting tangent space distance into attention
        self.metric_bias_scale = nn.Parameter(torch.tensor(0.1))

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d_model) asset tokens
        t_emb: (B, d_model) time embedding
        m_t: (B, N, N) current tangent matrix
        """
        B, N, D = x.shape

        # AdaLN modulation parameters
        mod = self.time_mod(t_emb) # (B, 6 * D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # 1. Modulated Self-Attention
        x_norm = self.norm1(x) * (1.0 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        
        Q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, d_k)
        K = self.k_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Equivariant Attention with Riemannian metric bias
        attn_scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim) # (B, H, N, N)
        # Add metric bias: preserves equivariance because P M P^T aligns with P Q K^T P^T
        attn_scores = attn_scores + self.metric_bias_scale * m_t.unsqueeze(1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, N, D)
        attn_out = self.out_proj(attn_out)

        # Residual with adaptive gate
        x = x + alpha1.unsqueeze(1) * attn_out

        # 2. Modulated Feed-Forward
        x_norm2 = self.norm2(x) * (1.0 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        ffn_out = self.ffn(x_norm2)
        x = x + alpha2.unsqueeze(1) * ffn_out

        return x


class AssetEquivariantMarketFlowNet(nn.Module):
    """
    Institutional Riemannian Flow Matching Network for Multi-Asset Markets.
    
    Guarantees:
      1. Permutation equivariance across asset dimensions under any permutation P in S_N
      2. Strict mathematical symmetry of predicted Lie algebra velocity: v_M = v_M^T
      3. Exact integration on the Riemannian SPD manifold cone S_{++}^N
    """
    def __init__(
        self,
        num_assets: int = 30,
        horizon: int = 20,
        lookback_len: int = 60,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4
    ):
        super().__init__()
        self.num_assets = num_assets
        self.horizon = horizon
        self.lookback_len = lookback_len
        self.d_model = d_model

        # 1. Flow Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # 2. Asset Context Feature Extractor (shared 1D convolution per asset across time)
        # Input per asset: lookback_len daily returns -> produces d_ctx features
        self.context_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        ) # Output: 64 dims per asset

        # 3. Asset Token Projection
        # Per asset inputs: Context (64) + Return state (horizon) + Self-variance (1) + Cross-covariance (1)
        in_per_asset = 64 + horizon + 1 + 1
        self.asset_in_proj = nn.Linear(in_per_asset, d_model)

        # 4. Equivariant Transformer Backbone
        self.blocks = nn.ModuleList([
            EquivariantTransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_model * 2)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # 5. Dual Equivariant Velocity Heads
        # Returns Head: predicts velocity for forward horizon H
        self.return_velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, horizon)
        )

        # Covariance Tangent Head:
        # Diagonal head:
        self.cov_diag_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1)
        )
        # Pairwise bilinear projection for off-diagonals (guarantees exact symmetry and equivariance)
        self.cov_u_proj = nn.Linear(d_model, 32)
        self.cov_v_proj = nn.Linear(d_model, 32)
        self.cov_res_weight = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        R_t: torch.Tensor,
        M_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        R_t: (B, H, N) current return trajectory
        M_t: (B, N, N) current tangent matrix in Lie algebra Sym(N)
        t: (B, 1) flow matching time
        context: (B, L, N) past lookback returns
        """
        B, H, N = R_t.shape
        L = context.shape[1]

        # 1. Flow Time Embedding
        t_emb = self.time_embed(t) # (B, d_model)

        # 2. Context Feature Extraction per asset
        # Reshape: (B, L, N) -> (B * N, 1, L)
        ctx_trans = context.transpose(1, 2).reshape(B * N, 1, L)
        ctx_feats = self.context_encoder(ctx_trans).squeeze(-1).view(B, N, 64)

        # 3. Return Features per asset
        # (B, H, N) -> (B, N, H)
        r_feats = R_t.transpose(1, 2)

        # 4. Matrix Features per asset
        # Diagonal log-variance: (B, N, 1)
        m_diag = torch.diagonal(M_t, dim1=1, dim2=2).unsqueeze(-1)
        
        # Mean off-diagonal cross interaction: sum_{j != i} M_{ij} / (N - 1)
        m_sum = M_t.sum(dim=-1, keepdim=True) - m_diag
        m_cross = m_sum / float(max(1, N - 1)) # (B, N, 1)

        # Combine into initial asset node tokens
        asset_raw = torch.cat([ctx_feats, r_feats, m_diag, m_cross], dim=-1) # (B, N, in_per_asset)
        tokens = self.asset_in_proj(asset_raw) # (B, N, d_model)

        # 5. Transformer Equivariant Processing
        for block in self.blocks:
            tokens = block(tokens, t_emb, M_t)

        tokens = self.final_norm(tokens) # (B, N, d_model)

        # 6. Returns Velocity: (B, N, H) -> (B, H, N)
        v_R = self.return_velocity_head(tokens).transpose(1, 2)

        # 7. Covariance Tangent Velocity: Strictly Symmetric & Equivariant
        # Diagonal terms: (B, N)
        v_diag = self.cov_diag_head(tokens).squeeze(-1)

        # Off-diagonal bilinear forms:
        U = self.cov_u_proj(tokens) # (B, N, 32)
        V = self.cov_v_proj(tokens) # (B, N, 32)
        
        # Symmetrized bilinear product: 0.5 * (U @ V^T + V @ U^T)
        v_bilinear = 0.5 * (U @ V.transpose(-1, -2) + V @ U.transpose(-1, -2))
        
        # Add residual connection from current tangent state
        v_M = v_bilinear + self.cov_res_weight * M_t

        # Replace diagonal with dedicated diagonal head
        diag_idx = torch.arange(N, device=R_t.device)
        v_M[:, diag_idx, diag_idx] = v_diag

        # Final exact symmetry guard
        v_M = 0.5 * (v_M + v_M.transpose(-1, -2))

        return v_R, v_M


@torch.no_grad()
def sample_geodesic_flow(
    model: AssetEquivariantMarketFlowNet,
    context: torch.Tensor,
    num_steps: int = 5,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast Geodesic ODE Solver on S_{++}^N (5-step Euler Integration).
    Initializes at Identity Prior: log(I_N) = 0.
    Integrates straight tangent vector field.
    Retracts to SPD manifold via Matrix Exponential: Sigma = Exp(M_1).
    """
    model.eval()
    B, L, N = context.shape
    H = model.horizon

    # Prior Base Sample at t = 0
    R = torch.randn(B, H, N, device=device) * 0.05
    # Uninformative Identity Prior on SPD manifold: log(I_N) = 0
    M = torch.zeros(B, N, N, device=device)

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((B, 1), t_val, device=device, dtype=torch.float32)

        # Vector field evaluation
        v_R, v_M = model(R, M, t_tensor, context)

        # Euler geodesic integration in tangent space
        R = R + v_R * dt
        M = M + v_M * dt

    # Manifold Retraction to S_{++}^N
    final_covariance = sym_matrix_exp(M)
    return R, final_covariance


def verify_permutation_equivariance():
    """
    Unit test mathematically verifying that AssetEquivariantMarketFlowNet
    is strictly equivariant under any arbitrary asset permutation matrix P.
    """
    print("\n" + "="*75)
    print("UNIT TEST: PERMUTATION EQUIVARIANCE VERIFICATION UNDER S_N")
    print("="*75)

    B, N, H, L = 2, 6, 10, 30
    model = AssetEquivariantMarketFlowNet(num_assets=N, horizon=H, lookback_len=L, d_model=64, num_layers=2)
    model.eval()

    # Generate random input
    R = torch.randn(B, H, N)
    A = torch.randn(B, N, N)
    M = 0.5 * (A + A.transpose(-1, -2))
    t = torch.tensor([[0.35], [0.72]])
    c = torch.randn(B, L, N)

    # 1. Base Forward Pass
    v_R, v_M = model(R, M, t, c)

    # 2. Construct Random Permutation Matrix P
    perm_idx = torch.randperm(N)
    P = torch.eye(N)[perm_idx] # (N, N)

    # Permuted Inputs:
    # R_perm = R @ P^T
    # M_perm = P @ M @ P^T
    # c_perm = c @ P^T
    R_perm = R @ P.T
    M_perm = P.unsqueeze(0) @ M @ P.T.unsqueeze(0)
    c_perm = c @ P.T

    # 3. Permuted Forward Pass
    v_R_perm, v_M_perm = model(R_perm, M_perm, t, c_perm)

    # 4. Check Equivariance Relations:
    # Expected v_R_perm == v_R @ P^T
    # Expected v_M_perm == P @ v_M @ P^T
    expected_v_R_perm = v_R @ P.T
    expected_v_M_perm = P.unsqueeze(0) @ v_M @ P.T.unsqueeze(0)

    diff_R = torch.max(torch.abs(v_R_perm - expected_v_R_perm)).item()
    diff_M = torch.max(torch.abs(v_M_perm - expected_v_M_perm)).item()
    sym_error = torch.max(torch.abs(v_M - v_M.transpose(-1, -2))).item()

    print(f"Max Returns Equivariance Error:    {diff_R:.2e}")
    print(f"Max Covariance Equivariance Error: {diff_M:.2e}")
    print(f"Matrix Symmetry Error:             {sym_error:.2e}")

    passed = (diff_R < 1e-5) and (diff_M < 1e-5) and (sym_error < 1e-7)
    print(f"Permutation Equivariance Status:   {'PASSED (Strictly Equivariant)' if passed else 'FAILED'}")
    print("="*75 + "\n")
    return passed


if __name__ == "__main__":
    verify_permutation_equivariance()
