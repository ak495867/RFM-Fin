"""
===============================================================================
RFM-Fin: Real Market Data Pipeline (Strict Zero-Leakage Engine)
===============================================================================
Downloads multi-asset OHLCV data across 30 liquid assets:
  - Mega-Cap Tech / Growth (AAPL, MSFT, GOOGL, AMZN, NVDA, META)
  - Cyclical / Value (JPM, XOM, JNJ, PG, HD, CVX)
  - Sector ETFs (XLK, XLF, XLE, XLV, XLI, XLP)
  - Rates / Credit (TLT, IEF, HYG, LQD)
  - Commodities / Currency (GLD, SLV, USO, UUP)
  - Broad / International (SPY, QQQ, EEM, EFA)

Data Splitting & Zero-Leakage Architecture:
  - Lookback Context Window: L = 60 trading days (~1 quarter)
  - Forward Forecast Horizon: H = 20 trading days (~1 month)
  - Target Covariance: Realized forward covariance computed strictly on [t+1, t+H]
  - Chronological splits with strict buffer:
      Train: 2018-01-01 to 2022-12-31
      Val:   2023-01-01 to 2023-12-31 (gap of L days ensures zero window overlap)
      Test:  2024-01-01 to 2026-03-01 (unseen out-of-sample forward evaluation)
===============================================================================
"""

import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional


ASSET_UNIVERSE = [
    # Mega-Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
    # Value / Cyclicals
    'JPM', 'XOM', 'JNJ', 'PG', 'HD', 'CVX',
    # Sector SPDRs
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP',
    # Fixed Income / Credit
    'TLT', 'IEF', 'HYG', 'LQD',
    # Commodities / Currencies
    'GLD', 'SLV', 'USO', 'UUP',
    # Broad & International
    'SPY', 'QQQ', 'EEM', 'EFA'
]


def sym_matrix_log(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Numerically stable matrix logarithm S_{++}^N -> Sym(N).
    Enforces exact symmetry and clamps eigenvalues away from zero.
    """
    A_sym = 0.5 * (A + A.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)
    clamped_evals = torch.clamp(eigenvalues, min=eps)
    log_evals = torch.log(clamped_evals)
    return eigenvectors @ torch.diag_embed(log_evals) @ eigenvectors.transpose(-1, -2)


def sym_matrix_exp(M: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
    """
    Matrix exponential Sym(N) -> S_{++}^N.
    Guarantees strictly positive eigenvalues.
    """
    M_sym = 0.5 * (M + M.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(M_sym)
    clamped_evals = torch.clamp(eigenvalues, max=max_val)
    exp_evals = torch.exp(clamped_evals)
    return eigenvectors @ torch.diag_embed(exp_evals) @ eigenvectors.transpose(-1, -2)


def download_market_data(
    tickers: List[str] = ASSET_UNIVERSE,
    start_date: str = '2018-01-01',
    end_date: str = '2026-03-01',
    cache_path: str = 'd:/RFM-Fin/data/market_prices.csv'
) -> pd.DataFrame:
    """
    Downloads historical adjusted close prices with local disk caching.
    """
    if os.path.exists(cache_path):
        print(f"[Data] Loading cached prices from {cache_path}...")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print(f"[Data] Downloading historical prices for {len(tickers)} assets ({start_date} to {end_date})...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    # Handle multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.levels[0]:
            df = data['Close']
        elif 'Adj Close' in data.columns.levels[0]:
            df = data['Adj Close']
        else:
            df = data.xs('Close', axis=1, level=0)
    else:
        df = data

    # Reorder columns to match universe and drop missing dates
    df = df[tickers].dropna()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path)
    print(f"[Data] Successfully saved {len(df)} trading days to {cache_path}.")
    return df


class FinancialManifoldDataset(Dataset):
    """
    PyTorch Dataset producing strictly aligned, non-leaking market windows:
      - Context C_t: (L, N) past daily log returns
      - Target R_t: (H, N) forward daily log returns
      - Target Sigma_t: (N, N) realized forward covariance matrix
      - Target M_t: (N, N) Log-Euclidean tangent matrix Log(Sigma_t)
    """
    def __init__(
        self,
        returns: np.ndarray,
        dates: List[pd.Timestamp],
        start_idx: int,
        end_idx: int,
        lookback_len: int = 60,
        forecast_horizon: int = 20,
        stride: int = 5,
        eps: float = 1e-5
    ):
        super().__init__()
        self.lookback_len = lookback_len
        self.forecast_horizon = forecast_horizon
        self.num_assets = returns.shape[1]
        self.eps = eps

        # Extract valid sample indices within [start_idx, end_idx]
        self.samples = []
        # t is the last day of the lookback context
        # Context uses [t - lookback_len + 1, ..., t]
        # Target uses [t + 1, ..., t + forecast_horizon]
        min_t = start_idx + lookback_len - 1
        max_t = end_idx - forecast_horizon

        for t in range(min_t, max_t + 1, stride):
            self.samples.append(t)

        self.returns = torch.tensor(returns, dtype=torch.float32)
        self.dates = dates

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.samples[idx]

        # 1. Past Context Window (strictly [t - lookback_len + 1 : t + 1])
        c_returns = self.returns[t - self.lookback_len + 1 : t + 1] # Shape: (L, N)

        # Compute Historical Trailing Covariance and Tangent State (Empirical Prior on S_{++}^N)
        c_mean = c_returns.mean(dim=0, keepdim=True)
        c_diff = c_returns - c_mean
        c_sigma = (c_diff.T @ c_diff) / float(self.lookback_len - 1) + self.eps * torch.eye(self.num_assets, dtype=torch.float32)
        c_tangent = sym_matrix_log(c_sigma.unsqueeze(0), eps=self.eps).squeeze(0)

        # 2. Forward Horizon Window (strictly [t + 1 : t + forecast_horizon + 1])
        f_returns = self.returns[t + 1 : t + self.forecast_horizon + 1] # Shape: (H, N)

        # 3. Realized Forward Covariance Matrix
        # Unbiased sample covariance over forward H days:
        # Sigma = (1 / (H - 1)) * (R - mean)^T (R - mean) + eps * I
        mean_ret = f_returns.mean(dim=0, keepdim=True)
        diff = f_returns - mean_ret # (H, N)
        sigma = (diff.T @ diff) / float(self.forecast_horizon - 1)
        
        # Add epsilon regularizer to guarantee interior of S_{++}^N
        sigma = sigma + self.eps * torch.eye(self.num_assets, dtype=torch.float32)

        # 4. Lie algebra tangent space representation M = log(Sigma)
        m_tangent = sym_matrix_log(sigma.unsqueeze(0), eps=self.eps).squeeze(0)

        # 5. Timestamp anchor
        anchor_date = str(self.dates[t])[:10]

        return {
            'context': c_returns,
            'context_sigma': c_sigma,
            'context_m': c_tangent,
            'target_returns': f_returns,
            'target_sigma': sigma,
            'target_m': m_tangent,
            'anchor_date': anchor_date,
            't_idx': t
        }


def get_market_dataloaders(
    lookback_len: int = 60,
    forecast_horizon: int = 20,
    stride: int = 5,
    batch_size: int = 32,
    tickers: List[str] = ASSET_UNIVERSE
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, List[str]]:
    """
    Constructs train, validation, and test dataloaders with strict zero-leakage dates:
      - Train: 2018-01-01 to 2022-12-31
      - Val:   2023-01-01 to 2023-12-31
      - Test:  2024-01-01 to 2026-03-01
    """
    df_prices = download_market_data(tickers=tickers)
    
    # Compute daily log returns: r_t = log(P_t / P_{t-1})
    df_returns = np.log(df_prices / df_prices.shift(1)).dropna()
    dates = list(df_returns.index)
    returns_arr = df_returns.values

    # Find date boundary indices
    dates_str = [d.strftime('%Y-%m-%d') for d in dates]
    
    # Split anchors
    train_end_idx = None
    val_end_idx = None

    for i, d in enumerate(dates_str):
        if d <= '2022-12-31':
            train_end_idx = i
        if d <= '2023-12-31':
            val_end_idx = i

    print(f"[Data] Total valid trading days: {len(dates)}")
    print(f"[Data] Train split: 0 to {train_end_idx} ({dates_str[0]} to {dates_str[train_end_idx]})")
    print(f"[Data] Val split:   {train_end_idx + 1} to {val_end_idx} ({dates_str[train_end_idx + 1]} to {dates_str[val_end_idx]})")
    print(f"[Data] Test split:  {val_end_idx + 1} to {len(dates) - 1} ({dates_str[val_end_idx + 1]} to {dates_str[-1]})")

    # Datasets
    train_dataset = FinancialManifoldDataset(
        returns=returns_arr,
        dates=dates,
        start_idx=0,
        end_idx=train_end_idx,
        lookback_len=lookback_len,
        forecast_horizon=forecast_horizon,
        stride=stride
    )

    val_dataset = FinancialManifoldDataset(
        returns=returns_arr,
        dates=dates,
        start_idx=train_end_idx + 1,
        end_idx=val_end_idx,
        lookback_len=lookback_len,
        forecast_horizon=forecast_horizon,
        stride=stride
    )

    # For testing, we use stride=forecast_horizon (or smaller) for realistic out-of-sample rebalancing
    test_dataset = FinancialManifoldDataset(
        returns=returns_arr,
        dates=dates,
        start_idx=val_end_idx + 1,
        end_idx=len(dates) - 1,
        lookback_len=lookback_len,
        forecast_horizon=forecast_horizon,
        stride=forecast_horizon # Step forward month by month
    )

    print(f"[Data] Windows constructed -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 for sequential out-of-sample backtesting

    return train_loader, val_loader, test_loader, df_returns, tickers


if __name__ == "__main__":
    train_loader, val_loader, test_loader, df_returns, tickers = get_market_dataloaders()
    sample = next(iter(train_loader))
    print("\n[Verification] Sample Batch Shapes:")
    print("  Context:        ", sample['context'].shape)
    print("  Target Returns: ", sample['target_returns'].shape)
    print("  Target Sigma:   ", sample['target_sigma'].shape)
    print("  Target Tangent: ", sample['target_m'].shape)
    print("  Anchor Date:    ", sample['anchor_date'][0])
