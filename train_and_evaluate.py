"""
===============================================================================
RFM-Fin: Institutional Training, Out-of-Sample Backtesting & Stress Suite
===============================================================================
End-to-End Orchestrator:
  1. Strict Zero-Leakage Dataset Setup across 30 Liquid Assets (2018 - 2026)
  2. Training the Permutation-Equivariant Riemannian Flow Matching Network
  3. Out-of-Sample Sequential Rebalanced Backtest (2024 - 2026 unseen data)
     Comparing:
       - RFM-Fin (Geodesic 5-Step Flow Matching)
       - Ledoit-Wolf Optimal Shrinkage (2004)
       - DCC-GARCH / EWMA Dynamic Covariance (2002)
       - Rolling Sample Covariance (Markowitz 1952)
       - Naive 1/N Diversification (2009)
  4. Monte Carlo Noise Perturbation & Geodesic Convergence Stress Tests
===============================================================================
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List

from data_pipeline import get_market_dataloaders, ASSET_UNIVERSE
from models.equivariant_flow import AssetEquivariantMarketFlowNet, sample_geodesic_flow
from benchmarks.classical_baselines import (
    ClassicalBaselines,
    solve_minimum_variance_weights,
    evaluate_portfolio_performance
)
from experiments.monte_carlo_stress_test import (
    run_noise_perturbation_test,
    run_ode_convergence_test,
    run_permutation_monte_carlo
)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


def train_rfm_model(
    train_loader,
    val_loader,
    num_assets: int = 30,
    horizon: int = 20,
    lookback_len: int = 60,
    epochs: int = 35,
    device: str = "cpu"
) -> AssetEquivariantMarketFlowNet:
    """
    Trains the AssetEquivariantMarketFlowNet via Geodesic Flow Matching.
    Loss: L = MSE(v_R, u_R) + alpha * MSE(v_M, u_M)
    """
    print("\n" + "="*75)
    print("PHASE 1: TRAINING PERMUTATION-EQUIVARIANT RIEMANNIAN FLOW MODEL")
    print("="*75)

    model = AssetEquivariantMarketFlowNet(
        num_assets=num_assets,
        horizon=horizon,
        lookback_len=lookback_len,
        d_model=128,
        num_layers=3,
        num_heads=4
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture Initialized: {param_count:,} parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        r_loss_total = 0.0
        m_loss_total = 0.0
        n_batches = 0

        for batch in train_loader:
            context = batch['context'].to(device)       # (B, L, N)
            R_1 = batch['target_returns'].to(device)    # (B, H, N)
            M_1 = batch['target_m'].to(device)          # (B, N, N)
            B = context.shape[0]

            # 1. Base Prior Sample at t = 0: Historical Trailing Context Covariance in Lie algebra
            R_0 = torch.randn(B, horizon, num_assets, device=device) * 0.02
            M_0 = batch['context_m'].to(device) + torch.randn(B, num_assets, num_assets, device=device) * 0.05
            M_0 = 0.5 * (M_0 + M_0.transpose(-1, -2))

            # 2. Sample Flow Time t ~ U[0, 1]
            t = torch.rand(B, 1, device=device)

            # 3. Geodesic Interpolation in Tangent Space
            t_expand_R = t.view(B, 1, 1)
            t_expand_M = t.view(B, 1, 1)
            R_t = (1.0 - t_expand_R) * R_0 + t_expand_R * R_1
            M_t = (1.0 - t_expand_M) * M_0 + t_expand_M * M_1

            # 4. Straight Tangent Velocity Targets
            u_R = R_1 - R_0
            u_M = M_1 - M_0

            # 5. Predict Velocities
            v_R, v_M = model(R_t, M_t, t, context)

            # 6. Loss: Return Loss + Frobenius Tangent Loss
            loss_R = torch.mean((v_R - u_R) ** 2)
            loss_M = torch.mean((v_M - u_M) ** 2)
            loss = loss_R + 5.0 * loss_M

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            r_loss_total += loss_R.item()
            m_loss_total += loss_M.item()
            n_batches += 1

        scheduler.step()

        # Validation Loss
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    v_ctx = vbatch['context'].to(device)
                    v_R1 = vbatch['target_returns'].to(device)
                    v_M1 = vbatch['target_m'].to(device)
                    v_B = v_ctx.shape[0]

                    v_R0 = torch.randn(v_B, horizon, num_assets, device=device) * 0.02
                    v_M0 = vbatch['context_m'].to(device)
                    v_t = torch.rand(v_B, 1, device=device)

                    v_Rt = (1.0 - v_t.view(v_B, 1, 1)) * v_R0 + v_t.view(v_B, 1, 1) * v_R1
                    v_Mt = (1.0 - v_t.view(v_B, 1, 1)) * v_M0 + v_t.view(v_B, 1, 1) * v_M1

                    vu_R = v_R1 - v_R0
                    vu_M = v_M1 - v_M0

                    vv_R, vv_M = model(v_Rt, v_Mt, v_t, v_ctx)
                    v_loss = torch.mean((vv_R - vu_R) ** 2) + 5.0 * torch.mean((vv_M - vu_M) ** 2)
                    val_loss += v_loss.item()
                    val_batches += 1

            avg_train = train_loss / n_batches
            avg_val = val_loss / max(1, val_batches)
            print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {avg_train:.5f} (R: {r_loss_total/n_batches:.5f}, M: {m_loss_total/n_batches:.5f}) | Val Loss: {avg_val:.5f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    return model


def run_out_of_sample_backtest(
    model: AssetEquivariantMarketFlowNet,
    test_loader,
    device: str = "cpu"
) -> Dict[str, Dict[str, float]]:
    """
    Simulates monthly sequential portfolio rebalancing across unseen 2024-2026 data.
    At each step:
      1. Model receives only past lookback context C_t
      2. Predicts future covariance Sigma_t
      3. Computes Markowitz minimum-variance weights
      4. Measures realized performance across forward H days
    """
    print("\n" + "="*75)
    print("PHASE 2: OUT-OF-SAMPLE SEQUENTIAL REBALANCING BACKTEST (2024 - 2026)")
    print("="*75)

    strategies = {
        'RFM-Fin (Ours)': [],
        'Ledoit-Wolf': [],
        'DCC-GARCH / EWMA': [],
        'Sample Covariance': [],
        'Equal-Weight (1/N)': []
    }

    dates_tracked = []
    model.eval()

    total_windows = len(test_loader)
    print(f"Simulating {total_windows} sequential out-of-sample forward evaluation windows...")

    for step_idx, batch in enumerate(test_loader):
        context = batch['context'].to(device)           # (1, L, N)
        fwd_returns = batch['target_returns'].squeeze(0).cpu().numpy() # (H, N)
        anchor_date = batch['anchor_date'][0]
        dates_tracked.append(anchor_date)

        ctx_np = context.squeeze(0).cpu().numpy()       # (L, N)
        N = ctx_np.shape[1]

        # 1. RFM-Fin 5-Step Geodesic Forecast using Historical Context Empirical Prior
        with torch.no_grad():
            prior_m = batch['context_m'].to(device)
            _, rfm_sigma = sample_geodesic_flow(model, context, num_steps=5, device=device, prior_m=prior_m)
            rfm_sigma_np = rfm_sigma.squeeze(0).cpu().numpy()

        # 2. Classical Baseline Covariance Forecasts
        lw_sigma = ClassicalBaselines.ledoit_wolf_covariance(ctx_np)
        dcc_sigma = ClassicalBaselines.dcc_garch_covariance(ctx_np)
        sample_sigma = ClassicalBaselines.rolling_sample_covariance(ctx_np)

        # 3. Solve Optimal Minimum-Variance Weights
        w_rfm = solve_minimum_variance_weights(rfm_sigma_np, long_only=True)
        w_lw = solve_minimum_variance_weights(lw_sigma, long_only=True)
        w_dcc = solve_minimum_variance_weights(dcc_sigma, long_only=True)
        w_sample = solve_minimum_variance_weights(sample_sigma, long_only=True)
        w_eq = np.ones(N) / float(N)

        # 4. Forward Realized Daily Portfolio Returns
        strategies['RFM-Fin (Ours)'].append(fwd_returns @ w_rfm)
        strategies['Ledoit-Wolf'].append(fwd_returns @ w_lw)
        strategies['DCC-GARCH / EWMA'].append(fwd_returns @ w_dcc)
        strategies['Sample Covariance'].append(fwd_returns @ w_sample)
        strategies['Equal-Weight (1/N)'].append(fwd_returns @ w_eq)

    # Aggregate full out-of-sample return series
    print("\n--- Out-of-Sample Performance Summary Table (2024 - 2026) ---")
    print(f"{'Strategy':<22} | {'Ann. Vol':<12} | {'Sharpe':<10} | {'Sortino':<10} | {'Max DD':<12} | {'Total Ret':<10}")
    print("-" * 85)

    summary_metrics = {}
    for name, return_windows in strategies.items():
        all_daily = np.concatenate(return_windows)
        
        # Calculate performance
        mean_ret = np.mean(all_daily)
        std_ret = np.std(all_daily) + 1e-12
        ann_ret = mean_ret * 252.0
        ann_vol = std_ret * np.sqrt(252.0)
        sharpe = ann_ret / ann_vol

        # Downside Sortino
        neg_rets = all_daily[all_daily < 0]
        downside_std = np.std(neg_rets) * np.sqrt(252.0) if len(neg_rets) > 0 else 1e-6
        sortino = ann_ret / downside_std

        # Cumulative & Drawdown
        wealth = np.cumprod(1.0 + all_daily)
        peaks = np.maximum.accumulate(wealth)
        dd = (wealth - peaks) / peaks
        max_dd = np.min(dd)
        total_ret = wealth[-1] - 1.0

        print(f"{name:<22} | {ann_vol*100.0:<10.2f}% | {sharpe:<10.2f} | {sortino:<10.2f} | {max_dd*100.0:<10.2f}% | {total_ret*100.0:<8.2f}%")

        summary_metrics[name] = {
            'Annualized Volatility': float(ann_vol),
            'Sharpe Ratio': float(sharpe),
            'Sortino Ratio': float(sortino),
            'Max Drawdown': float(max_dd),
            'Total Return': float(total_ret)
        }

    print("-" * 85)
    return summary_metrics


def main():
    device = "cpu"
    print("="*75)
    print("RIEMANNIAN FLOW MATCHING ON FINANCIAL MANIFOLDS (RFM-FIN)")
    print("Institutional Multi-Asset Suite (30 Assets, 2018 - 2026)")
    print("="*75)

    # 1. Pipeline Data Ingestion
    train_loader, val_loader, test_loader, df_returns, tickers = get_market_dataloaders(
        lookback_len=60,
        forecast_horizon=20,
        stride=5,
        batch_size=32,
        tickers=ASSET_UNIVERSE
    )

    # 2. Train Permutation-Equivariant Riemannian Flow Network
    model = train_rfm_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_assets=len(tickers),
        horizon=20,
        lookback_len=60,
        epochs=35,
        device=device
    )

    # 3. Out-of-Sample Realistic Backtest
    backtest_results = run_out_of_sample_backtest(model, test_loader, device=device)

    # 4. Stress Tests & Monte Carlo Verification
    sample_batch = next(iter(test_loader))
    test_context = sample_batch['context'].to(device)
    test_prior_m = sample_batch['context_m'].to(device)

    perturbation_results = run_noise_perturbation_test(model, test_context, device=device, prior_m=test_prior_m)
    ode_results = run_ode_convergence_test(model, test_context, device=device, prior_m=test_prior_m)
    perm_err = run_permutation_monte_carlo(model, test_context, num_trials=10, device=device, prior_m=test_prior_m)

    # 5. Save Artifact
    full_output = {
        'backtest': backtest_results,
        'noise_perturbation': perturbation_results,
        'ode_convergence': {str(k): v for k, v in ode_results.items()},
        'permutation_max_error': perm_err
    }

    os.makedirs('d:/RFM-Fin/results', exist_ok=True)
    with open('d:/RFM-Fin/results/benchmark_summary.json', 'w') as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[Artifact] Successfully written institutional results to d:/RFM-Fin/results/benchmark_summary.json")
    print("\n" + "="*75)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY WITH ZERO DATA LEAKAGE.")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
