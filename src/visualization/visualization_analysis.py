import pandas as pd
import numpy as np
import gurobipy as gp
from itertools import product
import matplotlib.pyplot as plt
from typing import List, Dict
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def create_visualizations(summary_df: pd.DataFrame, output_dir: str = "."):
    """
    Create comprehensive visualization suite for epsilon-constraint results.
    
    Args:
        summary_df: DataFrame with parameter sweep results
        output_dir: Directory to save plots
    """
    # Extract data
    lambdas = summary_df['lambda_diversification'].values
    ratios = summary_df['gain_loss_ratio'].values
    eff_ns = summary_df['effective_n'].values
    holdings = summary_df['num_holdings'].values
    concentrations = summary_df['concentration'].values
    top5s = summary_df['top5_concentration'].values
    top10s = summary_df['top10_concentration'].values
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 1. Main Efficient Frontier (Large, spanning 2 columns)
    # ═══════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(eff_ns, ratios, 'o-', linewidth=3, markersize=12, 
             color='steelblue', alpha=0.8)
    
    # Highlight recommended range
    mask_recommended = (lambdas >= 0.5) & (lambdas <= 0.9)
    ax1.scatter(eff_ns[mask_recommended], ratios[mask_recommended], 
               s=400, color='lightgreen', marker='o', 
               edgecolors='darkgreen', linewidth=3, zorder=10, 
               label='Recommended Range (λ=0.5-0.9)', alpha=0.7)
    
    # Highlight λ=0.7 sweet spot
    idx_07 = np.argmin(np.abs(lambdas - 0.7))
    ax1.scatter([eff_ns[idx_07]], [ratios[idx_07]], 
               s=600, color='gold', marker='*', 
               edgecolors='black', linewidth=3, zorder=11, 
               label=f'λ=0.7 (Sweet Spot)')
    
    # Annotate key points
    for i in [0, idx_07, len(lambdas)-1]:
        ax1.annotate(f'λ={lambdas[i]:.1f}\n{holdings[i]} holdings', 
                    xy=(eff_ns[i], ratios[i]),
                    xytext=(15, 15 if i != len(lambdas)-1 else -25), 
                    textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor='yellow' if i == idx_07 else 'white', 
                             alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax1.set_xlabel('Effective N (Diversification)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Gain-Loss Ratio (Performance)', fontsize=14, fontweight='bold')
    ax1.set_title('Efficient Frontier: Performance vs Diversification Trade-off', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 2. Concentration Metrics
    # ═══════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 2])
    
    ax2.plot(lambdas, top5s * 100, 'o-', linewidth=3, markersize=8, 
            label='Top 5', color='darkred')
    ax2.plot(lambdas, top10s * 100, 's-', linewidth=3, markersize=8, 
            label='Top 10', color='darkorange')
    
    # Risk zones
    ax2.axhspan(50, 100, alpha=0.2, color='red', label='High Risk (>50%)')
    ax2.axhspan(30, 50, alpha=0.2, color='yellow', label='Moderate Risk')
    ax2.axhspan(0, 30, alpha=0.2, color='green', label='Low Risk (<30%)')
    
    # Highlight recommended range
    ax2.axvspan(0.5, 0.9, alpha=0.1, color='green')
    
    ax2.set_xlabel('Lambda', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Concentration (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Top Holdings Concentration', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # ═══════════════════════════════════════════════════════════════════════
    # 3. Number of Holdings
    # ═══════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 2])
    
    colors = ['red' if h < 30 else 'orange' if h < 60 else 
              'lightgreen' if h < 100 else 'darkgreen' for h in holdings]
    bars = ax3.bar(lambdas, holdings, color=colors, edgecolor='black', 
                   linewidth=2, alpha=0.7)
    
    # Highlight recommended range
    ax3.axvspan(0.5, 0.9, alpha=0.15, color='green', label='Recommended')
    
    # Reference lines
    ax3.axhline(y=50, color='blue', linestyle='--', linewidth=2, 
               alpha=0.5, label='Min Institutional (50)')
    
    ax3.set_xlabel('Lambda', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Holdings', fontsize=12, fontweight='bold')
    ax3.set_title('Portfolio Cardinality', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ═══════════════════════════════════════════════════════════════════════
    # 4. Performance Loss
    # ═══════════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[0, 3])
    
    pct_loss = [(r / ratios[0] - 1) * 100 for r in ratios]
    colors_loss = ['lightgreen' if p > -10 else 'yellow' if p > -20 else 
                   'orange' if p > -50 else 'red' for p in pct_loss]
    
    bars = ax4.bar(lambdas, pct_loss, color=colors_loss, edgecolor='black', 
                   linewidth=2, alpha=0.7)
    
    # Reference lines
    ax4.axhline(y=-10, color='green', linestyle='--', linewidth=2, 
               alpha=0.7, label='-10% (Acceptable)')
    ax4.axhline(y=-20, color='orange', linestyle='--', linewidth=2, 
               alpha=0.7, label='-20% (Tolerable)')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    # Highlight recommended range
    ax4.axvspan(0.5, 0.9, alpha=0.15, color='green')
    
    ax4.set_xlabel('Lambda', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Performance Loss (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Cost of Diversification', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ═══════════════════════════════════════════════════════════════════════
    # 5. Marginal Cost Analysis
    # ═══════════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[1, 3])
    
    marginal_costs = []
    for i in range(1, len(ratios)):
        delta_ratio = abs(ratios[i] - ratios[i-1])
        delta_eff_n = eff_ns[i] - eff_ns[i-1]
        marginal_costs.append(delta_ratio / delta_eff_n if delta_eff_n > 0 else 0)
    
    ax5.plot(lambdas[1:], marginal_costs, 'o-', linewidth=3, markersize=10, 
            color='darkred')
    
    # Highlight recommended range
    mask_recommended_mc = (lambdas[1:] >= 0.5) & (lambdas[1:] <= 0.9)
    ax5.scatter(lambdas[1:][mask_recommended_mc], 
               np.array(marginal_costs)[mask_recommended_mc],
               s=300, color='lightgreen', edgecolors='darkgreen', 
               linewidth=3, zorder=10, alpha=0.7)
    
    ax5.set_xlabel('Lambda', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Marginal Cost\n(|ΔRatio| / ΔEff_N)', fontsize=12, fontweight='bold')
    ax5.set_title('Marginal Cost of Diversification', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 6. Risk-Return Heatmap (2x2 space)
    # ═══════════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[2, 0:2])
    
    # Create heatmap data
    metrics = ['Gain-Loss Ratio', 'Effective N', 'Holdings', 
               'Top 5 Conc %', 'Top 10 Conc %']
    heatmap_data = []
    
    # Select representative lambdas
    lambda_display = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    indices = [np.argmin(np.abs(lambdas - l)) for l in lambda_display]
    
    for idx in indices:
        heatmap_data.append([
            ratios[idx],
            eff_ns[idx],
            holdings[idx],
            top5s[idx] * 100,
            top10s[idx] * 100
        ])
    
    heatmap_data = np.array(heatmap_data).T
    
    # Normalize for color mapping
    heatmap_normalized = np.zeros_like(heatmap_data)
    for i in range(len(metrics)):
        row = heatmap_data[i]
        if i == 0:  # Ratio - higher is better
            heatmap_normalized[i] = (row - row.min()) / (row.max() - row.min())
        elif i in [1, 2]:  # Eff N, Holdings - higher is better
            heatmap_normalized[i] = (row - row.min()) / (row.max() - row.min())
        else:  # Concentration - lower is better
            heatmap_normalized[i] = 1 - (row - row.min()) / (row.max() - row.min())
    
    im = ax6.imshow(heatmap_normalized, cmap='RdYlGn', aspect='auto', 
                    vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(lambda_display)):
            value = heatmap_data[i, j]
            if i == 0:
                text = f'{value:.4f}'
            elif i <= 2:
                text = f'{int(value)}'
            else:
                text = f'{value:.1f}%'
            
            ax6.text(j, i, text, ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='black' if heatmap_normalized[i, j] > 0.5 else 'white')
    
    ax6.set_xticks(range(len(lambda_display)))
    ax6.set_xticklabels([f'λ={l:.1f}' for l in lambda_display], fontsize=11)
    ax6.set_yticks(range(len(metrics)))
    ax6.set_yticklabels(metrics, fontsize=11)
    ax6.set_title('Performance Metrics Heatmap (Green = Good)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Normalized Score', fontsize=11, fontweight='bold')
    
    # Highlight recommended column
    for idx in indices:
        if 0.5 <= lambdas[idx] <= 0.9:
            col_idx = indices.index(idx)
            rect = plt.Rectangle((col_idx - 0.5, -0.5), 1, len(metrics),
                               fill=False, edgecolor='green', linewidth=4)
            ax6.add_patch(rect)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 7. Summary Table
    # ═══════════════════════════════════════════════════════════════════════
    ax7 = fig.add_subplot(gs[2, 2:4])
    ax7.axis('off')
    
    # Focus on recommended range
    recommended_indices = [np.argmin(np.abs(lambdas - l)) for l in [0.5, 0.7, 0.9]]
    
    summary_data = []
    for idx in recommended_indices:
        summary_data.append([
            f"{lambdas[idx]:.1f}",
            f"{ratios[idx]:.4f}",
            f"{int(eff_ns[idx])}",
            f"{int(holdings[idx])}",
            f"{top5s[idx]*100:.1f}%",
            f"{top10s[idx]*100:.1f}%",
            f"{(ratios[idx]/ratios[0]-1)*100:.1f}%"
        ])
    
    table = ax7.table(
        cellText=summary_data,
        colLabels=['λ', 'Ratio', 'Eff N', 'Holdings', 'Top 5', 'Top 10', 'Loss%'],
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.15, 0.12, 0.12, 0.13, 0.13, 0.13]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Color code rows
    colors_rows = ['lightgreen', 'gold', 'lightgreen']
    for i, color in enumerate(colors_rows, start=1):
        for j in range(7):
            table[(i, j)].set_facecolor(color)
            if i == 2:  # λ=0.7 row
                table[(i, j)].set_text_props(weight='bold')
    
    # Style header
    for j in range(7):
        table[(0, j)].set_facecolor('lightblue')
        table[(0, j)].set_text_props(weight='bold')
    
    ax7.set_title('Recommended Configurations\n(λ=0.7 Highlighted)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('Epsilon-Constraint Multi-Objective Optimization Analysis\n' + 
                'Recommended Range: λ ∈ [0.5, 0.9] for Institutional Portfolios',
                fontsize=18, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig(f'{output_dir}/epsilon_constraint_analysis.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_dir}/epsilon_constraint_analysis.png")
    
    plt.show()


def print_detailed_analysis(summary_df: pd.DataFrame):
    """Print detailed text analysis of results."""
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS: EPSILON-CONSTRAINT RESULTS")
    print("="*80)
    
    # Extract data
    lambdas = summary_df['lambda_diversification'].values
    ratios = summary_df['gain_loss_ratio'].values
    eff_ns = summary_df['effective_n'].values
    holdings = summary_df['num_holdings'].values
    top5s = summary_df['top5_concentration'].values
    
    # Marginal analysis
    print("\n" + "-"*80)
    print("MARGINAL COST ANALYSIS")
    print("-"*80)
    print(f"{'λ Range':^12} | {'ΔRatio':^10} | {'ΔEff_N':^10} | {'Marg Cost':^12} | {'Value':^15}")
    print("-"*80)
    
    for i in range(1, len(ratios)):
        delta_ratio = abs(ratios[i] - ratios[i-1])
        delta_eff_n = eff_ns[i] - eff_ns[i-1]
        marg_cost = delta_ratio / delta_eff_n if delta_eff_n > 0 else 0
        
        # Classify value
        if 0.5 <= lambdas[i] <= 0.9:
            if marg_cost < 0.0002:
                value = "⭐ Excellent"
            elif marg_cost < 0.0003:
                value = "✓ Good"
            else:
                value = "○ Fair"
        else:
            value = "Outside range"
        
        print(f"{lambdas[i-1]:.1f}→{lambdas[i]:.1f}  | "
              f"{delta_ratio:>9.6f} | "
              f"{delta_eff_n:>9.0f} | "
              f"{marg_cost:>11.8f} | "
              f"{value:^15}")
    
    # Risk analysis
    print("\n" + "-"*80)
    print("SINGLE-NAME RISK ANALYSIS")
    print("-"*80)
    print(f"{'λ':^6} | {'Holdings':^10} | {'Max Wgt':^10} | {'Top 5':^10} | {'Risk Level':^20}")
    print("-"*80)
    
    for i in range(len(lambdas)):
        max_wgt = summary_df.iloc[i]['max_weight']
        
        # Assess risk
        if top5s[i] > 0.5 or max_wgt > 0.10:
            risk = "🔴 HIGH"
        elif top5s[i] > 0.35 or max_wgt > 0.08:
            risk = "🟡 MODERATE"
        elif top5s[i] > 0.25 or max_wgt > 0.05:
            risk = "🟢 LOW"
        else:
            risk = "✅ MINIMAL"
        
        highlight = " ⭐" if 0.5 <= lambdas[i] <= 0.9 else ""
        
        print(f"{lambdas[i]:>5.1f} | "
              f"{int(holdings[i]):>10} | "
              f"{max_wgt:>9.1%} | "
              f"{top5s[i]:>9.1%} | "
              f"{risk:^20}{highlight}")
    
    # Recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    idx_07 = np.argmin(np.abs(lambdas - 0.7))
    idx_05 = np.argmin(np.abs(lambdas - 0.5))
    idx_09 = np.argmin(np.abs(lambdas - 0.9))
    
    print("\n🥇 PRIMARY RECOMMENDATION: λ = 0.7")
    print(f"   Ratio: {ratios[idx_07]:.4f} ({(ratios[idx_07]/ratios[0]-1)*100:.1f}% vs optimal)")
    print(f"   Holdings: {int(holdings[idx_07])} (Eff N: {int(eff_ns[idx_07])})")
    print(f"   Concentration: Top 5 = {top5s[idx_07]*100:.1f}%, Max = {summary_df.iloc[idx_07]['max_weight']*100:.1f}%")
    print(f"   Risk Profile: Institutional standard, passes most mandates")
    
    print("\n🥈 AGGRESSIVE ALTERNATIVE: λ = 0.5")
    print(f"   Ratio: {ratios[idx_05]:.4f} ({(ratios[idx_05]/ratios[0]-1)*100:.1f}% vs optimal)")
    print(f"   Holdings: {int(holdings[idx_05])} (Eff N: {int(eff_ns[idx_05])})")
    print(f"   Concentration: Top 5 = {top5s[idx_05]*100:.1f}%, Max = {summary_df.iloc[idx_05]['max_weight']*100:.1f}%")
    print(f"   Risk Profile: High conviction, requires strong risk tolerance")
    
    print("\n🥉 CONSERVATIVE ALTERNATIVE: λ = 0.9")
    print(f"   Ratio: {ratios[idx_09]:.4f} ({(ratios[idx_09]/ratios[0]-1)*100:.1f}% vs optimal)")
    print(f"   Holdings: {int(holdings[idx_09])} (Eff N: {int(eff_ns[idx_09])})")
    print(f"   Concentration: Top 5 = {top5s[idx_09]*100:.1f}%, Max = {summary_df.iloc[idx_09]['max_weight']*100:.1f}%")
    print(f"   Risk Profile: Maximum diversification, minimal single-name risk")
    
    print("\n" + "="*80)