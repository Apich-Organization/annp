#!/usr/bin/env python3
"""
ANNP Training Metrics Visualizer
Generates high-resolution publication-quality plots and summary dashboard from training CSV logs.
"""

import argparse
import csv
import glob
import os
import sys

# Metric definitions with rich metadata: titles, units, colors, and formatting
METRIC_METADATA = {
    'step_loss': {
        'title': 'Step Loss Over Training',
        'ylabel': 'Loss (MSE / CE)',
        'raw_color': '#90caf9',
        'smooth_color': '#d32f2f',
        'desc': 'Instantaneous training step loss'
    },
    'ema_loss': {
        'title': 'EMA Smoothed Loss Over Training',
        'ylabel': 'EMA Loss',
        'raw_color': '#b0bec5',
        'smooth_color': '#c2185b',
        'desc': 'Exponential moving average loss'
    },
    'early_halting_rate': {
        'title': 'Particle Early Halting Rate Over Training',
        'ylabel': 'Early Halting Rate (%)',
        'raw_color': '#b2dfdb',
        'smooth_color': '#00796b',
        'percent_scale': True,
        'desc': 'Percentage of particles halted before max hops'
    },
    'gini': {
        'title': 'Node Workload Utilization (Gini Coefficient)',
        'ylabel': 'Gini Coefficient [0 = Uniform, 1 = Concentrated]',
        'raw_color': '#d1c4e9',
        'smooth_color': '#512da8',
        'desc': 'Decentralized workload balance across nodes'
    },
    'avg_hops': {
        'title': 'Average Particle Routing Hops Over Training',
        'ylabel': 'Average Hop Count',
        'raw_color': '#c8e6c9',
        'smooth_color': '#2e7d32',
        'desc': 'Average depth/hops traversed per particle'
    },
    'avg_energy': {
        'title': 'Average Signal Energy Variance Over Training',
        'ylabel': 'Signal Energy (sum of squared activations)',
        'raw_color': '#ffe082',
        'smooth_color': '#f57c00',
        'desc': 'Particle activation signal magnitude'
    },
    'memory_density': {
        'title': 'Attention Memory Density (Entropy)',
        'ylabel': 'Entropy / Memory Density',
        'raw_color': '#cfd8dc',
        'smooth_color': '#455a64',
        'desc': 'Associative memory concentration & diversity'
    },
    'active_subnodes': {
        'title': 'Active Subnodes per Node (Neurogenesis Population)',
        'ylabel': 'Average Subnodes Count',
        'raw_color': '#e1bee7',
        'smooth_color': '#7b1fa2',
        'desc': 'Average number of subnodes surviving per node'
    },
    'credit_volatility': {
        'title': 'Routing Credit Assignment Volatility',
        'ylabel': 'Credit Volatility (Reward Variance)',
        'raw_color': '#ffcdd2',
        'smooth_color': '#c62828',
        'desc': 'Variance of topological credit updates'
    },
    'temporal_affinity': {
        'title': 'Mean Temporal Affinity (Cosine Similarity)',
        'ylabel': 'Cosine Similarity: cos(p_t, p_last)',
        'raw_color': '#80deea',
        'smooth_color': '#00838f',
        'desc': 'Cosine alignment of consecutive inputs to same subnode'
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate high-quality plots for ANNP training metrics."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./stage_1/data",
        help="Directory containing training CSV files (default: ./stage_1/data)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./stage_1/plots",
        help="Directory to save generated plots (default: ./stage_1/plots)",
    )
    parser.add_argument(
        "--multiplier",
        "-m",
        type=int,
        default=2,
        help="Step multiplier for horizontal axis if logs are downsampled (default: 2)",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=None,
        help="Smoothing window size in logged points (default: auto ~40-50 points)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution DPI for exported PNG images (default: 200)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip generating the unified overview dashboard",
    )
    return parser.parse_args()


def load_csv_data(data_dir):
    csv_pattern = os.path.join(data_dir, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None, None

    print(f"Found {len(csv_files)} CSV file(s) in '{data_dir}':")
    for f in csv_files:
        print(f"  - {f}")

    data = []
    headers = None

    for f in csv_files:
        with open(f, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            try:
                file_headers = next(reader)
            except StopIteration:
                continue
            if headers is None:
                headers = [h.strip() for h in file_headers]
            for row in reader:
                if not row or len(row) != len(headers):
                    continue
                data.append(row)

    if not data:
        print("Error: No data rows found in CSV files.")
        return None, None

    # Sort data by epoch and batch if available, otherwise by timestamp
    if "epoch" in headers and "batch" in headers:
        epoch_idx = headers.index("epoch")
        batch_idx = headers.index("batch")
        try:
            data.sort(key=lambda r: (int(r[epoch_idx]), int(r[batch_idx])))
        except ValueError:
            pass
    elif "timestamp" in headers:
        ts_idx = headers.index("timestamp")
        data.sort(key=lambda r: r[ts_idx])

    print(f"Total valid log records loaded: {len(data):,}")
    return headers, data


def calculate_x_axis(data, headers, multiplier=2):
    """
    Computes true global training step for x-axis.
    If 'epoch' and 'batch' columns exist, derives exact global batch.
    Otherwise uses row index multiplied by the log step multiplier.
    """
    total_rows = len(data)
    if "epoch" in headers and "batch" in headers:
        epoch_idx = headers.index("epoch")
        batch_idx = headers.index("batch")
        try:
            epochs = [int(r[epoch_idx]) for r in data]
            batches = [int(r[batch_idx]) for r in data]
            max_batch_in_epoch = max(batches) if batches else 1
            x_steps = [
                (epochs[i] - 1) * max_batch_in_epoch + batches[i]
                for i in range(total_rows)
            ]
            return x_steps, max_batch_in_epoch
        except (ValueError, IndexError):
            pass

    x_steps = [(i + 1) * multiplier for i in range(total_rows)]
    return x_steps, None


def compute_rolling_mean(values, window):
    smoothed = []
    w = max(1, window)
    for i in range(len(values)):
        start = max(0, i - w + 1)
        subset = values[start : i + 1]
        smoothed.append(sum(subset) / len(subset))
    return smoothed


def setup_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        return plt, ticker
    except ImportError:
        print("Error: The 'matplotlib' library is required to generate plots.")
        print("Please install it using: pip install matplotlib numpy")
        sys.exit(1)


def plot_single_metric(
    plt,
    ticker,
    metric_key,
    x_steps,
    y_vals,
    window,
    multiplier,
    out_dir,
    dpi=200,
):
    meta = METRIC_METADATA.get(
        metric_key,
        {
            'title': f'{metric_key.replace("_", " ").title()} Over Training',
            'ylabel': metric_key,
            'raw_color': '#90caf9',
            'smooth_color': '#d32f2f',
            'desc': '',
        },
    )

    is_percent = meta.get('percent_scale', False)
    display_y = [v * 100.0 if is_percent else v for v in y_vals]
    smoothed_y = compute_rolling_mean(display_y, window)

    min_val = min(display_y) if display_y else 0.0
    max_val = max(display_y) if display_y else 0.0
    initial_val = display_y[0] if display_y else 0.0
    final_val = display_y[-1] if display_y else 0.0
    final_smooth = smoothed_y[-1] if smoothed_y else 0.0

    fig, ax = plt.subplots(figsize=(11, 5.8))

    # Raw curve
    ax.plot(
        x_steps,
        display_y,
        alpha=0.25,
        color=meta['raw_color'],
        linewidth=1.0,
        label=f'Raw (logged every {multiplier} steps)',
    )

    # Smoothed curve
    smooth_batch_span = window * multiplier
    ax.plot(
        x_steps,
        smoothed_y,
        color=meta['smooth_color'],
        linewidth=2.2,
        label=f'Smoothed (Rolling Mean, window = {smooth_batch_span:,} batches)',
    )

    # Title & Subtitle with Statistics (Placing stats in the top subtitle prevents any occlusion of plot curves!)
    ax.set_title(meta['title'], fontsize=13.5, fontweight='bold', pad=22)
    
    # Subtitle with Summary Stats placed cleanly above plot canvas
    unit_suffix = "%" if is_percent else ""
    stats_subtitle = (
        f"Initial: {initial_val:.4f}{unit_suffix}  |  "
        f"Min: {min_val:.4f}{unit_suffix}  |  "
        f"Max: {max_val:.4f}{unit_suffix}  |  "
        f"Final (Smoothed): {final_smooth:.4f}{unit_suffix}"
    )
    ax.text(
        0.5,
        1.02,
        stats_subtitle,
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=9.5,
        color='#424242',
        fontweight='medium',
    )

    ax.set_xlabel('Training Batches (Global Steps)', fontsize=11, fontweight='semibold')
    ax.set_ylabel(meta['ylabel'], fontsize=11, fontweight='semibold')

    # Formatting X and Y axes
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    if is_percent:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100.0, decimals=1))

    # Grid and layout
    ax.grid(True, linestyle='--', alpha=0.45, color='#9e9e9e')
    ax.set_axisbelow(True)

    # Clean, non-intrusive legend
    ax.legend(loc='upper right', framealpha=0.8, edgecolor='#cccccc', fontsize=9.5)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{metric_key}.png')
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ Saved plot: {out_path}")


def plot_overview_dashboard(
    plt,
    ticker,
    metrics_to_plot,
    metric_indices,
    data,
    x_steps,
    window,
    multiplier,
    out_dir,
    dpi=200,
):
    n_metrics = len(metrics_to_plot)
    if n_metrics == 0:
        return

    cols = 2
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, metric_key in enumerate(metrics_to_plot):
        ax = axes[idx]
        col_idx = metric_indices[metric_key]
        raw_vals = []
        for r in data:
            try:
                raw_vals.append(float(r[col_idx]))
            except ValueError:
                raw_vals.append(0.0)

        meta = METRIC_METADATA.get(
            metric_key,
            {
                'title': metric_key.replace('_', ' ').title(),
                'ylabel': metric_key,
                'raw_color': '#90caf9',
                'smooth_color': '#d32f2f',
            },
        )
        is_percent = meta.get('percent_scale', False)
        display_y = [v * 100.0 if is_percent else v for v in raw_vals]
        smoothed_y = compute_rolling_mean(display_y, window)

        ax.plot(x_steps, display_y, alpha=0.22, color=meta['raw_color'], linewidth=0.8)
        ax.plot(x_steps, smoothed_y, color=meta['smooth_color'], linewidth=1.8, label='Smoothed')

        final_val = smoothed_y[-1] if smoothed_y else 0.0
        unit_suffix = "%" if is_percent else ""
        title_with_stat = f"{meta['title']}  [Final: {final_val:.4f}{unit_suffix}]"

        ax.set_title(title_with_stat, fontsize=10.5, fontweight='bold', pad=8)
        ax.set_xlabel('Training Batches', fontsize=9)
        ax.set_ylabel(meta['ylabel'], fontsize=9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
        if is_percent:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100.0, decimals=1))
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    smooth_batch_span = window * multiplier
    fig.suptitle(
        f"ANNP Training Dynamics Overview Dashboard (Rolling Window = {smooth_batch_span:,} Batches)",
        fontsize=15,
        fontweight='bold',
        y=0.995,
    )
    plt.tight_layout()
    dashboard_path = os.path.join(out_dir, "overview_dashboard.png")
    plt.savefig(dashboard_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ★ Saved Overview Dashboard: {dashboard_path}")


def main():
    args = parse_args()
    headers, data = load_csv_data(args.data_dir)
    if headers is None or not data:
        return

    plt, ticker = setup_matplotlib()

    # Determine smoothing window:
    # Default is responsive (~40-50 points, ~80-100 batches) instead of overly sluggish 160 points.
    total_points = len(data)
    if args.window is not None:
        window = max(1, args.window)
    else:
        # Responsive adaptive window: ~total_points / 350, bounded between 5 and 50 points
        window = max(5, min(50, total_points // 350))

    # Compute horizontal global batches
    x_steps, max_batches = calculate_x_axis(data, headers, multiplier=args.multiplier)

    print(f"\nConfiguration:")
    print(f"  • Logged points: {total_points:,}")
    print(f"  • Multiplier per log: {args.multiplier}x (Total range: {x_steps[0]:,} -> {x_steps[-1]:,} batches)")
    print(f"  • Smoothing window: {window} logged points ({window * args.multiplier:,} training batches)")
    print(f"  • Export Directory: {args.out_dir}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    ordered_metrics = [
        'step_loss',
        'ema_loss',
        'early_halting_rate',
        'avg_hops',
        'gini',
        'avg_energy',
        'memory_density',
        'active_subnodes',
        'credit_volatility',
        'temporal_affinity',
    ]

    valid_metrics = [m for m in ordered_metrics if m in headers]
    # Include any extra numerical headers not in ordered_metrics
    exclude = {'timestamp', 'epoch', 'batch'}
    extra_metrics = [h for h in headers if h not in ordered_metrics and h not in exclude]
    valid_metrics.extend(extra_metrics)

    metric_indices = {m: headers.index(m) for m in valid_metrics}

    print("Generating individual metric plots...")
    for metric_key in valid_metrics:
        col_idx = metric_indices[metric_key]
        y_vals = []
        for r in data:
            try:
                y_vals.append(float(r[col_idx]))
            except ValueError:
                y_vals.append(0.0)

        plot_single_metric(
            plt,
            ticker,
            metric_key,
            x_steps,
            y_vals,
            window,
            args.multiplier,
            args.out_dir,
            dpi=args.dpi,
        )

    if not args.no_dashboard:
        plot_overview_dashboard(
            plt,
            ticker,
            valid_metrics,
            metric_indices,
            data,
            x_steps,
            window,
            args.multiplier,
            args.out_dir,
            dpi=args.dpi,
        )

    print(f"\nAll plots generated successfully in '{args.out_dir}'.")


if __name__ == "__main__":
    main()
