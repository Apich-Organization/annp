import csv
import glob
import os

def main():
    csv_files = glob.glob('./stage_1/data/*.csv')
    if not csv_files:
        print("No CSV files found in ./stage_1/data/")
        return

    print(f"Found {len(csv_files)} CSV file(s). Processing...")

    # Try to import matplotlib. If not present, we will guide the user.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: The 'matplotlib' library is required to generate plots.")
        print("Please install it using: pip install matplotlib pandas")
        return

    # We will read everything into lists for easy plotting
    # Assuming columns are: timestamp,epoch,batch,step_loss,ema_loss,early_halting_rate,gini,avg_hops,avg_energy,memory_density,active_subnodes,credit_volatility,temporal_affinity

    data = []
    headers = None

    for f in csv_files:
        with open(f, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            file_headers = next(reader)
            if headers is None:
                headers = [h.strip() for h in file_headers]
            for row in reader:
                if not row or len(row) != len(headers):
                    continue
                data.append(row)

    # Sort by timestamp (assuming timestamp is first column)
    timestamp_idx = headers.index('timestamp') if 'timestamp' in headers else 0
    data.sort(key=lambda x: x[timestamp_idx])

    metrics = [
        'step_loss', 'ema_loss', 'early_halting_rate', 'gini', 'avg_hops',
        'avg_energy', 'memory_density', 'active_subnodes', 'credit_volatility',
        'temporal_affinity'
    ]

    # Check which metrics are actually in the CSV
    valid_metrics = [m for m in metrics if m in headers]
    metric_indices = {m: headers.index(m) for m in valid_metrics}

    out_dir = './stage_1/plots'
    os.makedirs(out_dir, exist_ok=True)

    cumulative_batch = list(range(1, len(data) + 1))

    for metric in valid_metrics:
        idx = metric_indices[metric]

        # Parse values as floats
        y_vals = []
        for row in data:
            try:
                y_vals.append(float(row[idx]))
            except ValueError:
                y_vals.append(0.0)

        plt.figure(figsize=(12, 6))

        # Raw data
        plt.plot(cumulative_batch, y_vals, alpha=0.3, color='steelblue', label=f'Raw {metric}')

        # Smoothed data (rolling mean)
        window = max(1, len(y_vals) // 100)
        smoothed = []
        for i in range(len(y_vals)):
            start = max(0, i - window + 1)
            subset = y_vals[start:i+1]
            smoothed.append(sum(subset) / len(subset))

        plt.plot(cumulative_batch, smoothed, color='red', linewidth=2, label=f'Smoothed (window={window})')

        plt.title(f'{metric.replace("_", " ").title()} Over Training')
        plt.xlabel('Cumulative Logged Batches')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f'{metric}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot: {out_path}")

    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
