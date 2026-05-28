#!/usr/bin/env python3
"""
Tensor Lifetime Analysis Visualization

Creates matplotlib visualizations from tensor_lifetime_analysis.json including:
- Hot tensors HBM round-trips bar chart
- Arithmetic intensity per operation
- Roofline model visualization
- Engine utilization
- Memory breakdown
- Tensor classification distribution
"""

import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import FuncFormatter

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette
COLORS = {
    'hot': '#e74c3c',
    'warm': '#f39c12', 
    'cold': '#3498db',
    'compute': '#2ecc71',
    'memory': '#9b59b6',
    'primary': '#1abc9c',
    'secondary': '#34495e',
    'accent': '#e67e22',
    'light_gray': '#ecf0f1',
    'dark_gray': '#7f8c8d',
}


def load_analysis_data(json_path: str) -> dict:
    """Load the tensor lifetime analysis JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def format_bytes(x, pos):
    """Format bytes for axis labels."""
    if x >= 1e9:
        return f'{x/1e9:.1f} GB'
    elif x >= 1e6:
        return f'{x/1e6:.1f} MB'
    elif x >= 1e3:
        return f'{x/1e3:.1f} KB'
    return f'{x:.0f} B'


def format_flops(x, pos):
    """Format FLOPS for axis labels."""
    if x >= 1e12:
        return f'{x/1e12:.1f} T'
    elif x >= 1e9:
        return f'{x/1e9:.1f} G'
    elif x >= 1e6:
        return f'{x/1e6:.1f} M'
    return f'{x:.0f}'


def plot_hot_tensors_roundtrips(data: dict, ax: plt.Axes):
    """Bar chart of hot tensors and their HBM round-trips."""
    tensors = data['primary_tensors']['tensors']
    
    # Get hot tensors sorted by round trips
    hot_tensors = [t for t in tensors if t['classification'] == 'hot']
    hot_tensors.sort(key=lambda x: x['hbm_metrics']['round_trip_count'], reverse=True)
    
    if not hot_tensors:
        ax.text(0.5, 0.5, 'No hot tensors identified', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Hot Tensors: HBM Round-Trips', fontweight='bold')
        return
    
    # Take top 10
    hot_tensors = hot_tensors[:10]
    
    # Use semantic names if available, fall back to id
    names = [t.get('semantic_name') or t['id'] for t in hot_tensors]
    round_trips = [t['hbm_metrics']['round_trip_count'] for t in hot_tensors]
    
    # Color by category
    category_colors = {
        'kv_cache': COLORS['hot'],
        'runtime_input': COLORS['accent'],
        'attention_weight': COLORS['primary'],
        'mlp_weight': COLORS['compute'],
        'embedding': COLORS['memory'],
    }
    colors = [category_colors.get(t.get('category', ''), COLORS['hot']) for t in hot_tensors]
    
    bars = ax.barh(names, round_trips, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, round_trips):
        ax.text(bar.get_width() + max(round_trips) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', fontsize=9)
    
    ax.set_xlabel('HBM Round-Trips', fontweight='bold')
    ax.set_title('Hot Tensors: HBM Round-Trips', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(round_trips) * 1.15)
    
    # Add annotation
    ax.annotate('Higher = More cache thrashing', xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', fontsize=8, color=COLORS['dark_gray'], style='italic')


def plot_hot_tensors_traffic(data: dict, ax: plt.Axes):
    """Bar chart of hot tensors and their total HBM traffic."""
    tensors = data['primary_tensors']['tensors']
    
    # Get hot tensors sorted by total traffic
    hot_tensors = [t for t in tensors if t['classification'] == 'hot']
    hot_tensors.sort(key=lambda x: x['hbm_metrics']['total_read_bytes'] + x['hbm_metrics']['total_write_bytes'], 
                     reverse=True)
    
    if not hot_tensors:
        ax.text(0.5, 0.5, 'No hot tensors identified', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Hot Tensors: HBM Traffic', fontweight='bold')
        return
    
    hot_tensors = hot_tensors[:10]
    
    # Use semantic names if available
    names = [t.get('semantic_name') or t['id'] for t in hot_tensors]
    reads = [t['hbm_metrics']['total_read_bytes'] / 1e6 for t in hot_tensors]
    writes = [t['hbm_metrics']['total_write_bytes'] / 1e6 for t in hot_tensors]
    
    y_pos = np.arange(len(names))
    
    bars1 = ax.barh(y_pos, reads, color=COLORS['primary'], label='HBM Reads', 
                    edgecolor='white', linewidth=0.5)
    bars2 = ax.barh(y_pos, writes, left=reads, color=COLORS['accent'], label='HBM Writes',
                    edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('HBM Traffic (MB)', fontweight='bold')
    ax.set_title('Hot Tensors: HBM Traffic Breakdown', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    ax.invert_yaxis()


def plot_arithmetic_intensity(data: dict, ax: plt.Axes):
    """Bar chart of arithmetic intensity per operation type."""
    cma = data.get('compute_memory_analysis', {})
    ops = cma.get('per_operation_analysis', [])
    ridge_point = cma.get('arithmetic_intensity', {}).get('ridge_point_flops_per_byte', 100)
    
    if not ops:
        ax.text(0.5, 0.5, 'No operation data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Arithmetic Intensity by Operation', fontweight='bold')
        return
    
    # Sort by arithmetic intensity
    ops = sorted(ops, key=lambda x: x['arithmetic_intensity'], reverse=True)
    
    names = [op['operation'] for op in ops]
    intensities = [op['arithmetic_intensity'] for op in ops]
    classifications = [op['classification'] for op in ops]
    
    # Color based on classification
    colors = [COLORS['compute'] if c == 'compute_bound' else COLORS['memory'] 
              for c in classifications]
    
    bars = ax.barh(names, intensities, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add ridge point line
    ax.axvline(x=ridge_point, color=COLORS['hot'], linestyle='--', linewidth=2, 
               label=f'Ridge Point ({ridge_point:.1f})')
    
    # Add value labels
    max_val = max(intensities) if intensities else ridge_point
    for bar, val in zip(bars, intensities):
        ax.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)
    
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontweight='bold')
    ax.set_title('Arithmetic Intensity by Operation', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max_val, ridge_point) * 1.2)
    
    # Add legend for colors
    compute_patch = mpatches.Patch(color=COLORS['compute'], label='Compute Bound')
    memory_patch = mpatches.Patch(color=COLORS['memory'], label='Memory Bound')
    ax.legend(handles=[compute_patch, memory_patch], loc='lower right', fontsize=8)


def plot_roofline(data: dict, ax: plt.Axes):
    """Roofline model visualization."""
    cma = data.get('compute_memory_analysis', {})
    
    peak_compute = cma.get('flops_metrics', {}).get('peak_theoretical_tflops', 100)
    peak_bw = cma.get('bandwidth_metrics', {}).get('peak_theoretical_gbps', 700)
    achieved_compute = cma.get('flops_metrics', {}).get('achieved_throughput_tflops', 0)
    achieved_intensity = cma.get('arithmetic_intensity', {}).get('overall_flops_per_byte', 0)
    ridge_point = cma.get('arithmetic_intensity', {}).get('ridge_point_flops_per_byte', 100)
    
    # X-axis: arithmetic intensity (log scale)
    x_min, x_max = 0.1, 1000
    x = np.logspace(np.log10(x_min), np.log10(x_max), 500)
    
    # Memory-bound region: performance = bandwidth * intensity
    # Convert: GB/s * FLOP/Byte = GFLOP/s, then to TFLOP/s
    memory_bound = (peak_bw * x) / 1000  # TFLOP/s
    
    # Compute-bound region: performance = peak compute
    compute_bound = np.full_like(x, peak_compute)
    
    # Actual roofline is the minimum
    roofline = np.minimum(memory_bound, compute_bound)
    
    # Plot roofline
    ax.loglog(x, roofline, color=COLORS['secondary'], linewidth=3, label='Roofline')
    
    # Fill regions
    ax.fill_between(x, 0.01, roofline, where=(x < ridge_point), 
                    alpha=0.2, color=COLORS['memory'], label='Memory Bound Region')
    ax.fill_between(x, 0.01, roofline, where=(x >= ridge_point),
                    alpha=0.2, color=COLORS['compute'], label='Compute Bound Region')
    
    # Plot achieved performance
    if achieved_intensity > 0 and achieved_compute > 0:
        ax.scatter([achieved_intensity], [achieved_compute], 
                   s=200, c=COLORS['hot'], marker='*', zorder=5, 
                   edgecolors='white', linewidths=2,
                   label=f'Your Workload ({achieved_intensity:.1f} FLOP/B, {achieved_compute:.1f} TFLOP/s)')
    
    # Mark ridge point
    ax.axvline(x=ridge_point, color=COLORS['dark_gray'], linestyle=':', alpha=0.7)
    ax.annotate(f'Ridge\n({ridge_point:.0f})', xy=(ridge_point, peak_compute * 0.5),
                fontsize=8, ha='center', color=COLORS['dark_gray'])
    
    # Labels
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontweight='bold')
    ax.set_ylabel('Performance (TFLOP/s)', fontweight='bold')
    ax.set_title('Roofline Model Analysis', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.1, peak_compute * 2)
    ax.grid(True, alpha=0.3, which='both')


def plot_engine_utilization(data: dict, ax: plt.Axes):
    """Bar chart of engine utilization."""
    cma = data.get('compute_memory_analysis', {})
    util = cma.get('engine_utilization', {})
    
    engines = ['Tensor\nEngine', 'Vector\nEngine', 'Scalar\nEngine', 
               'GpSimd\nEngine', 'DMA\nEngine', 'Sync\nEngine']
    values = [
        util.get('tensor_engine_percent', 0),
        util.get('vector_engine_percent', 0),
        util.get('scalar_engine_percent', 0),
        util.get('gpsimd_engine_percent', 0),
        util.get('dma_engine_percent', 0),
        util.get('sync_engine_percent', 0),
    ]
    
    # Color: highlight DMA if it's the bottleneck
    colors = [COLORS['primary']] * len(engines)
    max_idx = np.argmax(values)
    colors[max_idx] = COLORS['hot']
    
    bars = ax.bar(engines, values, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=9)
    
    ax.set_ylabel('Utilization (%)', fontweight='bold')
    ax.set_title('Engine Utilization', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 100)
    
    # Add 100% reference line
    ax.axhline(y=100, color=COLORS['dark_gray'], linestyle='--', alpha=0.5)
    
    # Annotate bottleneck
    bottleneck_engine = engines[max_idx].replace('\n', ' ')
    ax.annotate(f'Bottleneck: {bottleneck_engine}', xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, color=COLORS['hot'], fontweight='bold')


def plot_memory_breakdown(data: dict, ax: plt.Axes):
    """Pie chart of memory traffic breakdown."""
    cma = data.get('compute_memory_analysis', {})
    bw = cma.get('bandwidth_metrics', {}).get('breakdown', {})
    
    hbm_read = bw.get('hbm_read_bytes', 0)
    hbm_write = bw.get('hbm_write_bytes', 0)
    spill = bw.get('spill_bytes', 0)
    
    if hbm_read + hbm_write + spill == 0:
        ax.text(0.5, 0.5, 'No memory data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('HBM Traffic Breakdown', fontweight='bold')
        return
    
    sizes = []
    labels = []
    colors = []
    
    if hbm_read > 0:
        sizes.append(hbm_read)
        labels.append(f'HBM Reads\n({hbm_read/1e9:.2f} GB)')
        colors.append(COLORS['primary'])
    if hbm_write > 0:
        sizes.append(hbm_write)
        labels.append(f'HBM Writes\n({hbm_write/1e9:.2f} GB)')
        colors.append(COLORS['accent'])
    if spill > 0:
        sizes.append(spill)
        labels.append(f'Spill Traffic\n({spill/1e9:.2f} GB)')
        colors.append(COLORS['hot'])
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       explode=[0.02] * len(sizes),
                                       textprops={'fontsize': 9})
    
    ax.set_title('HBM Traffic Breakdown', fontweight='bold', fontsize=12)


def plot_tensor_classification(data: dict, ax: plt.Axes):
    """Pie chart of tensor classification distribution."""
    summary = data.get('primary_tensors', {}).get('summary', {})
    by_class = summary.get('by_classification', {})
    
    hot = by_class.get('hot', 0)
    warm = by_class.get('warm', 0)
    cold = by_class.get('cold', 0)
    
    total = hot + warm + cold
    if total == 0:
        ax.text(0.5, 0.5, 'No tensor data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Tensor Classification', fontweight='bold')
        return
    
    sizes = []
    labels = []
    colors = []
    
    if hot > 0:
        sizes.append(hot)
        labels.append(f'Hot ({hot})')
        colors.append(COLORS['hot'])
    if warm > 0:
        sizes.append(warm)
        labels.append(f'Warm ({warm})')
        colors.append(COLORS['warm'])
    if cold > 0:
        sizes.append(cold)
        labels.append(f'Cold ({cold})')
        colors.append(COLORS['cold'])
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       explode=[0.05 if c == COLORS['hot'] else 0 for c in colors],
                                       textprops={'fontsize': 9})
    
    ax.set_title('Tensor Classification Distribution', fontweight='bold', fontsize=12)


def plot_category_breakdown(data: dict, ax: plt.Axes):
    """Bar chart of tensor categories with HBM traffic."""
    category_summary = data.get('category_summary', {})
    
    if not category_summary:
        ax.text(0.5, 0.5, 'No category data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Tensor Categories: HBM Traffic', fontweight='bold')
        return
    
    # Define colors for categories
    category_colors = {
        'kv_cache': COLORS['hot'],
        'runtime_input': COLORS['accent'],
        'attention_weight': COLORS['primary'],
        'mlp_weight': COLORS['compute'],
        'norm_weight': COLORS['memory'],
        'embedding': COLORS['secondary'],
        'output_weight': COLORS['warm'],
        'unknown': COLORS['dark_gray'],
    }
    
    # Sort categories by HBM traffic
    categories = [(cat, stats) for cat, stats in category_summary.items()]
    categories.sort(key=lambda x: x[1].get('total_hbm_traffic_mb', 0), reverse=True)
    
    names = [cat.replace('_', ' ').title() for cat, _ in categories]
    traffic = [stats.get('total_hbm_traffic_mb', 0) for _, stats in categories]
    counts = [stats.get('count', 0) for _, stats in categories]
    colors = [category_colors.get(cat, COLORS['dark_gray']) for cat, _ in categories]
    
    y_pos = np.arange(len(names))
    
    bars = ax.barh(y_pos, traffic, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + max(traffic) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{count} tensors', va='center', fontsize=8, color=COLORS['dark_gray'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('HBM Traffic (MB)', fontweight='bold')
    ax.set_title('Tensor Categories: HBM Traffic', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    
    # Highlight KV cache if it's the top
    if categories and categories[0][0] == 'kv_cache':
        ax.annotate('KV Cache dominates HBM traffic', xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', fontsize=8, color=COLORS['hot'], style='italic')


def plot_tensor_sizes_by_type(data: dict, ax: plt.Axes):
    """Bar chart of tensor sizes by type."""
    tensors = data['primary_tensors']['tensors']
    
    # Group by type
    by_type = {'IN': [], 'OUT': [], 'WEIGHT': []}
    for t in tensors:
        ttype = t['type']
        if ttype in by_type:
            by_type[ttype].append(t['size_bytes'])
    
    types = ['INPUT', 'OUTPUT', 'WEIGHT']
    counts = [len(by_type['IN']), len(by_type['OUT']), len(by_type['WEIGHT'])]
    total_sizes = [sum(by_type['IN'])/1e6, sum(by_type['OUT'])/1e6, sum(by_type['WEIGHT'])/1e6]
    
    x = np.arange(len(types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, counts, width, label='Count', color=COLORS['primary'],
                   edgecolor='white', linewidth=0.5)
    
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, total_sizes, width, label='Total Size (MB)', 
                    color=COLORS['accent'], edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel('Count', fontweight='bold', color=COLORS['primary'])
    ax2.set_ylabel('Total Size (MB)', fontweight='bold', color=COLORS['accent'])
    ax.set_title('Tensors by Type', fontweight='bold', fontsize=12)
    
    # Combined legend
    ax.legend([bars1, bars2], ['Count', 'Total Size (MB)'], loc='upper right', fontsize=8)


def plot_utilization_comparison(data: dict, ax: plt.Axes):
    """Compare compute vs bandwidth utilization."""
    cma = data.get('compute_memory_analysis', {})
    
    compute_util = cma.get('flops_metrics', {}).get('compute_utilization_percent', 0)
    bw_util = cma.get('bandwidth_metrics', {}).get('bandwidth_utilization_percent', 0)
    
    categories = ['Compute\nUtilization', 'Bandwidth\nUtilization']
    values = [compute_util, bw_util]
    
    # Determine which is bottleneck
    colors = [COLORS['compute'], COLORS['memory']]
    if bw_util > compute_util:
        colors[1] = COLORS['hot']  # Bandwidth is bottleneck
    else:
        colors[0] = COLORS['hot']  # Compute is bottleneck
    
    bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Utilization (%)', fontweight='bold')
    ax.set_title('Compute vs Bandwidth Utilization', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(y=100, color=COLORS['dark_gray'], linestyle='--', alpha=0.5)
    
    # Classification annotation
    classification = cma.get('classification', 'unknown').replace('_', ' ').upper()
    ax.annotate(f'Classification: {classification}', xy=(0.5, 0.02), xycoords='axes fraction',
                ha='center', fontsize=10, fontweight='bold', 
                color=COLORS['memory'] if 'MEMORY' in classification else COLORS['compute'])


def create_dashboard(data: dict, output_path: str):
    """Create a comprehensive dashboard with all visualizations."""
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    plot_hot_tensors_roundtrips(data, ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_hot_tensors_traffic(data, ax2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_category_breakdown(data, ax3)
    
    # Row 2
    ax4 = fig.add_subplot(gs[1, 0:2])
    plot_roofline(data, ax4)
    
    ax5 = fig.add_subplot(gs[1, 2])
    plot_engine_utilization(data, ax5)
    
    # Row 3
    ax6 = fig.add_subplot(gs[2, 0])
    plot_memory_breakdown(data, ax6)
    
    ax7 = fig.add_subplot(gs[2, 1])
    plot_arithmetic_intensity(data, ax7)
    
    ax8 = fig.add_subplot(gs[2, 2])
    plot_utilization_comparison(data, ax8)
    
    # Main title
    classification = data.get('compute_memory_analysis', {}).get('classification', 'unknown')
    classification_str = classification.replace('_', ' ').upper()
    
    fig.suptitle(f'Tensor Lifetime Analysis Dashboard\nWorkload: {classification_str}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Dashboard saved to: {output_path}")
    plt.close()


def create_individual_plots(data: dict, output_dir: str):
    """Create individual plot files."""
    os.makedirs(output_dir, exist_ok=True)
    
    plots = [
        ('hot_tensors_roundtrips.png', plot_hot_tensors_roundtrips),
        ('hot_tensors_traffic.png', plot_hot_tensors_traffic),
        ('category_breakdown.png', plot_category_breakdown),
        ('arithmetic_intensity.png', plot_arithmetic_intensity),
        ('roofline.png', plot_roofline),
        ('engine_utilization.png', plot_engine_utilization),
        ('memory_breakdown.png', plot_memory_breakdown),
        ('tensor_classification.png', plot_tensor_classification),
        ('utilization_comparison.png', plot_utilization_comparison),
    ]
    
    for filename, plot_func in plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_func(data, ax)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {filepath}")
        plt.close()


def main():
    # Default paths
    json_path = "/home/ubuntu/profiling_output/tensor_lifetime_analysis.json"
    output_dir = "/home/ubuntu/profiling_output/visualizations"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print("=" * 60)
    print("TENSOR ANALYSIS VISUALIZATION")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {json_path}")
    data = load_analysis_data(json_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dashboard
    print("\nCreating dashboard...")
    dashboard_path = os.path.join(output_dir, "tensor_analysis_dashboard.png")
    create_dashboard(data, dashboard_path)
    
    # Create individual plots
    print("\nCreating individual plots...")
    create_individual_plots(data, output_dir)
    
    print("\nDone!")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
