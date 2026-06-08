#!/usr/bin/env python3
"""
Neuron Profiling Visualization

Creates useful visualizations from raw Neuron Core profiling data,
focusing on what the data actually captures well:
- Engine utilization over time
- DMA transfer patterns
- Layer-level timing breakdown
- Memory bandwidth utilization
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


COLORS = {
    'tensor_engine': '#e74c3c',   # Red - matrix ops
    'vector_engine': '#3498db',   # Blue - elementwise
    'scalar_engine': '#2ecc71',   # Green - control
    'gpsimd_engine': '#9b59b6',   # Purple - GPSIMD
    'sync_engine': '#f39c12',     # Orange - sync
    'dma': '#1abc9c',             # Teal - data movement
    'idle': '#ecf0f1',            # Light gray
}


def load_profile(filepath: str) -> dict:
    """Load raw profiling JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_bytes(b: float) -> str:
    if b >= 1e9: return f"{b/1e9:.2f} GB"
    if b >= 1e6: return f"{b/1e6:.2f} MB"
    if b >= 1e3: return f"{b/1e3:.2f} KB"
    return f"{b:.0f} B"


def format_time(ns: float) -> str:
    if ns >= 1e9: return f"{ns/1e9:.2f} s"
    if ns >= 1e6: return f"{ns/1e6:.2f} ms"
    if ns >= 1e3: return f"{ns/1e3:.2f} us"
    return f"{ns:.0f} ns"


def plot_engine_utilization(ax, data: dict) -> None:
    """Bar chart of engine utilization."""
    summary = data['summary'][0]
    
    engines = [
        ('Tensor\n(MatMul)', summary.get('tensor_engine_active_time_percent', 0)),
        ('Vector\n(Elemwise)', summary.get('vector_engine_active_time_percent', 0)),
        ('Scalar', summary.get('scalar_engine_active_time_percent', 0)),
        ('GPSIMD', summary.get('gpsimd_engine_active_time_percent', 0)),
        ('Sync', summary.get('sync_engine_active_time_percent', 0)),
        ('DMA', summary.get('dma_active_time_percent', 0)),
    ]
    
    names, values = zip(*engines)
    colors = [COLORS['tensor_engine'], COLORS['vector_engine'], 
              COLORS['scalar_engine'], COLORS['gpsimd_engine'],
              COLORS['sync_engine'], COLORS['dma']]
    
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=1)
    
    for bar, val in zip(bars, values):
        if val > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Active Time (%)')
    ax.set_title('Engine Utilization')
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 10)
    ax.grid(True, alpha=0.3, axis='y')


def plot_active_time_timeline(ax, data: dict) -> None:
    """Timeline of engine activity."""
    active_time = data.get('active_time', [])
    
    if not active_time:
        ax.text(0.5, 0.5, 'No active_time data', ha='center', va='center')
        return
    
    # Group by engine
    engine_events = defaultdict(list)
    for event in active_time:
        engine = event.get('engine', 'unknown')
        start = event.get('start_ts', 0) / 1e6  # to ms
        end = event.get('end_ts', 0) / 1e6
        engine_events[engine].append((start, end - start))
    
    # Plot as horizontal bars
    engine_names = ['TensorE', 'VectorE', 'ScalarE', 'GpSimd', 'SyncE']
    engine_map = {
        'TensorE': 'tensor_engine', 'VectorE': 'vector_engine',
        'ScalarE': 'scalar_engine', 'GpSimd': 'gpsimd_engine', 
        'SyncE': 'sync_engine'
    }
    
    for i, eng_name in enumerate(engine_names):
        events = engine_events.get(eng_name, [])
        if events:
            color = COLORS.get(engine_map.get(eng_name, 'idle'), '#999999')
            ax.broken_barh(events, (i - 0.4, 0.8), facecolors=color, alpha=0.7)
    
    ax.set_yticks(range(len(engine_names)))
    ax.set_yticklabels(engine_names, fontsize=9)
    ax.set_xlabel('Time (ms)')
    ax.set_title('Engine Activity Timeline')
    ax.grid(True, alpha=0.3, axis='x')


def plot_dma_timeline(ax, data: dict) -> None:
    """DMA transfer activity over time."""
    dma_events = data.get('dma', [])
    
    if not dma_events:
        ax.text(0.5, 0.5, 'No DMA data', ha='center', va='center')
        return
    
    # Bin by time
    metadata = data.get('metadata', [{}])[0]
    total_time = metadata.get('last_hw_timestamp', 1e6)
    
    num_bins = 100
    bin_size = total_time / num_bins
    
    read_bytes = np.zeros(num_bins)
    write_bytes = np.zeros(num_bins)
    
    for dma in dma_events:
        ts = dma.get('timestamp', 0)
        size = dma.get('transfer_size', 0)
        bin_idx = min(int(ts / bin_size), num_bins - 1)
        
        # Determine direction from source/dest
        source = dma.get('source', [])
        dest = dma.get('dest', [])
        
        # If source contains 'REMOTE' or dest is 'SB', it's a read into SBUF
        source_str = str(source)
        dest_str = str(dest)
        
        if 'REMOTE' in source_str or 'HBM' in source_str:
            read_bytes[bin_idx] += size
        else:
            write_bytes[bin_idx] += size
    
    # Convert to MB/bin and calculate bandwidth
    bin_time_ms = bin_size / 1e6
    read_bw = read_bytes / 1e6 / bin_time_ms * 1000  # MB/s
    write_bw = write_bytes / 1e6 / bin_time_ms * 1000
    
    times_ms = np.arange(num_bins) * bin_size / 1e6
    
    ax.fill_between(times_ms, 0, read_bw, alpha=0.7, color=COLORS['dma'], label='Read BW')
    ax.fill_between(times_ms, 0, -write_bw, alpha=0.7, color='#e74c3c', label='Write BW')
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Bandwidth (MB/s)')
    ax.set_title('DMA Bandwidth Over Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_layer_breakdown(ax, data: dict) -> None:
    """Pie chart of time spent per operation type."""
    layers = data.get('layer_summary', [])
    
    if not layers:
        ax.text(0.5, 0.5, 'No layer data', ha='center', va='center')
        return
    
    # Group by operation type
    op_times = defaultdict(int)
    for layer in layers:
        name = layer.get('name', '')
        duration = layer.get('duration', 0)
        
        # Extract operation type from name
        if '_dot' in name or '_matmul' in name:
            op_type = 'MatMul'
        elif '_multiply' in name or '_add' in name:
            op_type = 'Elementwise'
        elif '_gather' in name or '_scatter' in name:
            op_type = 'Gather/Scatter'
        elif '_custom-call' in name:
            op_type = 'Custom Ops'
        elif 'sg00' in name or 'sg01' in name:
            continue  # Skip top-level subgraphs
        elif 'fn' in name:
            continue  # Skip function wrappers
        elif 'Unknown' in name:
            op_type = 'Other'
        else:
            op_type = 'Other'
        
        op_times[op_type] += duration
    
    if not op_times:
        ax.text(0.5, 0.5, 'No operation breakdown available', ha='center', va='center')
        return
    
    # Sort and plot
    sorted_ops = sorted(op_times.items(), key=lambda x: -x[1])
    labels, values = zip(*sorted_ops)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90)
    for autotext in autotexts:
        autotext.set_fontsize(8)
    
    ax.set_title('Time by Operation Type')


def plot_memory_bandwidth(ax, data: dict) -> None:
    """Memory bandwidth utilization."""
    summary = data['summary'][0]
    metadata = data.get('metadata', [{}])[0]
    
    # Get actual bandwidth used
    total_time_s = metadata.get('last_hw_timestamp', 1) / 1e9
    hbm_read = summary.get('hbm_read_bytes', 0)
    hbm_write = summary.get('hbm_write_bytes', 0)
    
    actual_read_bw = hbm_read / total_time_s / 1e9  # GB/s
    actual_write_bw = hbm_write / total_time_s / 1e9
    
    # Peak HBM bandwidth (from metadata)
    peak_bw = metadata.get('hbm_ddr_bandwidth', 716e9) / 1e9  # GB/s
    
    categories = ['HBM Read', 'HBM Write', 'Peak BW']
    values = [actual_read_bw, actual_write_bw, peak_bw]
    colors = [COLORS['dma'], '#e74c3c', '#95a5a6']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='white')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'{val:.0f} GB/s', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Memory Bandwidth Utilization')
    ax.grid(True, alpha=0.3, axis='y')


def plot_summary_text(ax, data: dict) -> None:
    """Summary statistics as text."""
    ax.axis('off')
    
    summary = data['summary'][0]
    metadata = data.get('metadata', [{}])[0]
    
    total_time = metadata.get('last_hw_timestamp', 0)
    
    text = f"""
NEURON PROFILING SUMMARY
{'═' * 50}

TIMING
{'─' * 50}
Total Execution Time:   {format_time(total_time)}
Tensor Engine Active:   {summary.get('tensor_engine_active_time_percent', 0):.2f}%
DMA Active:             {summary.get('dma_active_time_percent', 0):.2f}%

COMPUTE EFFICIENCY  
{'─' * 50}
MFU (estimated):        {summary.get('mfu_estimated_percent', 0):.2f}%
Model FLOPs:            {summary.get('model_flops', 0)/1e9:.2f} GFLOPS
MatMul Instructions:    {summary.get('matmul_instruction_count', 0):,}

DATA MOVEMENT
{'─' * 50}
HBM Reads:              {format_bytes(summary.get('hbm_read_bytes', 0))}
HBM Writes:             {format_bytes(summary.get('hbm_write_bytes', 0))}
Total DMA Transfers:    {summary.get('dma_transfer_count', 0):,}
Spill/Reload:           {format_bytes(summary.get('spill_reload_bytes', 0))}

MEMORY
{'─' * 50}
Input Size:             {format_bytes(summary.get('input_queue_bytes', 0))}
Weight Size:            {format_bytes(summary.get('weight_size_bytes', 0))}
Output Size:            {format_bytes(summary.get('output_queue_bytes', 0))}
"""
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            verticalalignment='top')


def plot_dma_queue_activity(ax, data: dict) -> None:
    """DMA activity by queue."""
    dma_events = data.get('dma', [])
    
    if not dma_events:
        ax.text(0.5, 0.5, 'No DMA data', ha='center', va='center')
        return
    
    # Count by queue
    queue_bytes = defaultdict(int)
    queue_count = defaultdict(int)
    
    for dma in dma_events:
        queue = dma.get('dma_queue', 'unknown')
        queue_type = dma.get('queue_type', 'unknown')
        size = dma.get('transfer_size', 0)
        
        key = f"Q{queue} ({queue_type})"
        queue_bytes[key] += size
        queue_count[key] += 1
    
    # Sort by bytes transferred
    sorted_queues = sorted(queue_bytes.items(), key=lambda x: -x[1])[:10]
    
    if not sorted_queues:
        ax.text(0.5, 0.5, 'No queue data', ha='center', va='center')
        return
    
    names, values = zip(*sorted_queues)
    values_mb = [v / 1e6 for v in values]
    
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values_mb, color=COLORS['dma'], alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Data Transferred (MB)')
    ax.set_title('DMA Queue Activity')
    ax.grid(True, alpha=0.3, axis='x')


def create_visualization(profile_path: str, output_path: str = None) -> None:
    """Create the visualization."""
    print(f"Loading profile from {profile_path}...", file=sys.stderr)
    data = load_profile(profile_path)
    
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Neuron Core Execution Profile', fontsize=16, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3,
                          left=0.06, right=0.94, top=0.92, bottom=0.06)
    
    # Row 1: Summary, Engine Utilization, Memory BW
    ax_summary = fig.add_subplot(gs[0, 0])
    ax_engines = fig.add_subplot(gs[0, 1])
    ax_membw = fig.add_subplot(gs[0, 2])
    
    # Row 2: Engine Timeline (wide), Layer Breakdown
    ax_timeline = fig.add_subplot(gs[1, :2])
    ax_layers = fig.add_subplot(gs[1, 2])
    
    # Row 3: DMA Timeline (wide), DMA Queues
    ax_dma = fig.add_subplot(gs[2, :2])
    ax_queues = fig.add_subplot(gs[2, 2])
    
    print("Creating plots...", file=sys.stderr)
    plot_summary_text(ax_summary, data)
    plot_engine_utilization(ax_engines, data)
    plot_memory_bandwidth(ax_membw, data)
    plot_active_time_timeline(ax_timeline, data)
    plot_layer_breakdown(ax_layers, data)
    plot_dma_timeline(ax_dma, data)
    plot_dma_queue_activity(ax_queues, data)
    
    if output_path:
        print(f"Saving to {output_path}...", file=sys.stderr)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved!", file=sys.stderr)
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Neuron profiling data")
    parser.add_argument("profile_path", help="Path to raw profiling JSON file")
    parser.add_argument("-o", "--output", help="Output image path")
    
    args = parser.parse_args()
    
    try:
        create_visualization(args.profile_path, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
