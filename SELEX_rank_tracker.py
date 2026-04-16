import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import gc

# Target motif sequence
target_sequences = {
    'AU_short#1': 'ATACTTAGCGTTCGTGCGGGAGCAGTCGCAAGCCGCCTAT',
    'B1m15#1': 'CAACCGTTGTGCACCACCCGAGATGCCCAGTGCCCTAGTA'
}

# Save results
results = {seq_name: {} for seq_name in target_sequences.keys()}

# If the file name: NagR_round1.csv, NagR_round2.csv, ..., NagR_round14.csv
file_pattern = 'NagR_round{}_with_RPM.csv'  # or 'nagr_round{}.csv' Adjust base on different path

print("="*80)
print("Start to analysis sequence rank of SELEX...")
print("="*80)

# Traversing 14 files
for round_num in range(1, 15):
    filename = file_pattern.format(round_num)
    
    if not os.path.exists(filename):
        print(f"⚠️  Warning: can't find file {filename}，skip...")
        continue
    
    print(f"\nprocessing Round {round_num}: {filename}")
    
    # Initiate sequence counter in current round
    found_sequences = {seq_name: None for seq_name in target_sequences.keys()}
    
    try:
        # Read 10000 per cycle in chunks 
        chunk_size = 100000
        rank = 0
        
        # Calculate the total number of lines (For progress bar)
        total_rows = sum(1 for _ in open(filename)) - 1  # substract header
        print(f"  toehold number: {total_rows:,}")
        
        # Read CSV file in chunks
        for chunk_idx, chunk in enumerate(pd.read_csv(filename, chunksize=chunk_size)):
            # Descending orders based on RPM
            chunk = chunk.sort_values('RPM', ascending=False).reset_index(drop=True)
            
            # Traversing current chunk
            for idx, row in chunk.iterrows():
                rank += 1
                sequence = row['Sequence']
                
                # Check whether match the target sequence
                for seq_name, target_seq in target_sequences.items():
                    if found_sequences[seq_name] is None and sequence == target_seq:
                        found_sequences[seq_name] = {
                            'rank': rank,
                            'count': row['Count'],
                            'rpkm': row['RPM']
                        }
                        print(f"  ✓ Found {seq_name}: Rank={rank}, Count={row['Count']}, RPKM={row['RPM']:.2f}")
                
                # If found two sequences, exit in advance
                if all(v is not None for v in found_sequences.values()):
                    break
            
            # If found two sequences, stop extract the following chunk
            if all(v is not None for v in found_sequences.values()):
                break
            
            # Show progress
            processed = min((chunk_idx + 1) * chunk_size, total_rows)
            print(f"  Progress: {processed:,} / {total_rows:,} ({processed/total_rows*100:.1f}%)", end='\r')
            
            # Clean ROM memory
            del chunk
            gc.collect()
        
        # Save results
        for seq_name, data in found_sequences.items():
            if data is not None:
                results[seq_name][round_num] = data
            else:
                print(f"  ✗ Not Found {seq_name}")
                results[seq_name][round_num] = {
                    'rank': None,
                    'count': 0,
                    'rpkm': 0
                }
    
    except Exception as e:
        print(f"  ❌ False: {e}")
        continue

print("\n" + "="*80)
print("Data collection complete！")
print("="*80)

# ====================Plot rank line chart====================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Fig.1: Rank comparison
ax1 = axes[0, 0]
for seq_name, color in zip(target_sequences.keys(), ['#FF6B6B', '#4ECDC4']):
    rounds = sorted(results[seq_name].keys())
    ranks = [results[seq_name][r]['rank'] if results[seq_name][r]['rank'] is not None else np.nan 
             for r in rounds]
    
    ax1.plot(rounds, ranks, marker='o', linewidth=2.5, markersize=10, 
            label=seq_name, color=color, alpha=0.8)
    
    # Add annotation
    for r, rank in zip(rounds, ranks):
        if not np.isnan(rank):
            ax1.annotate(f'{int(rank)}', (r, rank), 
                        textcoords="offset points", xytext=(0, 8), 
                        ha='center', fontsize=8, alpha=0.7)

ax1.set_xlabel('SELEX Round', fontsize=12, fontweight='bold')
ax1.set_ylabel('Rank (Lower is Better)', fontsize=12, fontweight='bold')
ax1.set_title('Sequence Rank Trajectory - Linear Scale', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.invert_yaxis()  # Rank the small the better
ax1.set_xticks(range(1, 15))

# Fig.2: Rank comparison（log scale）
ax2 = axes[0, 1]
for seq_name, color in zip(target_sequences.keys(), ['#FF6B6B', '#4ECDC4']):
    rounds = sorted(results[seq_name].keys())
    ranks = [results[seq_name][r]['rank'] if results[seq_name][r]['rank'] is not None else np.nan 
             for r in rounds]
    
    ax2.plot(rounds, ranks, marker='o', linewidth=2.5, markersize=10, 
            label=seq_name, color=color, alpha=0.8)

ax2.set_xlabel('SELEX Round', fontsize=12, fontweight='bold')
ax2.set_ylabel('Rank (Log Scale)', fontsize=12, fontweight='bold')
ax2.set_title('Sequence Rank Trajectory - Log Scale', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='best')
ax2.grid(True, alpha=0.3, linestyle='--', which='both')
ax2.set_yscale('log')
ax2.invert_yaxis()
ax2.set_xticks(range(1, 15))

# Fig.3: RPM trajectory
ax3 = axes[1, 0]
for seq_name, color in zip(target_sequences.keys(), ['#FF6B6B', '#4ECDC4']):
    rounds = sorted(results[seq_name].keys())
    rpkms = [results[seq_name][r]['rpm'] if results[seq_name][r]['rpm'] > 0 else np.nan 
             for r in rounds]
    
    ax3.plot(rounds, rpkms, marker='s', linewidth=2.5, markersize=10, 
            label=seq_name, color=color, alpha=0.8)

ax3.set_xlabel('SELEX Round', fontsize=12, fontweight='bold')
ax3.set_ylabel('RPM', fontsize=12, fontweight='bold')
ax3.set_title('RPM Trajectory', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10, loc='best')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xticks(range(1, 15))

# Fig.4: Count Trajectory
ax4 = axes[1, 1]
for seq_name, color in zip(target_sequences.keys(), ['#FF6B6B', '#4ECDC4']):
    rounds = sorted(results[seq_name].keys())
    counts = [results[seq_name][r]['count'] if results[seq_name][r]['count'] > 0 else np.nan 
              for r in rounds]
    
    ax4.plot(rounds, counts, marker='^', linewidth=2.5, markersize=10, 
            label=seq_name, color=color, alpha=0.8)

ax4.set_xlabel('SELEX Round', fontsize=12, fontweight='bold')
ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
ax4.set_title('Count Trajectory', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10, loc='best')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xticks(range(1, 15))

plt.tight_layout()
plt.savefig('sequence_rank_trajectory_2.svg', dpi=600, bbox_inches='tight')
plt.show()

# ==================== Export detailed statistic results ====================
print("\n" + "="*80)
print("Detailed statistic results")
print("="*80)

for seq_name in target_sequences.keys():
    print(f"\n【{seq_name}】")
    print(f"Sequence: {target_sequences[seq_name]}")
    print("-" * 80)
    print(f"{'Round':<8} {'Rank':<15} {'Count':<15} {'RPKM':<15} {'State':<10}")
    print("-" * 80)
    
    for round_num in sorted(results[seq_name].keys()):
        data = results[seq_name][round_num]
        if data['rank'] is not None:
            rank_str = f"{data['rank']:,}"
            count_str = f"{data['count']:,}"
            rpkm_str = f"{data['rpkm']:.2f}"
            status = "✓ Found"
        else:
            rank_str = "Not Found"
            count_str = "-"
            rpkm_str = "-"
            status = "✗ Missing"
        
        print(f"{round_num:<8} {rank_str:<15} {count_str:<15} {rpkm_str:<15} {status:<10}")

# ==================== Excel Export Excel ====================
try:
    # Create DataFrame for data export
    export_data = []
    for seq_name in target_sequences.keys():
        for round_num in sorted(results[seq_name].keys()):
            data = results[seq_name][round_num]
            export_data.append({
                'Sequence_Name': seq_name,
                'Sequence': target_sequences[seq_name],
                'Round': round_num,
                'Rank': data['rank'],
                'Count': data['count'],
                'RPKM': data['rpkm']
            })
    
    df_export = pd.DataFrame(export_data)
    df_export.to_excel('sequence_rank_summary_2.xlsx', index=False)
    print("\n✓ Results have exported: sequence_rank_summary.xlsx")
except Exception as e:
    print(f"\n⚠️  Failed to export Excel: {e}")

print("\n" + "="*80)
print("Analysis complete！")
print("Generating files...:")
print("  - sequence_rank_trajectory_2.svg")
print("  - sequence_rank_summary_2.xlsx")
print("="*80)
