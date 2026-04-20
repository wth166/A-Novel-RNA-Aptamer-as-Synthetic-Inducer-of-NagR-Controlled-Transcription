import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Set log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set plot format
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def analyze_selex_enrichment(summary_files, total_sequences_file, output_dir='selex_analysis'):
    """
    Enrichment analysis of target sequences during SELEX process
    
    Parameters:
    - summary_files: 字典，{round_name: summary_csv_path}
      例如: {'Round1': 'statistics_round1_summary.csv', 'Round2': ...}
    - total_sequences_file: 包含每轮总序列数的CSV文件
    - output_dir: export directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("="*80)
    logging.info("Start SELEX enrichment analysis")
    logging.info("="*80)
    
    # Read Summary data for each round
    all_data = []
    
    for round_name, summary_file in sorted(summary_files.items()):
        logging.info(f"\nRead {round_name}: {summary_file}")
        
        try:
            df = pd.read_csv(summary_file)
            df['Round'] = round_name
            
            # Extract Round number（e.g.: Round1, Round2, ...）
            round_num = int(round_name.replace('Round', ''))
            df['Round_Number'] = round_num
            
            all_data.append(df)
            logging.info(f"  Found {len(df)} Target_Type")
            
        except Exception as e:
            logging.error(f"  Failed in reading file: {str(e)}")
            continue
    
    if not all_data:
        logging.error("Couldn't find any data！")
        return
    
    # Combined all rounds
    combined_df = pd.concat(all_data, ignore_index=True)
    logging.info(f"\n Total number after combine {len(combined_df)} records")
    
    # Read total sequence
    logging.info(f"\nRead total number files: {total_sequences_file}")
    try:
        total_seq_df = pd.read_csv(total_sequences_file)
        logging.info(f"  File includes: {total_seq_df.columns.tolist()}")
        
        # If file includes'Round' and 'Total_Sequences' column 
        # If name of column is wrong, will adjust
        if 'Round' not in total_seq_df.columns:
            # Try to find possible rounds
            round_cols = [col for col in total_seq_df.columns if 'round' in col.lower()]
            if round_cols:
                total_seq_df.rename(columns={round_cols[0]: 'Round'}, inplace=True)
        
        # Search total sequence columns
        total_cols = [col for col in total_seq_df.columns 
                     if any(keyword in col.lower() for keyword in ['total', 'count', 'number', 'sequences'])]
        
        if total_cols:
            total_seq_df.rename(columns={total_cols[0]: 'Total_Sequences'}, inplace=True)
        
        logging.info(f"  In total {len(total_seq_df)} rounds")
        logging.info(f"  Total sequence range: {total_seq_df['Total_Sequences'].min()} - {total_seq_df['Total_Sequences'].max()}")
        
    except Exception as e:
        logging.error(f"  Failed in Read total sequence file: {str(e)}")
        logging.info("  Will use matched sequences as approximate total matched sequence")
        # Create a estimated total matched records
        total_seq_df = combined_df.groupby('Round')['Total_Matched_Records'].sum().reset_index()
        total_seq_df.columns = ['Round', 'Total_Sequences']
    
    # Combine total sequence
    combined_df = combined_df.merge(total_seq_df, on='Round', how='left')
    
    # Calculate percentage
    combined_df['Percentage'] = (combined_df['Total_Matched_Records'] / 
                                 combined_df['Total_Sequences']) * 100
    
    # Calculate enrichment fold (compared to first round)
    target_types = combined_df['Target_Type'].unique()
    
    for target_type in target_types:
        target_data = combined_df[combined_df['Target_Type'] == target_type].copy()
        target_data = target_data.sort_values('Round_Number')
        
        if len(target_data) > 0:
            first_round_pct = target_data.iloc[0]['Percentage']
            if first_round_pct > 0:
                enrichment = target_data['Percentage'] / first_round_pct
            else:
                enrichment = pd.Series([0] * len(target_data))
            
            combined_df.loc[combined_df['Target_Type'] == target_type, 'Enrichment_Fold'] = \
                enrichment.values
    
    # Save and combine the data
    output_csv = f'{output_dir}/selex_enrichment_data.csv'
    combined_df.to_csv(output_csv, index=False)
    logging.info(f"\nData Saved: {output_csv}")
    
    # Generate enrichment plot
    generate_enrichment_plots(combined_df, output_dir)
    
    # Generate enrichment report
    generate_enrichment_report(combined_df, output_dir)
    
    return combined_df

def generate_enrichment_plots(df, output_dir):
    """enerate Enrichment analysis plots"""
    
    target_types = sorted(df['Target_Type'].unique())
    rounds = sorted(df['Round_Number'].unique())
    
    logging.info(f"\n Generate plots...")
    logging.info(f"  Number of target type: {len(target_types)}")
    logging.info(f"  Number of rounds: {len(rounds)}")
    
    # 1. Percentage Barchart
    generate_percentage_barplot(df, target_types, rounds, output_dir)
    
    # 2. Enrichment_fold linechart
    generate_enrichment_lineplot(df, target_types, rounds, output_dir)
    
    # 3. Heatmap
    generate_enrichment_heatmap(df, target_types, rounds, output_dir)
    
    # 4. Comprejemsive plot
    generate_comprehensive_plot(df, target_types, rounds, output_dir)

def generate_percentage_barplot(df, target_types, rounds, output_dir):
    """Generate percentage barchart"""
    
    fig, ax = plt.subplots(figsize=(max(12, len(rounds)*1.5), 8))
    
    x = np.arange(len(rounds))
    width = 0.8 / len(target_types)
    colors = plt.cm.Set3(np.linspace(0, 1, len(target_types)))
    
    for i, target_type in enumerate(target_types):
        target_data = df[df['Target_Type'] == target_type].sort_values('Round_Number')
        percentages = target_data['Percentage'].values
        
        offset = (i - len(target_types)/2 + 0.5) * width
        bars = ax.bar(x + offset, percentages, width, label=target_type, 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add labels
        for j, (bar, pct) in enumerate(zip(bars, percentages)):
            if pct > 0.1:  # Only show values > 0.1%
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.2f}%', ha='center', va='bottom', 
                       fontsize=8, rotation=0)
    
    ax.set_xlabel('SELEX Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage of Total Sequences (%)', fontsize=13, fontweight='bold')
    ax.set_title('Target Sequence Enrichment Across SELEX Rounds', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Round {r}' for r in rounds], fontsize=11)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = f'{output_dir}/enrichment_percentage_barplot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  Save: {output_file}")

def generate_enrichment_lineplot(df, target_types, rounds, output_dir):
    """Generate line chart for enrichment_fold"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(target_types)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, target_type in enumerate(target_types):
        target_data = df[df['Target_Type'] == target_type].sort_values('Round_Number')
        
        ax.plot(target_data['Round_Number'], target_data['Enrichment_Fold'], 
               marker=markers[i % len(markers)], linewidth=2.5, markersize=8,
               label=target_type, color=colors[i], alpha=0.8)
        
        # Add labels
        for x, y in zip(target_data['Round_Number'], target_data['Enrichment_Fold']):
            if not np.isnan(y):
                ax.text(x, y, f'{y:.1f}x', fontsize=8, 
                       ha='center', va='bottom', alpha=0.7)
    
    # Add line of reference
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, 
              alpha=0.5, label='No enrichment (1x)')
    
    ax.set_xlabel('SELEX Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('Enrichment Fold (relative to Round 1)', fontsize=13, fontweight='bold')
    ax.set_title('Target Sequence Enrichment Dynamics', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(rounds)
    ax.set_xticklabels([f'R{r}' for r in rounds], fontsize=11)
    ax.legend(loc='best', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = f'{output_dir}/enrichment_fold_lineplot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  Save: {output_file}")

def generate_enrichment_heatmap(df, target_types, rounds, output_dir):
    """Generate enrichment heatmap"""
    
    # Create matrix
    matrix = np.zeros((len(target_types), len(rounds)))
    
    for i, target_type in enumerate(target_types):
        for j, round_num in enumerate(rounds):
            data_point = df[(df['Target_Type'] == target_type) & 
                           (df['Round_Number'] == round_num)]
            if len(data_point) > 0:
                matrix[i, j] = data_point['Percentage'].values[0]
    
    fig, ax = plt.subplots(figsize=(max(10, len(rounds)*0.8), max(6, len(target_types)*0.6)))
    
    # Use logarithmic scale color mapping (if the numerical range is large).
    vmax = matrix.max()
    vmin = matrix[matrix > 0].min() if (matrix > 0).any() else 0.01
    
    if vmax / vmin > 100:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(vmin, 0.01), vmax=vmax)
        cmap = 'YlOrRd'
    else:
        norm = None
        cmap = 'YlOrRd'
    
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    
    # Set scale
    ax.set_xticks(np.arange(len(rounds)))
    ax.set_yticks(np.arange(len(target_types)))
    ax.set_xticklabels([f'Round {r}' for r in rounds], fontsize=10)
    ax.set_yticklabels(target_types, fontsize=10)
    
    # Add values
    for i in range(len(target_types)):
        for j in range(len(rounds)):
            value = matrix[i, j]
            if value > 0:
                text_color = 'white' if value > vmax * 0.5 else 'black'
                text = ax.text(j, i, f'{value:.2f}%', ha="center", va="center",
                             color=text_color, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('SELEX Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Type', fontsize=12, fontweight='bold')
    ax.set_title('Target Sequence Percentage Heatmap', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Color bar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Percentage (%)', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    output_file = f'{output_dir}/enrichment_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  Save: {output_file}")

def generate_comprehensive_plot(df, target_types, rounds, output_dir):
    """Generate comprehensive plot"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(target_types)))
    
    # Sub-figure 1: Change of Percentage
    for i, target_type in enumerate(target_types):
        target_data = df[df['Target_Type'] == target_type].sort_values('Round_Number')
        ax1.plot(target_data['Round_Number'], target_data['Percentage'], 
                marker='o', linewidth=2, label=target_type, color=colors[i])
    
    ax1.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('A. Percentage Trends', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Sub-figure 2: Enrichment Fold
    for i, target_type in enumerate(target_types):
        target_data = df[df['Target_Type'] == target_type].sort_values('Round_Number')
        ax2.plot(target_data['Round_Number'], target_data['Enrichment_Fold'], 
                marker='s', linewidth=2, label=target_type, color=colors[i])
    
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Enrichment Fold', fontsize=11, fontweight='bold')
    ax2.set_title('B. Enrichment Dynamics', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Sub-figure 3: Absolute Counts
    for i, target_type in enumerate(target_types):
        target_data = df[df['Target_Type'] == target_type].sort_values('Round_Number')
        ax3.plot(target_data['Round_Number'], target_data['Total_Matched_Records'], 
                marker='^', linewidth=2, label=target_type, color=colors[i])
    
    ax3.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Matched Sequences Count', fontsize=11, fontweight='bold')
    ax3.set_title('C. Absolute Counts', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Sub-figure 4: Last round vs First round
    first_round = rounds[0]
    last_round = rounds[-1]
    
    first_data = df[df['Round_Number'] == first_round].set_index('Target_Type')['Percentage']
    last_data = df[df['Round_Number'] == last_round].set_index('Target_Type')['Percentage']
    
    x_pos = np.arange(len(target_types))
    width = 0.35
    
    ax4.bar(x_pos - width/2, [first_data.get(t, 0) for t in target_types], 
           width, label=f'Round {first_round}', alpha=0.8, color='lightblue')
    ax4.bar(x_pos + width/2, [last_data.get(t, 0) for t in target_types], 
           width, label=f'Round {last_round}', alpha=0.8, color='coral')
    
    ax4.set_xlabel('Target Type', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax4.set_title(f'D. First vs Last Round Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(target_types, rotation=45, ha='right', fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Comprehensive SELEX Enrichment Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = f'{output_dir}/enrichment_comprehensive_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  Save: {output_file}")

def generate_enrichment_report(df, output_dir):
    """Generate enrichment report"""
    
    report_file = f'{output_dir}/enrichment_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SELEX enrichment analysis report\n")
        f.write("="*80 + "\n\n")
        
        target_types = sorted(df['Target_Type'].unique())
        rounds = sorted(df['Round_Number'].unique())
        
        f.write(f"分析概况:\n")
        f.write(f"  - Target types: {len(target_types)}\n")
        f.write(f"  - SELEX Round: Round {rounds[0]} - Round {rounds[-1]}\n")
        f.write(f"  - Total rounds: {len(rounds)}\n\n")
        
        # Summarize each target type
        for target_type in target_types:
            f.write(f"\n{'='*80}\n")
            f.write(f"Target Type: {target_type}\n")
            f.write(f"{'='*80}\n\n")
            
            target_data = df[df['Target_Type'] == target_type].sort_values('Round_Number')
            
            first_round = target_data.iloc[0]
            last_round = target_data.iloc[-1]
            
            f.write(f"Frist round (Round {int(first_round['Round_Number'])}):\n")
            f.write(f"  - Total matched sequence: {int(first_round['Total_Matched_Records'])}\n")
            f.write(f"  - Proportion: {first_round['Percentage']:.4f}%\n\n")
            
            f.write(f"Last Round (Round {int(last_round['Round_Number'])}):\n")
            f.write(f"  - Total matched sequence: {int(last_round['Total_Matched_Records'])}\n")
            f.write(f"  - Proportion: {last_round['Percentage']:.4f}%\n")
            f.write(f"  - Enrichment_fold: {last_round['Enrichment_Fold']:.2f}x\n\n")
            
            # Find maximal and minimal 
            max_row = target_data.loc[target_data['Percentage'].idxmax()]
            min_row = target_data.loc[target_data['Percentage'].idxmin()]
            
            f.write(f"Peak (Round {int(max_row['Round_Number'])}):\n")
            f.write(f"  - Proportion: {max_row['Percentage']:.4f}%\n\n")
            
            f.write(f"Dip (Round {int(min_row['Round_Number'])}):\n")
            f.write(f"  - Peopoerion: {min_row['Percentage']:.4f}%\n\n")
    
    logging.info(f"\nEnrichment report saved: {report_file}")

if __name__ == "__main__":
    # Config
    summary_files = {
        'Round1': 'alignment_details_round1.csv',
        'Round2': 'alignment_details_round2.csv',
        'Round3': 'alignment_details_round3.csv',
        'Round4': 'alignment_details_round4.csv',
        'Round5': 'alignment_details_round5.csv',
        'Round6': 'alignment_details_round6.csv',
        'Round7': 'alignment_details_round7.csv',
        'Round8': 'alignment_details_round8.csv',
        'Round9': 'alignment_details_round9.csv',
        'Round10': 'alignment_details_round10.csv',
        'Round11': 'alignment_details_round11.csv',
        'Round12': 'alignment_details_round12.csv',
        'Round13': 'alignment_details_round13.csv',
        'Round14': 'alignment_details_round14.csv',
    }
    
    total_sequences_file = 'NagR_round1-14.csv'
    output_dir = 'selex_analysis_position_variation'
    
    # Run analysis
    results = analyze_selex_enrichment(summary_files, total_sequences_file, output_dir)
    
    logging.info("\n" + "="*80)
    logging.info("SELEX enrichment analysis complete！")
    logging.info("="*80)
    logging.info("\nGenerate files:")
    logging.info("  1. selex_enrichment_data.csv - Combine enrichment analysis data")
    logging.info("  2. enrichment_percentage_barplot.png - percentage_barchart")
    logging.info("  3. enrichment_fold_lineplot.png - enrichment_fold_linechart")
    logging.info("  4. enrichment_heatmap.png - enrichment_heatmap")
    logging.info("  5. enrichment_comprehensive_plot.png - Comprehensive plot")
    logging.info("  6. enrichment_report.txt - Detailed report")
