# Extract sequence and counts from Fastq
import csv
from collections import defaultdict

def parse_fastq(fastq_file):
    seq_counts = defaultdict(int)
    with open(fastq_file, 'r') as f:
        while True:
            header = f.readline().strip()  # Read header line
            if not header:  # Check for end of file
                break
            seq = f.readline().strip()  # Read sequence line
            f.readline()  # Skip the '+' line
            f.readline()  # Skip the quality line
            seq_counts[seq] += 1  # Count sequences
    return seq_counts

def main(fastq_files, output_file):
    all_counts = defaultdict(int)

    for fastq_file in fastq_files:
        seq_counts = parse_fastq(fastq_file)
        for seq, count in seq_counts.items():
            all_counts[seq] += count  # Aggregate counts

    # Filter out sequences with counts below threshold
    filtered_counts = {seq: count for seq, count in all_counts.items() if count >= 1}

    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Sequence', 'Count'])
        for seq, count in sorted(filtered_counts.items()):
            writer.writerow([seq, count])

fastq_files = ['DasR-NagR.assembled.fastq']
output_file = 'DasR-NagR.assemble.csv'
main(fastq_files, output_file)

