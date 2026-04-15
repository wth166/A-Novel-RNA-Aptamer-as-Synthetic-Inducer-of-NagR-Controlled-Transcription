import csv
from collections import defaultdict
import time
import math
from Bio import pairwise2

# Integrated barcode information
BARCODES = {
    'CTAAGTC': ('DasR', 1),
    'GACTTAG': ('DasR', 1),
    'TGTGACA': ('DasR', 2),
    'TGTCACA': ('DasR', 2),
    'GATAGGT': ('DasR', 3),
    'ACCTATC': ('DasR', 3),
    'GGAAGAA': ('DasR', 4),
    'TTCTTCC': ('DasR', 4),
    'CAGAGAA': ('DasR', 5),
    'TTCTCTG': ('DasR', 5),
    'ACCAGAA': ('DasR', 6),
    'TTCTGGT': ('DasR', 6),
    'GACGGAA': ('DasR', 7),
    'TTCCGTC': ('DasR', 7),
    'AGTGGAA': ('DasR', 8),
    'TTCCACT': ('DasR', 8),
    'TCACGAA': ('DasR', 9),
    'TTCGTGA': ('DasR', 9),
    'GTTCGAA': ('DasR', 10),
    'TTCGAAC': ('DasR', 10),
    'CGCTGAA': ('DasR', 11),
    'TTCAGCG': ('DasR', 11),
    'CCAACAA': ('DasR', 12),
    'TTGTTGG': ('DasR', 12),
    'GTAGCAA': ('DasR', 13),
    'TTGCTAC': ('DasR', 13),
    'AAGGCAA': ('DasR', 14),
    'TTGCCTT': ('DasR', 14),
    'AGACCAA': ('NagR', 1),
    'TTGGTCT': ('NagR', 1),
    'CTGCCAA': ('NagR', 2),
    'TTGGCAG': ('NagR', 2),
    'GCGTCAA': ('NagR', 3),
    'TTGACGC': ('NagR', 3),
    'CCGGTAA': ('NagR', 4),
    'TTACCGG': ('NagR', 4),
    'TGGCTAA': ('NagR', 5),
    'TTAGCCA': ('NagR', 5),
    'CACCTAA': ('NagR', 6),
    'TTAGGTG': ('NagR', 6),
    'TTGGAGA': ('NagR', 7),
    'TCTCCAA': ('NagR', 7),
    'CCTTAGA': ('NagR', 8),
    'TCTAAGG': ('NagR', 8),
    'TAGCCGA': ('NagR', 9),
    'TCGGCTA': ('NagR', 9),
    'AACTCGA': ('NagR', 10),
    'TCGAGTT': ('NagR', 10),
    'GCTATGA': ('NagR', 11),
    'TCATAGC': ('NagR', 11),
    'TTCCTGA': ('NagR', 12),
    'TCAGGAA': ('NagR', 12),
    'GAACACA': ('NagR', 13),
    'TGTGTTC': ('NagR', 13),
    'TCGCACA': ('NagR', 14),
    'TGTGCGA': ('NagR', 14)
}

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))

def identify_barcode(sequence):
    for barcode, info in BARCODES.items():
        if barcode in sequence[:30] or barcode in sequence[-30:]:
            return info
    return None, None

def extract_aptamer(sequence):
    forward_start = "CCTTCACTGC"
    forward_end = "GGCACCA"
    reverse_start = "CCGTGGTGCC"
    reverse_end = "GCAGTGA"

    # Check for forward sequence
    start_index = sequence.find(forward_start)
    if start_index != -1:
        end_index = sequence.find(forward_end, start_index)
        if end_index != -1:
            aptamer = sequence[start_index + len(forward_start):end_index]
            if len(aptamer) == 40:
                return aptamer

    # Check for reverse sequence
    start_index = sequence.find(reverse_start)
    if start_index != -1:
        end_index = sequence.find(reverse_end, start_index)
        if end_index != -1:
            aptamer = sequence[start_index + len(reverse_start):end_index]
            aptamer = reverse_complement(aptamer)
            if len(aptamer) == 40:
                return aptamer

    # If no exact match found, try alignment
    return extract_aptamer_by_alignment(sequence)

def extract_aptamer_by_alignment(sequence):
    reference_sequence = "GAGCTCAGCCTTCACTGCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGGCACCACGGTCGGATCC"
    alignments = pairwise2.align.globalms(sequence, reference_sequence, 2, -1, -2, -1)
    if not alignments:
        return None

    best_alignment = alignments[0]
    aligned_seq, aligned_ref, score, start_pos, end_pos = best_alignment

    start_marker = "GAGCTCAGCCTTCAC"
    end_marker = "GGCACCACGGTCGGA"

    start_marker_pos = aligned_ref.find(start_marker)
    end_marker_pos = aligned_ref.find(end_marker, start_marker_pos + len(start_marker))

    if start_marker_pos == -1 or end_marker_pos == -1:
        return None

    start_pos = start_marker_pos + len(start_marker)
    end_pos = end_marker_pos

    aptamer = aligned_seq[start_pos:end_pos].replace('-', '')

    if len(aptamer) == 40:
        return aptamer
    return None

def calculate_rpm(count, total_reads_in_round):
    if count == 0 or total_reads_in_round == 0:
        return 0
    return (count * 10**6) / total_reads_in_round

def log10_transform(value):
    return math.log10(value) if value > 0 else 0

def process_csv_files(csv_files, output_file):
    # aptamer_counts[aptamer_type][round][aptamer_sequence] = total count
    aptamer_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # round_totals[aptamer_type][round] = total reads in that group
    round_totals = defaultdict(lambda: defaultdict(int))

    total_sequences = 0
    processed_sequences = 0
    found_aptamers = 0
    start_time = time.time()

    # First pass: accumulate counts and round totals
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                total_sequences += 1
                sequence = row[0].strip().upper()
                count = int(row[1])

                aptamer_type, round_no = identify_barcode(sequence)
                if aptamer_type is None or round_no is None:
                    continue

                aptamer = extract_aptamer(sequence)

                if aptamer is not None:
                    aptamer_counts[aptamer_type][round_no][aptamer] += count
                    round_totals[aptamer_type][round_no] += count
                    found_aptamers += 1

                processed_sequences += 1
                if processed_sequences % 10000 == 0:
                    elapsed_time = time.time() - start_time
                    print(
                        f"Processed {processed_sequences} sequences. "
                        f"Found aptamers: {found_aptamers}. "
                        f"Elapsed time: {elapsed_time:.2f} seconds"
                    )

    # Second pass: calculate RPM and write output
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([
            "Aptamer_Type",
            "Round",
            "Sequence",
            "Count",
            "Round_Total_Reads",
            "RPM",
            "Log10_RPM"
        ])

        for apt_type, rounds in aptamer_counts.items():
            for rnd, aptamers in rounds.items():
                total_reads_in_round = round_totals[apt_type][rnd]

                for aptamer, cnt in aptamers.items():
                    rpm = calculate_rpm(cnt, total_reads_in_round)
                    log10_rpm = log10_transform(rpm)

                    writer.writerow([
                        apt_type,
                        rnd,
                        aptamer,
                        cnt,
                        total_reads_in_round,
                        rpm,
                        log10_rpm
                    ])

    # Optional: also export round totals for checking
    with open("round_total_reads_summary.csv", 'w', newline='') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["Aptamer_Type", "Round", "Total_Reads"])

        for apt_type in sorted(round_totals.keys()):
            for rnd in sorted(round_totals[apt_type].keys()):
                writer.writerow([apt_type, rnd, round_totals[apt_type][rnd]])

    print(f"Total processed sequences: {total_sequences}")
    print(f"Total aptamers found: {found_aptamers}")
    print(f"Results written to {output_file}")
    print("Round total reads summary written to round_total_reads_summary.csv")

def main():
    input_files = ["DasR-NagR.assemble.csv"]  # List of input CSV files
    output_file = "processed_results_combined_by_pear.csv"

    start_time = time.time()
    print("Processing CSV files...")
    process_csv_files(input_files, output_file)
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
