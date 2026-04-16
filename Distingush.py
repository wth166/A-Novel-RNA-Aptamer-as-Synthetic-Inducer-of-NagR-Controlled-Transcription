import csv

def filter_nagr_data(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            if row['Aptamer_Type'] == 'NagR':
                writer.writerow(row)

# Usage example
input_file = 'processed_results_combined_by_pear.csv'
output_file = 'NagR_new_constant.csv'
filter_nagr_data(input_file, output_file)
