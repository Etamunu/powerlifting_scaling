import pandas as pd

# Replace with the path to your CSV file
# The dataset can be downloaded at https://openpowerlifting.gitlab.io/opl-csv/bulk-csv.html
file_path = 'openpowerlifting.csv'

df = pd.read_csv(file_path)
df = df[(df['Equipment'] == 'Raw') & (df['Division'] == 'Open')]
df = df[['Sex', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg']]
df = df.dropna(subset=['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'BodyweightKg'])
output_file_path = 'openpower-filtered.csv'
df.to_csv(output_file_path, index=False)
