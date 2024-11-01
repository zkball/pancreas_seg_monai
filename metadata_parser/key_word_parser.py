import pandas as pd
import re



def split_and_count(text):
    # Split the text by commas, periods, and semicolons
    parts = re.split('[，；。,;.]', str(text))
    # Remove leading/trailing whitespace and empty strings
    return [part.strip() for part in parts if part.strip()]

least_required_times = 20

# Read the CSV file
import os
base_dir='/raid/datasets'
df = pd.read_csv(os.path.join(base_dir,'425病例信息.csv'), sep=',', skiprows=1)  # Skip the second row

# Process column E (index 4)
df['SplitParts'] = df.iloc[:, 4].apply(split_and_count)

# Count occurrences of each part
all_parts = [part for parts in df['SplitParts'] for part in parts]
part_counts = pd.Series(all_parts).value_counts()

# Get parts that appear at least 20 times
reserved_parts = part_counts[part_counts >= least_required_times]
print("\nReserved parts:", reserved_parts)
reserved_parts = reserved_parts.index.tolist()
reserved_txt = ",".join(reserved_parts)
with open("./key_words.txt", "w") as kw:
    kw.writelines([reserved_txt])

# Create binary columns for reserved parts
for part in reserved_parts:
    df[f'Has_{part}'] = df['SplitParts'].apply(lambda x: int(part in x))

# Combine binary columns into a single column
df['ReservedParts'] = df[[f'Has_{part}' for part in reserved_parts]].apply(
    lambda row: ','.join(row.astype(str)), axis=1
)

# import pdb;pdb.set_trace()

# Drop intermediate columns
df = df.drop(columns=['SplitParts'] + [f'Has_{part}' for part in reserved_parts])

# Save the result
df.to_csv('./processed_test.csv', index=False)

print(df)

print()