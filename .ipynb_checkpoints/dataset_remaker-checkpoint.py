import pandas as pd
import os

input_filename = "train.csv"
df = pd.read_csv(input_filename)

df = df.drop(columns=["keyword", "location"], errors='ignore')

name, ext = os.path.splitext(input_filename)
output_filename = f"{name}_modified{ext}"

df.to_csv(output_filename, index=False)
