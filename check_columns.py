import pandas as pd

# Load your CSV
df = pd.read_csv("archive/urls.csv")

# Print all column names
print("Columns in CSV:", df.columns)
