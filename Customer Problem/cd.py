import pandas as pd

# Manually create the data as a dictionary
data = {
    "Age": [66, 52, 22, 25, 44, 39, 19, 33, 53, 64, 58, 33],
    "Number of cars owned": [1, 2, 0, 1, 0, 1, 0, 1, 2, 2, 2, 1],
    "Owns house": ["yes", "yes", "no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "no"],
    "Number of children": [2, 3, 0, 1, 0, 2, 0, 1, 2, 3, 2, 1],
    "Marital status": [
        "widowed", "married", "married", "single", "divorced", 
        "married", "single", "married", "divorced", "divorced", 
        "married", "single"
    ],
    "Owns a dog": ["no", "no", "yes", "no", "yes", "no", "no", "no", "no", "no", "yes", "no"],
    "Bought a boat": ["yes", "yes", "no", "no", "no", "no", "no", "no", "no", "no", "yes", "no"],
}

# Convert the dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

# Export to a CSV file
csv_file = "customer_data.csv"
df.to_csv(csv_file, index=False)

# Print the DataFrame to verify
print("Customer Data:")
print(df)
