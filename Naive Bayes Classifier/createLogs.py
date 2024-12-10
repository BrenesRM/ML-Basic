import csv
import random

# Define EventIDs and some sample sources/messages for each
event_data = [
    {"EventID": "500", "Source": "System", "Message": "A critical system error occurred."},
    {"EventID": "4624", "Source": "Security", "Message": "A successful logon occurred."},
    {"EventID": "4625", "Source": "Security", "Message": "A failed logon attempt was made."},
    {"EventID": "7036", "Source": "Service Control Manager", "Message": "A service entered the running state."},
    {"EventID": "7040", "Source": "Service Control Manager", "Message": "A service changed its configuration."},
]

# Generate a balanced dataset
balanced_data = []
samples_per_event = 3000  # Define the number of samples per EventID

for event in event_data:
    for _ in range(samples_per_event):
        balanced_data.append({
            "EventID": event["EventID"],
            "Source": event["Source"],
            "Message": event["Message"] + f" Random ID: {random.randint(1000, 9999)}"
        })

# Write the dataset to a CSV file
with open("windows_system_logs.csv", "w", newline="") as csvfile:
    fieldnames = ["EventID", "Source", "Message"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(balanced_data)

print("windows_system_logs.csv generated successfully!")
