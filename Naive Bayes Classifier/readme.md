# Event Analysis README

This README file explains the steps to interpret the results of the event analysis program (`event_analysis.py`) and details its functionality for future use.

---

## Overview
The `event_analysis.py` script is a machine learning program designed to analyze Windows system logs stored in CSV files. It classifies system events based on their content and generates a performance report including accuracy and classification metrics.

---

## Prerequisites
Before running the script, ensure the following:

1. **Python Environment**
   - Install Python 3.7 or higher.
   - Install required libraries listed in `requirements.txt`.
   
2. **CSV File**
   - Prepare a Windows system log CSV file containing event details in the following format:
     ```
     EventID,Source,Message
     4624,Security,A successful logon occurred.
     4625,Security,A failed logon attempt was made.
     ...
     ```

---

## Running the Program
1. Place your CSV file in the same directory as the script.
2. Run the script:
   ```bash
   python event_analysis.py
   ```
3. Enter the path to your CSV file when prompted, e.g., `./windows_system_logs.csv`.

---

## Outputs
The program outputs the following:

### 1. **Accuracy**
- This is the percentage of events the model correctly classified. For example, `Accuracy: 1.0` means 100% of the test events were correctly identified.

### 2. **Classification Report**
A detailed report showing the model's performance for each event type (EventID):
- **Precision**: How often the model's predictions were correct.
- **Recall**: How many of the actual events were detected by the model.
- **F1-Score**: A balance between precision and recall.
- **Support**: The number of samples for each EventID in the test data.

Example:
```
               precision    recall  f1-score   support

         500       1.00      1.00      1.00       900
        4624       1.00      1.00      1.00       900
        4625       1.00      1.00      1.00       900
        7036       1.00      1.00      1.00       900
        7040       1.00      1.00      1.00       900

    accuracy                           1.00      4500
   macro avg       1.00      1.00      1.00      4500
weighted avg       1.00      1.00      1.00      4500
```

### 3. **Sample Predictions**
A list of events from the test data showing the input text, true labels, and the model's predictions.

Example:
```
                                                    Text  True Label  Predicted Label
6920   Security A failed logon attempt was made. Rand...        4625              4625
11204  Service Control Manager A service entered the ...        7036              7036
```

---

## Future Use Cases
The script can be used in the following scenarios:
1. **System Monitoring:** Analyze and classify system logs to detect security-related events such as failed logins or service changes.
2. **Security Incident Response:** Quickly identify patterns or trends in system logs to respond to potential threats.
3. **Training and Evaluation:** Test the model on new datasets to validate its accuracy with real-world data.

---

## Troubleshooting
1. **Accuracy is Low**
   - Check if the training data matches the structure of the test data.
   - Ensure enough diverse samples are included in the training dataset.

2. **Undefined Metric Warnings**
   - These occur if certain event types are missing in the test dataset or if no predictions were made for an event type. This is typically a dataset issue, not a program error.

3. **CSV File Format Errors**
   - Ensure the CSV file has the correct headers (`EventID`, `Source`, `Message`).

---

## Notes
- For production use, test the model with varied and real-world datasets to ensure its robustness.
- Update the model periodically to include new types of events as they occur in logs.

---

## Author
Marlon Brenes

## License
This script is free to use for educational and testing purposes. For commercial use, contact the author.
