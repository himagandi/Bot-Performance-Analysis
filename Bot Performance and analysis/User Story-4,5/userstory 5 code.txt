
import os
import pandas as pd
import numpy as np
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logging.info("Pandas session initialized.")

# === 1. Load cleaned chatbot data from CSV ===
data_path = r"C:\Users\jagadeesh02.TRN\Desktop\Project Details\new data1000.csv"
df = pd.read_csv(data_path)

# Check the column names to identify the correct ones
logging.info("Columns in the dataset: %s", df.columns)

# === 2. Prepare Data ===
# Convert the 'timestamp' to datetime object
def parse_timestamp(x):
    try:
        return pd.to_datetime(x, format='%m/%d/%Y %H:%M')
    except ValueError:
        return pd.NaT

df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)

# === 3. Metrics Analysis ===

## 1. Average Response Time
df['day'] = df['Timestamp'].dt.date  # Extract date for daily analysis
daily_avg_response = df.groupby('day').agg(
    avg_response_time_ms=('Response Time (ms)', 'mean'),
    stddev_response_time_ms=('Response Time (ms)', 'std')
).reset_index()

## 2. Daily Success Rate (Based on 'conversation_success')
df['is_success'] = np.where(df['Conversation Success'] == 'Successful', 1, 0)
daily_success_rate = df.groupby('day').agg(
    success_rate_percent=('is_success', 'mean')
).reset_index()

# Convert success_rate_percent to percentage
daily_success_rate['success_rate_percent'] *= 100

## 3. Segmentation: Response by Intent Type
seg_response_intent = df.groupby(['Intent Detected', 'day']).agg(
    avg_response_time_ms=('Response Time (ms)', 'mean'),
    success_rate_percent=('is_success', 'mean')
).reset_index()

# Convert success_rate_percent to percentage
seg_response_intent['success_rate_percent'] *= 100

## 4. Segmentation: Response by User Sentiment
seg_response_sentiment = df.groupby(['User Sentiment', 'day']).agg(
    avg_response_time_ms=('Response Time (ms)', 'mean'),
    success_rate_percent=('is_success', 'mean')
).reset_index()

# Convert success_rate_percent to percentage
seg_response_sentiment['success_rate_percent'] *= 100

## 5. Outlier Analysis (based on Response Time)
mean_rt = df['Response Time (ms)'].mean()
stddev_rt = df['Response Time (ms)'].std()

outliers_df = df[df['Response Time (ms)'] > (mean_rt + 2 * stddev_rt)]

# === 4. Export Results to Local Files (CSV) for Reporting ===
output_dir = r"C:\Users\jagadeesh02.TRN\Desktop\Project Details\UserStories5Code"
os.makedirs(output_dir, exist_ok=True)

# Save the dataframes to CSV files
daily_avg_response.to_csv(os.path.join(output_dir, "daily_avg_response.csv"), index=False)
daily_success_rate.to_csv(os.path.join(output_dir, "daily_success_rate.csv"), index=False)
seg_response_intent.to_csv(os.path.join(output_dir, "seg_response_intent.csv"), index=False)
seg_response_sentiment.to_csv(os.path.join(output_dir, "seg_response_sentiment.csv"), index=False)
outliers_df.to_csv(os.path.join(output_dir, "response_outliers.csv"), index=False)

logging.info("Trend analysis completed and saved to the specified output directory.")


