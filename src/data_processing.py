import pandas as pd
import data_processing_utils as dpu

from datetime import datetime

df_bounce_raw = pd.read_csv('data/raw/shaylaz_bounce_augmented_raw.csv')
df_bounce_raw['due_date'] = pd.to_datetime(df_bounce_raw['due_date'], format = 'mixed', errors='coerce')
df_bounce_raw['time_since_last_due'] = df_bounce_raw.groupby('lan')['due_date'].diff().dt.days
df_bounce_raw = dpu.add_streak_features(df_bounce_raw)

df_disposition_raw = pd.read_csv('data/raw/shaylaz_disposition_augmented_raw.csv')
df_disposition_raw['created_at'] = pd.to_datetime(df_disposition_raw['created_at'], format = 'mixed', errors = 'coerce')

df_raw_merged = dpu.merge_dispositions_to_dues(df_bounce_raw, df_disposition_raw)

df_bounce_lan = dpu.aggregate_bounce_lan(df_bounce_raw)
df_disposition_lan = dpu.aggregate_disposition_lan(df_raw_merged)

start_date_for_split = datetime(2023, 9, 5)
num_intervals = 6
interval_magnitude = 12

# Test intervals for raw data
df_bounce_raw_split = dpu.training_set_split(start_date_for_split, num_intervals, 
    interval_magnitude, df_bounce_raw, 'due_date')
df_disposition_raw_split = dpu.training_set_split(start_date_for_split, num_intervals, 
    interval_magnitude, df_raw_merged, 'created_at')

df_bounce_lan_split = df_bounce_raw_split.copy()
df_disposition_lan_split = df_bounce_raw_split.copy()

dpu.apply_function(df_bounce_lan_split, dpu.aggregate_bounce_lan)
dpu.apply_function(df_disposition_lan_split, dpu.aggregate_bounce_lan)