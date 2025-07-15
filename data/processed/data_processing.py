import pandas as pd
import data_processing_utils as dpu

from datetime import datetime

df_bounce_raw = pd.read_csv('shaylaz_bounce_augmented_raw.csv')
df_bounce_raw['due_date'] = pd.to_datetime(df_bounce_raw['due_date'], format = 'mixed', errors='coerce')

df_disposition_raw = pd.read_csv('shaylaz_disposition_augmented_raw.csv')
df_disposition_raw['created_at'] = pd.to_datetime('df_disposition_augmented_raw.csv', format = 'mixed', errors = 'coerce')

df_bounce_lan = dpu.aggregate_bounce_lan(df_bounce_raw)
df_disposition_lan = dpu.aggregate_disposition_lan(df_disposition_raw)

start_date_for_split = datetime(2023, 9, 5)
num_intervals = 6
interval_magnitude = 12

# Test intervals for raw data
df_bounce_raw_split = dpu.training_set_split(start_date_for_split, num_intervals, 
    interval_magnitude, df_bounce_raw, 'due_date')
df_disposition_raw_split = dpu.training_set_split(start_date_for_split, num_intervals, 
    interval_magnitude, df_disposition_raw, 'created_at')

# Test intervals for lan grouped
df_bounce_lan_split = df_bounce_raw_split.copy()
df_disposition_lan_split = df_bounce_raw_split.copy()

dpu.apply_function(
    df_bounce_lan_split, 
    lambda df_list: [dpu.aggregate_bounce_lan(interval) for interval in df_list])
dpu.apply_function(
    df_disposition_lan_split,
    lambda df_list: [dpu.aggregate_disposition_lan(interval) for interval in df_list])
