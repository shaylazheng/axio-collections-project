import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import DateOffset

def aggregate_disposition_lan(raw: pd.DataFrame) -> pd.DataFrame:
    '''
    An aggregation function that creates various features after
    grouping the original disposition dataframe by the 'lan' field
    '''
    categorized = raw.groupby('lan').agg(
        # Disposition counts
        call_count = ('type', lambda x: (x == 'CALL').sum()),
        sms_count = ('type', lambda x: (x == 'SMS').sum()),
        field_count = ('type', lambda x: (x == 'DISPOSITION').sum()),
        total_contacts = ('type', 'count'),

        # Characteristics of dispositions
        answered = ('contact_category', lambda x: (x == 'answered/contactable').sum()),
        not_answered = ('contact_category', lambda x: (x == 'not answered/uncontactable').sum()),
        pos_response = ('response_sentiment', lambda x: (x == 'positive response').sum()),
        neg_response = ('response_sentiment', lambda x: (x == 'negative response').sum()),
        neu_response = ('response_sentiment', lambda x: (x == 'neutral response').sum()),
        unknown_response = ('response_sentiment', lambda x: (x == 'unclassified_response_sentiment').sum()),

        # Contact duration / interval
        avg_time_between_calls = ('time_since_last_call', 'mean'),
        avg_time_between_sms = ('time_since_last_sms', 'mean'),
        avg_time_between_field = ('time_since_last_disposition_event', 'mean'),
        avg_time_between_all = ('time_since_last_contact', 'mean'),
        avg_call_duration = ('call_duration','mean')
    
    ).reset_index()

    # Disposition proportions
    categorized['call_frequency'] = np.where(
        categorized['total_contacts'] != 0, categorized['call_count'] / categorized['total_contacts'], 0.0)
    categorized['sms_frequency'] = np.where(
        categorized['total_contacts'] != 0, categorized['sms_count'] / categorized['total_contacts'], 0.0)
    categorized['field_frequency'] = np.where(
        categorized['total_contacts'] != 0, categorized['field_count'] / categorized['total_contacts'], 0.0)
    
    # Special message-only variable
    categorized['is_message_only'] = categorized.apply(
        lambda row: 1 if row['call_count'] == 0 and row['field_count'] == 0 else 0, axis=1)
    return categorized

def aggregate_bounce_lan(raw: pd.DataFrame) -> pd.DataFrame:
    '''
    An aggregation function that creates various features after
    grouping the original bounce dataframe by the 'lan' field
    '''

    categorized = raw.groupby('lan').agg(
        avg_duration_between_dues=('time_since_last_due', 'mean'),
        num_pl_streaks=('is_pl_streak_end', 'count'),
        avg_pl_streak_length = ('current_pl_streak', lambda x: x[raw.loc[x.index, 'is_pl_streak_end'] == 1].mean()),

        total_num_dues=('due_date', 'count'),

        avg_mob_due=('mob_due', 'mean'),
        avg_pre_due_pos=('pre_due_pos', 'mean'),
        avg_due_loan_count=('due_loan_count', 'mean'),
        avg_monthend_pos=('monthend_pos', 'mean'),
        avg_bounce_pos=('bounce_pos', 'mean'),
        avg_unresolved_pos=('unresolved_pos', 'mean'),
        avg_current_bounce_pos=('current_bounce_pos', 'mean'),
        avg_current_unresolved_pos=('current_unresolved_pos', 'mean'),
        avg_emi_amount_expected_retro=('emi_amount_expected_retro', 'mean'),
        avg_bounce_tp3_pos=('bounce_tp3_pos', 'mean'),
        avg_bounce_tp5_pos=('bounce_tp5_pos', 'mean'),
        avg_current_monthend_pos=('current_monthend_pos', 'mean'),

        avg_proportion_of_payment=('proportion_of_payment', 'mean'),
        avg_credit_utilization=('credit_utilization', 'mean'),

        # Proportions of fl variables
        prop_fl_bounce=('fl_bounce', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0),
        prop_fl_unresolved=('fl_unresolved', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0),
        prop_fl_current_due=('fl_current_due', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0),
        prop_fl_bounce_tp3=('fl_bounce_tp3', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0),
        prop_fl_bounce_tp5=('fl_bounce_tp5', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0),
    )

def training_set_split(start: datetime, num_intervals: int, interval_magnitude: int, 
                       raw: pd.DataFrame, date_col_name: str) -> dict:
    """
    Split the data set into an inputted number of intervals, with an inputted length
    of that interval, returning a dictionary
    """
    
    df_split = {}

    for i in range(num_intervals):
        current_start = start + relativedelta(months = i)
        current_end = current_start + relativedelta(months = interval_magnitude) - relativedelta(days = 1)

        key = current_start.strftime('%b_%y').lower()

        df_interval_split = raw[
            ((raw[date_col_name].dt.date >= current_start.date())&
             (raw[date_col_name].dt.date <= current_end.date()))
        ].copy()

        df_split[key] = df_interval_split
    
    return df_split

def add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Analyzes streaks of PL only dues and adds two feature columns
    to the input DataFrame.
    '''
    df['is_pl'] = (df['due_level_tenor_type'] == 'PL only').astype(int)

    df['streak_id'] = (df['is_pl'] != df.groupby('lan')['is_pl'].shift()).cumsum()
    df['current_pl_streak'] = df.groupby('streak_id').cumcount() + 1
    df['current_pl_streak'] = df['current_pl_streak'] * df['is_pl']

    is_next_not_pl = df.groupby('lan')['is_pl'].shift(-1).fillna(0) != 1
    df['is_pl_streak_end'] = ((df['is_pl'] == 1) & is_next_not_pl).astype(int)

    df.drop(columns=['is_pl', 'streak_id'], inplace=True)    
    return df

def create_dummy_data(num_customers=50, num_dues_per_customer=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic dummy due and disposition DataFrames for testing.
    """    
    num_rows_due = num_customers * num_dues_per_customer
    
    due_data = {
        'lan': np.repeat(np.arange(1001, 1001 + num_customers), num_dues_per_customer),
        'due_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 730, size=num_rows_due), unit='d'),
        'mob_due': np.random.randint(1, 24, size=num_rows_due),
        'fl_amz_thin': np.random.choice([0, 1], size=num_rows_due, p=[0.9, 0.1]),
        'experian_bucket': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=num_rows_due),
        'pre_due_pos': np.random.uniform(1000, 20000, size=num_rows_due),
        'due_loan_count': np.random.randint(1, 5, size=num_rows_due),
        'due_level_tenor_type': np.random.choice(['PL only', 'EMI only', 'mixed'], size=num_rows_due, p=[0.6, 0.3, 0.1]),
        'due_level_interest_type': np.random.choice(['fixed', 'floating'], size=num_rows_due),
        'cust_level_tenor_type': np.random.choice(['PL only', 'EMI only', 'mixed'], size=num_rows_due),
        'cust_level_interest_type': np.random.choice(['fixed', 'floating'], size=num_rows_due),
        'line_bucket': np.random.randint(1, 6, size=num_rows_due),
        'age': np.random.randint(22, 65, size=num_rows_due),
        'emi_counter': np.random.randint(1, 12, size=num_rows_due),
        'fl_bounce': np.random.choice([0, 1], size=num_rows_due, p=[0.7, 0.3]),
        'monthend_pos': np.random.uniform(0, 15000, size=num_rows_due),
        'bounce_pos': np.random.uniform(0, 5000, size=num_rows_due),
        'fl_unresolved': np.random.choice([0, 1], size=num_rows_due, p=[0.8, 0.2]),
        'unresolved_pos': np.random.uniform(0, 5000, size=num_rows_due),
        'fl_current_due': np.random.choice([0, 1], size=num_rows_due, p=[0.95, 0.05]),
        'current_bounce_pos': np.random.uniform(0, 5000, size=num_rows_due),
        'fl_current_unresolved': np.random.choice([0, 1], size=num_rows_due, p=[0.9, 0.1]),
        'current_unresolved_pos': np.random.uniform(0, 5000, size=num_rows_due),
        'emi_amount_expected_retro': np.random.uniform(500, 5000, size=num_rows_due),
        'fl_bounce_tp3': np.random.choice([0, 1], size=num_rows_due, p=[0.75, 0.25]),
        'fl_bounce_hybrid': np.random.choice([0, 1], size=num_rows_due, p=[0.9, 0.1]),
        'bounce_tp3_pos': np.random.uniform(0, 4000, size=num_rows_due),
        'fl_bounce_tp5': np.random.choice([0, 1], size=num_rows_due, p=[0.8, 0.2]),
        'bounce_tp5_pos': np.random.uniform(0, 4000, size=num_rows_due),
        'current_monthend_pos': np.random.uniform(0, 15000, size=num_rows_due),
        'annual_limit': np.random.choice([50000, 100000, 200000], size=num_rows_due),
        'proportion_of_payment': np.random.rand(num_rows_due),
        'credit_utilization': np.random.rand(num_rows_due),
        'current_pl_only_streak': np.random.randint(0, 6, size=num_rows_due),
        'previous_pl_only_streak': np.random.randint(0, 6, size=num_rows_due),
        'fl_pl_only': np.random.choice([0, 1], size=num_rows_due, p=[0.4, 0.6]) # Needed for streak calculations
    }
    df_due = pd.DataFrame(due_data)
    df_due = df_due.sort_values(by=['lan', 'due_date']).reset_index(drop=True)

    num_rows_disp = num_rows_due * 3 # Assume 3 contacts per due on average
    disp_data = {
        'lan': np.random.choice(df_due['lan'].unique(), size=num_rows_disp),
        'created_at': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 730, size=num_rows_disp), unit='d'),
        'disp_source': np.random.choice(['AUTO', 'MANUAL'], size=num_rows_disp),
        'type': np.random.choice(['CALL', 'SMS', 'DISPOSITION'], size=num_rows_disp, p=[0.5, 0.4, 0.1]),
        'disp_outcome': np.random.choice(['PAID', 'PTP', 'RNR', 'INVALID'], size=num_rows_disp),
        'disp_outcome_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 760, size=num_rows_disp), unit='d'),
        'label': 'some_label',
        'call_type': np.random.choice(['INBOUND', 'OUTBOUND'], size=num_rows_disp),
        'call_status': np.random.choice(['ANSWERED', 'NOT_ANSWERED'], size=num_rows_disp),
        'call_duration': np.random.uniform(0, 300, size=num_rows_disp),
        'fl_bounce': np.random.choice([0, 1], size=num_rows_disp, p=[0.7, 0.3]),
        'fl_current_due': np.random.choice([0, 1], size=num_rows_disp, p=[0.95, 0.05]),
        'contact_category': np.random.choice(['answered/contactable', 'not answered/uncontactable'], size=num_rows_disp),
        'response_sentiment': np.random.choice(['positive response', 'negative response', 'neutral response', 'unclassified_response_sentiment'], size=num_rows_disp),
    }
    df_disposition = pd.DataFrame(disp_data)
    
    df_disposition = pd.merge_asof(
        df_disposition.sort_values('created_at'),
        df_due[['lan', 'due_date']].sort_values('due_date'),
        left_on='created_at',
        right_on='due_date',
        by='lan',
        direction='nearest'
    )
    
    df_disposition = df_disposition.sort_values(by=['lan', 'created_at'])
    df_disposition['time_since_last_contact'] = df_disposition.groupby('lan')['created_at'].diff().dt.days
    
    for contact_type, col_name in [('CALL', 'time_since_last_call'), ('SMS', 'time_since_last_sms'), ('DISPOSITION', 'time_since_last_disposition_event')]:
        df_disp_type = df_disposition[df_disposition['type'] == contact_type].copy()
        df_disp_type[col_name] = df_disp_type.groupby('lan')['created_at'].diff().dt.days
        df_disposition = pd.merge(df_disposition, df_disp_type[['lan', 'created_at', col_name]], on=['lan', 'created_at'], how='left')

    return df_due, df_disposition

def merge_dispositions_to_dues(df_due: pd.DataFrame, df_disposition: pd.DataFrame) -> pd.DataFrame:
    """
    Correctly merges disposition events to their corresponding dues using a 
    time-aware join, bringing along the final outcome flags.
    """
    df_due_lookup = df_due[['lan', 'due_date', 'fl_bounce_tp3', 'fl_bounce_tp5']].copy()

    df_disposition_sorted = df_disposition.sort_values(by='created_at')
    df_due_lookup_sorted = df_due_lookup.sort_values(by='due_date')

    merged = pd.merge_asof(
        df_disposition_sorted,
        df_due_lookup_sorted,
        left_on='created_at',
        right_on='due_date',
        by='lan',
        direction='forward'
    )
    return merged

def apply_function(raw: dict, function):
    """
    Simply applies a function to a dictionary
    """
    for item in raw:
        raw[item] = function(raw[item])