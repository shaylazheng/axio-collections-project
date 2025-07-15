import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

def aggregate_disposition_lan(raw: pd.DataFrame) -> pd.DataFrame:
    '''
    An aggregation function that creates various features after
    grouping the original raw dataframe by the 'lan' field
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
    categorized['call_proportion'] = np.where(
        categorized['total_contacts'] != 0, categorized['call_frequency'] / categorized['total_contacts'], 0.0)
    categorized['sms_proportion'] = np.where(
        categorized['total_contacts'] != 0, categorized['sms_frequency'] / categorized['total_contacts'], 0.0)
    categorized['disposition_proportion'] = np.where(
        categorized['total_contacts'] != 0, categorized['disposition_frequency'] / categorized['total_contacts'], 0.0)
    
    # Special message-only variable
    categorized['is_message_only'] = categorized.apply(
        lambda row: 1 if row['call_count'] == 0 and row['field_count'] == 0 else 0, axis=1)
    return categorized

def aggregate_bounce_lan(raw: pd.DataFrame) -> pd.DataFrame:
    categorized = raw.groupby('lan').agg(
        avg_duration_between_dues=('time_since_last_due', 'mean'),
        total_num_dues=('due_date', 'count'),
        # Corrected lambda for num_pl_streaks: x is already the Series for 'streak_group_id_calc' in the group
        # This lambda will only count unique streak_group_id_calc where fl_pl_only is 1 FOR THAT SPECIFIC GROUP
        num_pl_streaks=('streak_group_id_calc', lambda x: x[df.loc[x.index, 'fl_pl_only'] == 1].nunique()),
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
        avg_current_pl_only_streak=('current_pl_only_streak', 'mean'),
        avg_previous_pl_only_streak=('previous_pl_only_streak', 'mean'),
        
        # Proportions of fl variables
        prop_fl_bounce=('fl_bounce', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0),
        prop_fl_unresolved=('fl_unresolved', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0),
        prop_fl_current_due=('fl_current_due', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0),
        prop_fl_bounce_tp3=('fl_bounce_tp3', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0),
        prop_fl_bounce_tp5=('fl_bounce_tp5', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0),
        prop_fl_pl_only=('fl_pl_only', lambda x: x.sum() / x.count() if x.count() > 0 else 0.0)
    )

def training_set_split(start: datetime, num_intervals: int, interval_magnitude: int, 
                       raw: pd.DataFrame, date_col_name: str) -> dict:
    df_split = {}

    for i in range(num_intervals):
        current_start = start + relativedelta(months = i)
        current_end = current_start + relativedelta(months = interval_magnitude) - relativedelta(days = 1)

        key = current_start.strfttime('%b_%y').lower()

        df_interval_split = raw[
            ((raw[date_col_name].dt.date >= current_start.date())&
             (raw[date_col_name].dt.date >= current_end.date()))
        ].copy()

        df_split[key] = df_interval_split
    
    return df_split

def apply_function(raw: dict, function):
    for item in raw:
        raw[item] = function(raw[item])