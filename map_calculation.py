SUBMISSION_FILENAME = '../input/leash-BELKA/sample_submission.csv'

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from collections import defaultdict

key = pd.read_csv('/kaggle/input/belka-solution-key/solution_with_groups.csv')
sub = pd.read_csv(SUBMISSION_FILENAME)
sub_key = pd.merge(sub, key, on = 'id')

scores = []
public_scores = []
private_scores = []
    
precision_dict = defaultdict(list)

for protein in sub_key['protein_name'].unique():
    for group in sub_key['split_group'].unique():
        rows = (sub_key['protein_name'] == protein) & (sub_key['split_group'] == group)
        y_true = sub_key['binds_y'][rows]
        y_pred = sub_key['binds_x'][rows]
        precision = average_precision_score(y_true, y_pred)
        precision_dict[protein].append(precision)
        scores.append(precision)
        
        rows_public = (sub_key['protein_name'] == protein) & (sub_key['split_group'] == group) & (sub_key['Usage'] == 'Public')
        if any(rows_public):
            y_true = sub_key['binds_y'][rows_public]
            y_pred = sub_key['binds_x'][rows_public]
            precision = average_precision_score(y_true, y_pred)
            public_scores.append(precision)

        rows_private = (sub_key['protein_name'] == protein) & (sub_key['split_group'] == group) &  (sub_key['Usage'] == 'Private')
        if any(rows_private):
            y_true = sub_key['binds_y'][rows_private]
            y_pred = sub_key['binds_x'][rows_private]
            precision = average_precision_score(y_true, y_pred)
            private_scores.append(precision)

print("Overall MAP: ", round(np.mean(scores), 4))
print("Public MAP: ", round(np.mean(public_scores), 4))
print("Private MAP: ", round(np.mean(private_scores), 4))
pd.DataFrame.from_dict(precision_dict, orient = 'index', columns = sub_key['split_group'].unique())