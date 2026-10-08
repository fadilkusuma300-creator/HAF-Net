#!/usr/bin/env python3
"""Create numerical comparison tables directly from saved evaluation predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from hafnet.metrics import classification_metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    frame=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in args.predictions],ignore_index=True)
    frame=frame[frame['split']=='test'].copy()
    keys=['dataset','model','seed','row_id']
    if frame.duplicated(keys).any():raise ValueError('Duplicate model/seed/test rows were supplied.')
    if frame.empty:raise ValueError('No test predictions were supplied.')
    for dataset, part in frame.groupby('dataset', sort=False):
        reference = None
        definitions = part['target_definition'].dropna().unique() if 'target_definition' in part else []
        if len(definitions) > 1:
            raise ValueError('A dataset table mixes target definitions.')
        for _, block in part.groupby(['model','seed'], sort=False):
            labels = block.set_index('row_id')['y_true'].sort_index()
            if reference is None:
                reference = labels
            elif not reference.equals(labels):
                raise ValueError('All models and seeds must share test row identities and labels.')
    records=[]
    for (dataset,model,seed),block in frame.groupby(['dataset','model','seed'],sort=False):
        if block['threshold'].nunique()!=1:raise ValueError('Every run must use one fixed threshold.')
        threshold=float(block.threshold.iloc[0])
        implied=(block.score.to_numpy()>=threshold).astype(int)
        if 'prediction' in block and not np.array_equal(implied,block.prediction.to_numpy()):
            raise ValueError('Stored decisions disagree with scores and thresholds.')
        metrics=classification_metrics(block.y_true.to_numpy(),block.score.to_numpy(),threshold)
        records.append({'dataset':dataset,'model':model,'seed':int(seed),**metrics})
    runs=pd.DataFrame(records);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    runs.to_csv(output/'metrics_by_seed.csv',index=False,float_format='%.17g')
    metrics=['accuracy','precision','recall','f1','roc_auc','pr_auc','mcc','gmean']
    summary=runs.groupby(['dataset','model'])[metrics].agg(['mean','std']).fillna(0.0)
    summary.to_csv(output/'metrics_summary.csv',float_format='%.17g')
    for dataset,block in runs.groupby('dataset'):
        rows=[]
        for model,m in block.groupby('model',sort=False):
            row={'Method':str(model).replace('_', chr(92)+'_')}
            for metric in metrics:
                sd=float(m[metric].std(ddof=1)) if len(m)>1 else 0.0
                row['AP' if metric=='pr_auc' else metric]=f'{m[metric].mean():.3f} $\\pm$ {sd:.3f}'
            rows.append(row)
        pd.DataFrame(rows).to_latex(output/f'{dataset}_metrics.tex',index=False,escape=False)
    print(f'Tables calculated from saved predictions in {output}.')

if __name__=='__main__':main()
