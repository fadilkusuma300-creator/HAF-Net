#!/usr/bin/env python3
"""Export dataset partitions, preprocessing bounds, and the deterministic Altman reference."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from hafnet.config import load_config
from hafnet.data import prepare_dataset
from hafnet.baselines import altman_risk_score
from hafnet.metrics import classification_metrics
from hafnet.provenance import data_contract
from hafnet.temporal import TemporalPairBank
from hafnet.utils import ensure_dir,write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',nargs='+',default=['configs/taiwan.yaml','configs/american.yaml'])
    parser.add_argument('--output',default='data/derived')
    args=parser.parse_args()
    out=ensure_dir(args.output)
    membership=[];scales=[];formula_rows=[];formula_predictions=[];summaries={}
    for file in args.configs:
        config=load_config(file);d=prepare_dataset(config)
        summary={'data_contract':data_contract(config),'n_features':len(d.feature_ids),'partitions':{}}
        for name in ['train','validation','test']:
            split=getattr(d,name)
            meta=split.metadata.copy();meta.insert(0,'dataset',d.name);meta['split']=name;meta['y_true']=split.y
            membership.append(meta)
            summary['partitions'][name]={'rows':len(split.y),'positive_rows':int(split.y.sum()),'negative_rows':int((split.y==0).sum())}
            if d.name=='american':
                raw=getattr(d,f'raw_{name}_frame')
                summary['partitions'][name]['firms']=int(split.metadata.company_name.nunique())
                summary['partitions'][name]['nonpositive_total_liabilities']=int((raw.X17<=0).sum())
                if name!='train':
                    risk=altman_risk_score(raw);threshold=float(np.nextafter(-2.675,np.inf))
                    metrics=classification_metrics(split.y,risk,threshold)
                    y=split.y;prediction=risk>=threshold
                    formula_rows.append({'dataset':d.name,'split':name,'formula':'original_public_firm_Z',
                        'target':'terminal_failed_proxy','z_cutoff':2.675,**metrics,
                        'tp':int(np.sum((y==1)&prediction)),'fp':int(np.sum((y==0)&prediction)),
                        'fn':int(np.sum((y==1)&~prediction)),'tn':int(np.sum((y==0)&~prediction))})
                    pred=meta.copy();pred['risk_score']=risk;pred['prediction']=prediction.astype(int)
                    formula_predictions.append(pred)
        if d.name=='american':
            summary['eligible_training_temporal_pairs']=len(TemporalPairBank(d.train.metadata,d.train.y).pairs)
        pre=d.preprocessor
        for i,f in enumerate(d.feature_ids):
            scales.append({'dataset':d.name,'feature_id':f,'feature_label':d.feature_labels[i],
                'training_q01':pre.lower_[i],'training_q99':pre.upper_[i],
                'training_clipped_mean':pre.mean_[i],'training_clipped_population_sd':pre.scale_[i],
                'standardized_lower_bound':(pre.lower_[i]-pre.mean_[i])/pre.scale_[i],
                'standardized_upper_bound':(pre.upper_[i]-pre.mean_[i])/pre.scale_[i]})
        summaries[d.name]=summary
    pd.concat(membership,ignore_index=True).to_csv(out/'partition_membership.csv',index=False)
    pd.DataFrame(scales).to_csv(out/'preprocessing_statistics.csv',index=False,float_format='%.17g')
    if formula_rows:
        pd.DataFrame(formula_rows).to_csv(out/'altman_formula_metrics.csv',index=False,float_format='%.17g')
        pd.concat(formula_predictions,ignore_index=True).to_csv(out/'altman_formula_predictions.csv',index=False,float_format='%.17g')
    write_json(out/'dataset_summary.json',summaries)
    print(f'Data summaries written to {out}. No learned model was fitted.')

if __name__=='__main__':main()
