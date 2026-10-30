"""Publication-quality figures from generated metrics (simulation labels required)."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_plots(metrics_csv,output_dir):
    df=pd.read_csv(metrics_csv)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    for metric,label in [('mean_latency_ms','Modelled average latency (ms)'),('normalized_cost','Modelled normalized cost'),
                         ('sla_pct','Modelled SLA satisfaction (%)'),('throughput_qps','Modelled throughput (QPS)'),
                         ('data_transfer_mb','Modelled transfer volume (MB)')]:
        aggregated=df.groupby('method')[metric].agg(['mean','std']).sort_index()
        fig,ax=plt.subplots(figsize=(7,4.3))
        x=np.arange(len(aggregated))
        ax.bar(x,aggregated['mean'].fillna(0),yerr=aggregated['std'].fillna(0),capsize=3)
        ax.set_xticks(x,aggregated.index,rotation=20,ha='right')
        ax.set_ylabel(label)
        ax.set_xlabel('Execution strategy')
        ax.grid(axis='y',alpha=.2)
        fig.tight_layout()
        fig.savefig(out/f'{metric}.png',dpi=600)
        plt.close(fig)
    return len(aggregated)
