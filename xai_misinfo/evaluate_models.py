import os, json, glob
import pandas as pd

def aggregate_metrics(models_dir='models'):
    metrics = {}
    for path in glob.glob(os.path.join(models_dir,'metrics_*.json')):
        name = os.path.splitext(os.path.basename(path))[0].replace('metrics_','')
        with open(path) as f:
            metrics[name] = json.load(f)
    return metrics

if __name__=='__main__':
    mets = aggregate_metrics()
    print(json.dumps(mets, indent=2))
