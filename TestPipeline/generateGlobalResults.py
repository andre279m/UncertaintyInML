import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level = logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

for t in ['Undersampling']:
    logging.info("Starting training with " + t + " dataset")
    for m in ['PU','Normal','Weighted']:
        for clf in ["RandomForestClassifier","XGBClassifier"]:
            global_results = pd.DataFrame(columns=['PRECISION(MEDIAN)','PRECISION(IQR)','RECALL(MEDIAN)','RECALL(IQR)','WAF(MEDIAN)','WAF(IQR)','ACCURACY(MEDIAN)','ACCURACY(IQR)','ROC AUC(MEDIAN)','ROC AUC(IQR)'])
            for i in range(800, -1, -200):
                d = pd.read_csv('../Results/30_04Test/Folds/' + t + 'T' + str(i) + 'metrics_' + m + clf + '.csv',header=0)
                global_results.loc[i] = {'PRECISION(MEDIAN)':d['precision'].median(),'PRECISION(IQR)':d['precision'].quantile(0.75)-d['precision'].quantile(0.25),'RECALL(MEDIAN)':d['recall'].median(),'RECALL(IQR)':d['recall'].quantile(0.75)-d['recall'].quantile(0.25),'WAF(MEDIAN)':d['WAF'].median(),'WAF(IQR)':d['WAF'].quantile(0.75)-d['WAF'].quantile(0.25),'ACCURACY(MEDIAN)':d['accuracy'].median(),'ACCURACY(IQR)':d['accuracy'].quantile(0.75)-d['accuracy'].quantile(0.25),'ROC AUC(MEDIAN)':d['AUC'].median(),'ROC AUC(IQR)':d['AUC'].quantile(0.75)-d['AUC'].quantile(0.25)}
            global_results.to_csv('../Results/30_04Test/' + t + 'metrics_' + m + clf + '.csv')


# for Random testing
# for m in ['PU','Normal','Weighted']:
#     for clf in ["RandomForestClassifier","XGBClassifier"]:
#         global_results = pd.DataFrame(columns=['PRECISION(MEDIAN)','PRECISION(IQR)','RECALL(MEDIAN)','RECALL(IQR)','WAF(MEDIAN)','WAF(IQR)','ACCURACY(MEDIAN)','ACCURACY(IQR)','ROC AUC(MEDIAN)','ROC AUC(IQR)'])
#         d = pd.read_csv('Results/22_03Test/Random/metrics_' + m + clf + '.csv',header=0)
#         global_results.loc[0] = {'PRECISION(MEDIAN)':d['precision'].median(),'PRECISION(IQR)':d['precision'].quantile(0.75)-d['precision'].quantile(0.25),'RECALL(MEDIAN)':d['recall'].median(),'RECALL(IQR)':d['recall'].quantile(0.75)-d['recall'].quantile(0.25),'WAF(MEDIAN)':d['WAF'].median(),'WAF(IQR)':d['WAF'].quantile(0.75)-d['WAF'].quantile(0.25),'ACCURACY(MEDIAN)':d['accuracy'].median(),'ACCURACY(IQR)':d['accuracy'].quantile(0.75)-d['accuracy'].quantile(0.25),'ROC AUC(MEDIAN)':d['AUC'].median(),'ROC AUC(IQR)':d['AUC'].quantile(0.75)-d['AUC'].quantile(0.25)}
#         global_results.to_csv('Results/22_03Test/Random/Global/metrics_' + m + clf + '.csv')
