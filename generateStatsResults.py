from scipy import stats
import pandas as pd
from pathlib import Path

def run_statiscal_test(results1, results2, test='Wilcoxon', side="less"):
    if test == 'Kruskal':
        p_value = stats.kruskal(results1, results2)[1]
    if test == 'Mood':
        p_value = stats.median_test(results1, results2)[1]
    if test == 'Wilcoxon':
        p_value = stats.wilcoxon(results1, results2, alternative=side)[1]
    return p_value

for i in range(800, -1, -200):
    accDF = pd.DataFrame(columns=['StaticvsGrowing','StaticvsUniform','GrowingvsUniform'])
    wafDF = pd.DataFrame(columns=['StaticvsGrowing','StaticvsUniform','GrowingvsUniform'])
    rocDF = pd.DataFrame(columns=['StaticvsGrowing','StaticvsUniform','GrowingvsUniform'])
    for t1 in ['Static','Growing']:
        for t2 in ['Uniform','Growing']:
            if t1 != t2:
                for m in ['Normal','Weighted','PU']:
                    for clf in ["RandomForestClassifier","XGBClassifier"]:
                        d1 = pd.read_csv('Results/7_03Test/Folds/' + t1 + 'T' + str(i) + 'metrics_' + m + clf + '.csv',header=0)
                        d2 = pd.read_csv('Results/7_03Test/Folds/' + t2 + 'T' + str(i) + 'metrics_' + m + clf + '.csv',header=0)
                        w = run_statiscal_test(d1['WAF'].values, d2['WAF'].values,test='Kruskal')
                        a = run_statiscal_test(d1['accuracy'].values, d2['accuracy'].values,test='Kruskal')
                        auc = run_statiscal_test(d1['AUC'].values, d2['AUC'].values,test='Kruskal')
                        accDF.loc[m + clf,t1 + 'vs' + t2] = a
                        wafDF.loc[m + clf,t1 + 'vs' + t2] = w
                        rocDF.loc[m + clf,t1 + 'vs' + t2] = auc
    Path('Results/7_03Test/StatisticalTests/').mkdir(parents=True, exist_ok=True)
    accDF.to_csv('Results/7_03Test/StatisticalTests/DBaccT' + str(i) + '.csv')
    wafDF.to_csv('Results/7_03Test/StatisticalTests/DBwafT' + str(i) + '.csv')
    rocDF.to_csv('Results/7_03Test/StatisticalTests/DBrocT' + str(i) + '.csv')

for i in range(800, -1, -200):
    accDF = pd.DataFrame()
    wafDF = pd.DataFrame()
    rocDF = pd.DataFrame()
    for m1 in ['Normal','Weighted']:
        for m2 in ['Weighted','PU']:
            if m1 != m2:
                for t in ['Static','Growing','Uniform']:
                    for clf in ["RandomForestClassifier","XGBClassifier"]:
                        d1 = pd.read_csv('Results/7_03Test/Folds/' + t + 'T' + str(i) + 'metrics_' + m1 + clf + '.csv',header=0)
                        d2 = pd.read_csv('Results/7_03Test/Folds/' + t + 'T' + str(i) + 'metrics_' + m2 + clf + '.csv',header=0)
                        w = run_statiscal_test(d1['WAF'].values, d2['WAF'].values,test='Kruskal')
                        a = run_statiscal_test(d1['accuracy'].values, d2['accuracy'].values,test='Kruskal')
                        auc = run_statiscal_test(d1['AUC'].values, d2['AUC'].values,test='Kruskal')
                        accDF.loc[t + clf,m1 + 'vs' + m2] = a
                        wafDF.loc[t + clf,m1 + 'vs' + m2] = w
                        rocDF.loc[t + clf,m1 + 'vs' + m2] = auc
    Path('Results/7_03Test/StatisticalTests/').mkdir(parents=True, exist_ok=True)
    accDF.to_csv('Results/7_03Test/StatisticalTests/MLaccT' + str(i) + '.csv')
    wafDF.to_csv('Results/7_03Test/StatisticalTests/MLwafT' + str(i) + '.csv')
    rocDF.to_csv('Results/7_03Test/StatisticalTests/MLrocT' + str(i) + '.csv')