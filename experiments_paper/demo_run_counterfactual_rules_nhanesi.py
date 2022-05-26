from acv_explainers import ACXplainer
from acv_explainers.utils import *
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from utils import MyTabNetClassifier
from utils import DatasetHelper, DATASETS_NAME
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

seed = 2022
dataset = 'n'
dataset_name = DATASETS_NAME[dataset]
model= 'X'
np.random.seed(0)

if(model=='L'):
    print('* Classifier: LogisticRegression')
    mdl = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    print('\t* C: {}'.format(mdl.C)); print('\t* penalty: {}'.format(mdl.penalty));
elif(model=='X'):
    print('* Classifier: LightGBM')
    mdl = LGBMClassifier(n_estimators=50, num_leaves=8)
    print('\t* n_estimators: {}'.format(mdl.n_estimators)); print('\t* num_leaves: {}'.format(mdl.num_leaves));
elif(model=='T'):
    print('* Classifier: TabNet')
    mdl = MyTabNetClassifier(D.feature_types, verbose=0)


D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
X_tr, X_ts, y_tr, y_ts = D.train_test_split()

from sklearn.ensemble import IsolationForest

isolation = IsolationForest()
isolation.fit(X_tr)

mdl = mdl.fit(X_tr, y_tr, X_vl=X_ts, y_vl=y_ts) if model=='T' else mdl.fit(X_tr, y_tr)
X = X_tr[mdl.predict(X_tr)==1]; X_vl = X_ts[mdl.predict(X_ts)==1];
print('\t* train score: ', mdl.score(X_tr, y_tr)); print('\t* train denied: ', X.shape[0]);
print('\t* test score: ', mdl.score(X_ts, y_ts)); print('\t* test denied: ', X_vl.shape[0]); print();

x_train = X_tr.copy()
x_test = X_ts.copy()
y_train = mdl.predict(X_tr)
y_test = mdl.predict(X_ts)

### Train Explainer (ACXplainer)
ac_explainer = ACXplainer(classifier=True, n_estimators=20, max_depth=12)
ac_explainer.fit(x_train, y_train)

print('# Trained ACXplainer -- score = {}'.format(accuracy_score(y_test, ac_explainer.predict(x_test))))

# idx = 0
# size = idx + 500
x, y = x_test[:500], y_test[:500]
x_rules, y_rules = x_train[:1000], y_train[:1000]

columns_name = D.feature_names

results = RunExperiments(ac_explainer, x_train, x_test, y_train, y_test, columns_name, model=mdl)

results.run_local_divergent_set(x, y)

results.run_local_counterfactual_rules(x, y, acc_level=0.9, pi_level=0.9)

results.run_local_counterfactual_rules(x, y, acc_level=0.9, pi_level=0.9)

results.run_sampling_local_counterfactuals(x, y, batch=1000, max_iter=1000, temp=0.5)

print('Local Accuracy = {} -- Local Coverage = {}'.format(results.accuracy_local, results.coverage_local))

results.run_sufficient_rules(x_rules, y_rules, pi_level=0.9)

results.run_regional_divergent_set(stop=True, pi_level=0.9)

results.run_regional_counterfactual_rules(acc_level=0.9, pi_level=0.9)

results.run_sampling_regional_counterfactuals_alltests(max_obs=x_test.shape[0],batch=1000, max_iter=1000, temp=0.5)

print('Regional Accuracy = {} -- Regional Coverage = {}'.format(results.accuracy_regional, results.coverage_regional))
save_model(results, name='{}CR_results'.format(D.dataset_fullname))



if np.mean(mdl.predict(results.x_test) == results.y_test):
    print('CONSISTENT')
else:
    raise ValueError


x = []
for i, c in enumerate(results.counterfactuals_samples_local):
    if len(c) !=0:
        x.append(results.x_test[i])

x = np.array(x)
ce = np.array(results.dist_local)
ce_r = np.array(results.dist_regional)

print('all acc', np.mean(mdl.predict(x_test) != mdl.predict(ce_r)))

x_pos = x[mdl.predict(x) == 1]
ce_pos = ce[mdl.predict(x) == 1]

print('LOCAL positive accuracy', np.mean(mdl.predict(x_pos) != mdl.predict(ce_pos)))

print('LOCAL positive sparsity', np.mean(np.sum(x_pos-ce_pos!=0, axis=1)))

inlier_pos = np.mean(results.isolation.predict(ce_pos) == 1)
print('LOCAL positive inlier', inlier_pos)



x_neg = x[mdl.predict(x) == 0]
ce_neg = ce[mdl.predict(x) == 0]

print('LOCAL negative accuracy', np.mean(mdl.predict(x_neg) != mdl.predict(ce_neg)))

print('LOCAL negative sparsity', np.mean(np.sum(x_neg-ce_neg!=0, axis=1)))

inlier_neg = np.mean(results.isolation.predict(ce_neg) == 1)
print('LOCAL negative inlier', inlier_neg)



inlier_pos = np.mean(results.isolation.predict(ce_pos_r) == 1)
print('REGIONAL positive inlier', inlier_pos)



print('Regional Accuracy = {}'.format(results.accuracy_regional))

print('Local Coverage = {} -- Global Coverage {}'.format(results.coverage_local,
                                                        results.coverage_regional))

save_model(results, name='{}CR_results'.format(D.dataset_fullname))