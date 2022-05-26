from utils import DatasetHelper, submodular_picking, DATASETS_NAME
from acv_explainers.utils import *
from ares import AReS
from cet import CounterfactualExplanationTree


seed = 2022

GAMMA = 1.0
dataset = 'n'
dataset_name = DATASETS_NAME[dataset]
model= 'X'

np.random.seed(0)
LAMBDA = 0.01
GAMMA = 1.0

D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
X_tr, X_ts, y_tr, y_ts = D.train_test_split()

from sklearn.ensemble import IsolationForest

isolation = IsolationForest()
isolation.fit(X_tr)

y_tr = 1 - y_tr
y_ts = 1 - y_ts

mdl = LGBMClassifier(n_estimators=50, num_leaves=8)
mdl.fit(X_tr, y_tr)

X = X_tr[mdl.predict(X_tr)==1]
X_vl = X_ts[mdl.predict(X_ts)==1]

print('## Actionable Recourse Summary')
ares = AReS(mdl, X_tr, max_rule=8, max_rule_length=8, discretization_bins=10, minimum_support=0.05, print_objective=False,
            feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories,
            feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
ares = ares.fit(X, max_change_num=8, cost_type='MPS', lambda_acc=1.0, lambda_cov=1.0, lambda_cst=1.0)
print('* Parameters:')
print('\t* lambda_acc: {}'.format(ares.lambda_acc)); print('\t* lambda_cov: {}'.format(ares.lambda_cov)); print('\t* lambda_cst: {}'.format(ares.lambda_cst));
print('\t* minimum support: {}'.format(ares.rule_miner_.minsup_)); print('\t* discretization bins: {}'.format(ares.rule_miner_.fd_.bins)); print('\t* pre-processing time[s]: {}'.format(ares.preprocess_time_));
print('\t* max rule: {}'.format(ares.max_rule_)); print('\t* max rule length: {}'.format(ares.max_rule_length_)); print('\t* Time[s]:', ares.time_);
print('\t* uncover test: {}'.format(ares.uncover(X_vl))); print('\t* conflict: {}'.format(ares.conflict(X_vl))); print();
print('### Learned AReS')
print(ares.to_markdown())

x_ares = ares.predict(X_vl) + X_vl
y_ares = mdl.predict(x_ares)

print('Negative accuracy of AReS = {}'.format(np.mean(y_ares != mdl.predict(X_vl))))
print('Negative Inlier AReS = {}'.format(np.mean(isolation.predict(x_ares) == 1)))
print('Sparsity', np.mean(np.sum(x_ares-X_vl!=0, axis=1)))

# Counterfactual TREE

### 1- CET All

print('## Counterfactual Explanation Tree')
cet = CounterfactualExplanationTree(mdl, X_tr, y_tr, max_iteration=500, lime_approximation=(model!='L'),
                                    feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories,
                                    feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)

cet = cet.fit(X, max_change_num=8, cost_type='MPS', C=LAMBDA, gamma=GAMMA, max_leaf_size=-1, time_limit=180, verbose=True)
print('* Parameters:'); print('\t* lambda: {}'.format(cet.lambda_)); print('\t* gamma: {}'.format(cet.gamma_)); print('\t* max_iteration: {}'.format(cet.max_iteration_));
print('\t* leaf size bound:', cet.leaf_size_bound_); print('\t* leaf size:', cet.n_leaves_); print('\t* LIME approximation:', cet.lime_approximation_); print('\t* Time[s]:', cet.time_); print();
print('### Learned CET')
cet.print_tree()

x_cet = cet.predict(X_vl) + X_vl
y_cet = mdl.predict(x_cet)

print('Negative accuracy of CET = {}'.format(np.mean(y_cet != mdl.predict(X_vl))))
print('Negative Oulier CET = {}'.format(np.mean(isolation.predict(x_cet) == -1)))
print('Sparsity', np.mean(np.sum(x_cet-X_vl!=0, axis=1)))
