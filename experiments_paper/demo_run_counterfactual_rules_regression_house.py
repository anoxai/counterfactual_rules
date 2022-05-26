from acv_explainers import ACXplainer
from acv_explainers.utils import *
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error
from sklearn.datasets import load_digits, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

dataset_name = 'californiaHouse'
model = 'RF'
seed = 10

X = pd.read_csv('data/housing.csv')
maxval2 = X['median_house_value'].max()  # get the maximum value
X = X[X['median_house_value'] != maxval2]
X['diag_coord'] = (X['longitude'] + X['latitude'])  # 'diagonal coordinate', works for this coord
X['bedperroom'] = X['total_bedrooms'] / X['total_rooms']  # feature w/ bedrooms/room ratio
X = X.dropna()
y = X['median_house_value']
X.drop(['median_house_value', 'ocean_proximity'], inplace=True, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

x_train = X_train.values
x_test = X_test.values
y_train = y_train.values
y_test = y_test.values

columns_name = list(X.columns)

### Train Explainer (ACXplainer)
ac_explainer = ACXplainer(classifier=False, n_estimators=20, max_depth=10)
ac_explainer.fit(x_train, y_train)

print('# Trained ACXplainer -- score = {}'.format(mean_absolute_error(y_test, ac_explainer.predict(x_test))))

y_train = ac_explainer.predict(x_train)
y_test = ac_explainer.predict(x_test)

size = 2000
list_id = [i for i in range(x_test.shape[0]) if y_test[i] <= 150000] # get all the observations that has price <= 15K
x, y = x_test[list_id][:size], y_test[list_id][:size]
x_rules, y_rules = x_train[:10], y_train[:10]

results = RunExperiments(ac_explainer, x_train, x_test, y_train, y_test, columns_name)  # initialize the demo

down = np.array(x.shape[0] * [250000.])  # it corresponds to the lower bound of the target region for each x
up = np.array(x.shape[0] * [400000.])  # it corresponds to the upper bound of the target region for each x

results.ddp_importance_local, results.ddp_index_local, results.size_local, results.ddp_local = results.acv_explainer.importance_cdp_rf(
    x, y, down, up, x_train, y_train, t=0,
    stop=False, pi_level=0.2)  # compute the Local Divergent Explanations

results.S_star_local, results.S_bar_set_local = acv_explainers.utils.get_active_null_coalition_list(
    results.ddp_index_local, results.size_local)

results.ddp_local, results.w_local = results.acv_explainer.compute_cdp_weights(x, y, down, up, x_train, y_train,
                                                                               S=results.S_bar_set_local) # Compute the weights of each observations


def return_xy_cnt_reg(x, y_target, down, up, S_bar_set, x_train, y_train, w):
    """
    return the observations with y in y_target that fall in the projected leaf of x
    when we condition given S=S_bar of x.
    """

    x_train_cnt = []
    y_train_cnt = []
    for i, wi in enumerate(w):
        if wi != 0 and down <= y_train[i] <= up:
            x_train_cnt.append(x_train[i].copy())
            y_train_cnt.append(y_train[i].copy())

    x_train_cnt = np.array(x_train_cnt)
    #     x_train_cnt[:, S_bar_set] = x[S_bar_set]
    y_train_cnt = np.array(y_train_cnt)

    return x_train_cnt, y_train_cnt


def return_leaf_cnt_reg(ac_explainer, S_star, x_train_cnt, y_train_cnt, down, up, x_train, y_train, pi):
    """
    return the original leaves of the observations with y=y_target that fall in the projected leaf
    when we condition given S=S_bar of x.
    """
    size = x_train_cnt.shape[0]
    sdp, rules = ac_explainer.compute_cdp_rule(x_train_cnt, y_train_cnt, np.array(size * [down]), np.array(size * [up]),
                                               x_train, y_train,
                                               size * [list(range(x_train.shape[1]))]
                                               )
    #     print(sdp)
    if np.sum(sdp >= pi) != 0:
        rules_unique = np.unique(rules[sdp >= pi], axis=0)
    else:
        rules_unique = np.expand_dims(rules[np.argmax(sdp)], axis=0)

    r_buf = rules_unique.copy()
    for i in range(rules_unique.shape[0]):
        list_ric = [r.copy() for r in r_buf if not np.allclose(r, rules_unique[i])]
        find_union(rules_unique[i], list_ric, S=S_star)

    return rules_unique


def remove_in(ra):
    """
    remove A if A subset of B in the list of compatible leaves
    """
    for i in range(ra.shape[0]):
        for j in range(ra.shape[0]):
            if i != j and np.prod([(ra[i, s, 1] <= ra[j, s, 1]) * (ra[i, s, 0] >= ra[j, s, 0])
                                   for s in range(ra.shape[1])], axis=0).astype(bool):
                ra[i] = ra[j]
    return np.unique(ra, axis=0)


def get_compatible_leaf_reg(acvtree, x, y_target, down, up, S_star, S_bar_set, w, x_train, y_train, pi, acc_level):
    """
    Compute the compatible leaves and order given their accuracy
    """
    x_train_cnt, y_train_cnt = return_xy_cnt_reg(x, y_target, down, up, S_bar_set, x_train, y_train, w)

    compatible_leaves = return_leaf_cnt_reg(acvtree, S_star, x_train_cnt, y_train_cnt, down, up, x_train, y_train, pi)
    compatible_leaves = np.unique(compatible_leaves, axis=0)
    compatible_leaves = remove_in(compatible_leaves)
    #     compatible_leaves = np.rint(compatible_leaves)
    compatible_leaves = np.round(compatible_leaves, 2)

    partition_leaf = compatible_leaves.copy()
    d = partition_leaf.shape[1]
    nb_leaf = partition_leaf.shape[0]
    leaves_acc = []
    suf_leaf = []

    for i in range(nb_leaf):
        x_in = np.prod([(x_train[:, s] <= partition_leaf[i, s, 1]) * (x_train[:, s] > partition_leaf[i, s, 0])
                        for s in range(d)], axis=0).astype(bool)

        y_in = y_train[x_in]
        acc = np.mean((down <= y_in) * (y_in <= up))

        leaves_acc.append(acc)

        if acc >= acc_level:
            suf_leaf.append(partition_leaf[i])

    best_id = np.argmax(leaves_acc)
    return suf_leaf, partition_leaf, leaves_acc, partition_leaf[best_id], leaves_acc[best_id]


def return_counterfactuals_reg(ac_explainer, suf_leaf, S_star, S_bar_set, x, y, down, up, x_train, y_train, pi_level):
    """
    Compute the SDP of each C_S and return the ones that has sdp >= pi_level
    """
    counterfactuals = []
    counterfactuals_sdp = []
    counterfactuals_w = []

    for leaf in suf_leaf:

        cond = np.ones(shape=(1, x_train.shape[1], 2))
        cond[:, :, 0] = -1e+10
        cond[:, :, 1] = 1e+10

        for s in S_bar_set:
            cond[:, s, 0] = x[:, s]
            cond[:, s, 1] = x[:, s]

        cond[:, S_star] = leaf[S_star]

        size = x.shape[0]
        sdp, w = ac_explainer.compute_cdp_cond_weights(x, y, np.array(size * [down]), np.array(size * [up]), x_train,
                                                       y_train, S=[S_bar_set], cond=cond,
                                                       pi_level=pi_level)
        #         print(sdp)
        if sdp >= pi_level:
            counterfactuals.append(cond)
            counterfactuals_sdp.append(sdp)
            counterfactuals_w.append(w)

    return np.unique(counterfactuals, axis=0), np.unique(counterfactuals_sdp, axis=0), \
           np.unique(counterfactuals_w, axis=0)


def return_global_counterfactuals_reg(ac_explainer, data, y_data, down, up, s_star, n_star, x_train, y_train, w,
                                      acc_level, pi_level):
    """
    stack all to compute the C_S for each observation
    """
    N = data.shape[0]
    suf_leaves = []
    counterfactuals_samples = []
    counterfactuals_samples_sdp = []
    counterfactuals_samples_w = []

    for i in tqdm(range(N)):
        suf_leaf, _, _, _, _ = get_compatible_leaf_reg(ac_explainer, data[i], 1 - y_data[i], down[i], up[i], s_star[i],
                                                       n_star[i], w[i],
                                                       x_train, y_train, pi=pi_level, acc_level=acc_level)
        suf_leaves.append(suf_leaf)
        #         print(suf_leaf)
        #         print(np.unique(suf_leaf, axis=0).shape, len(suf_leaf))
        counterfactuals, counterfactuals_sdp, w_cond = \
            return_counterfactuals_reg(ac_explainer, suf_leaf, s_star[i], n_star[i], data[i].reshape(1, -1),
                                       y_data[i].reshape(1, -1), down[i], up[i], x_train, y_train, pi_level)

        counterfactuals_samples.append(counterfactuals)
        counterfactuals_samples_sdp.append(counterfactuals_sdp)
        counterfactuals_samples_w.append(w_cond)
    #     print(counterfactuals_samples)
    return counterfactuals_samples, counterfactuals_samples_sdp, counterfactuals_samples_w


acc_level = 0.7
pi_level = 0.7

results.counterfactuals_samples_local, results.counterfactuals_samples_sdp_local, \
results.counterfactuals_samples_w_local = return_global_counterfactuals_reg(results.acv_explainer, x, y, down, up,
                                                                            results.S_star_local,
                                                                            results.S_bar_set_local,
                                                                            x_train, y_train, results.w_local,
                                                                            acc_level=acc_level,
                                                                            pi_level=acc_level) # compute the local counterfactual rules


def run_sampling_local_counterfactuals(results, x, y, down, up, batch=10000, max_iter=1000, temp=0.5):
    print('### Sampling using the local counterfactual rules of (x, y)')

    results.isolation = IsolationForest()
    results.isolation.fit(results.x_train)
    outlier_score = lambda x: results.isolation.decision_function(x)

    results.dist_local = []
    results.score_local = []
    results.errs_local = []
    results.errs_local_original = []
    for i in tqdm(range(x.shape[0])):
        if len(results.counterfactuals_samples_local[i]) != 0:
            #             print(np.max(results.counterfactuals_samples_sdp_local[i]))
            a, sco = simulated_annealing(outlier_score, x[i], results.S_star_local[i], results.x_train,
                                         results.counterfactuals_samples_local[i][
                                             np.argmax(results.counterfactuals_samples_sdp_local[i])][0],
                                         batch, max_iter, temp)

            results.dist_local.append(np.squeeze(a))
            results.score_local.append(sco)
            #             print(results.acv_explainer.predict(down[i]<=results.dist_local[-1].reshape(1, -1)))
            results.errs_local.append(
                down[i] <= results.acv_explainer.predict(results.dist_local[-1].reshape(1, -1)) <= up[i])
            if results.model != None:
                results.errs_local_original.append(
                    results.model.predict(results.dist_local[-1].reshape(1, -1)) != results.model.predict(
                        x[i].reshape(1, -1)))

    results.accuracy_local = np.mean(results.errs_local)
    results.accuracy_local_original = np.mean(results.errs_local_original)
    results.coverage_local = len(results.errs_local) / x.shape[0]


run_sampling_local_counterfactuals(results, x, y, down, up) # sample using the local counterfactual rules

results.show_global_counterfactuals()  # Show the Local Counterfactual Rules
results.show_local_counterfactuals(x, y)  # Show the Regional Counterfactual Rules

save_model(results, name='{}CR_results'.format(dataset_name))
