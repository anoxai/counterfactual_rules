# Counterfactual Rules (CR)

Temporary repository for the paper: **Rethinking Counterfactual Explanations as Local and Regional Counterfactual Policies**.

*Counterfactual Rules* (CR) is a python package that gives Counterfactual Explanations as Rules. For any individual or sub-population, it gives
 the simplest policies (rules) that change the decision with high probability.
 
## Requirements
Python >= 3.7 

**OSX**: CR uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)

**Windows**: Install MinGW (a Windows distribution of gcc) or Microsoft’s Visual C

Install the required packages:

```
$ pip install -r requirements.txt
```

## Installation

Clone the repo and run the following command in the main directory
```
$ python setup.py install
```

## Counterfactual Rules (CR)
The CR give Counterfactual Explanations for any data (**X**, **Y**) or model (**X**, **f(X)**) using the following 
explanation methods:

* **Divergent Explanations**
* **Local and Global Counterfactual Rules**

See the paper for more details.

**I. First, we need to fit our explainer (ACXplainers) to input-output of the data **(X, Y)** or model
**(X, f(X))** if we want to explain the data or the model respectively.**

```python
from acv_explainers import ACXplainer

# It has the same params as a Random Forest, and it should be tuned to maximize the performance.  
acv_xplainer = ACXplainer(classifier=True, n_estimators=50, max_depth=5)
acv_xplainer.fit(x_train, y_train)

roc = roc_auc_score(acv_xplainer.predict(X_test), y_test)
```

**II. Then, we can use the temporary Class "RunExperiments" to compute
the different Counterfactual Rules. Note that it is a demo version of the package and currently it only works on classification**.
See the file [experiments_paper/demo_run_counterfactual_rules_regression_house.py]() for a demo on regression problem.

```python 
results = RunExperiments(acv_xplainer, x_train, x_test, y_train, y_test, columns_name) # Initialize the demo 

results.run_local_divergent_set(x, y) # Compute the Divergent Explanations.

results.run_local_counterfactual_rules(x, y, acc_level=0.8, pi_level=0.8) # Compute the Local Counterfactual Rules

results.run_sampling_local_counterfactuals(x, y, batch=1000, max_iter=1000, temp=0.5) # Sample CE using the Local Counterfactual Rules

print('Local Accuracy = {} -- Local Coverage = {}'.format(results.accuracy_local, results.coverage_local))

results.run_sufficient_rules(x_rules, y_rules, pi_level=0.9) # Compute the Sufficient Rules that are used as init for the Regional Counterfactual Rules

results.run_regional_divergent_set(stop=True, pi_level=0.8) # Compute the Regional Divergent Explanations

results.run_regional_counterfactual_rules(acc_level=0.8, pi_level=0.8) # Compute the Regional Counterfactual Rules

results.run_sampling_regional_counterfactuals_alltests(max_obs=x_test.shape[0], batch=1000, max_iter=1000, temp=0.5) # Sample CE using the Regional Counterfactual Rules

results.show_global_counterfactuals() # Show the Local Counterfactual Rules
results.show_local_counterfactuals(x, y) # Show the Regional Counterfactual Rules
```

## Notebooks

You can find the experiments of the paper in the directory "experiment_paper".
