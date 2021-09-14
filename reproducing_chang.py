# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
FIGWIDTH = 8

import os
NUM_SHUFFLES = int(os.environ.get("NUM_SHUFFLES") or 200)

# %%
# %load_ext autoreload
# %autoreload 2

import itertools

import pandas as pd
import numpy as np
import seaborn as sns
import diffprivlib.models as dp

from tqdm import autonotebook as tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from scipy import stats

from model_zoo import model_zoo, lr_setup, renaming_dict
from mia import run_threshold_estimator, run_shadow_model_attack
from utils import infer_from

import plot_params


# %%
# Control model
class ThreshClassifier:
    def __init__(self, threshold=0.):
        self.threshold = threshold

    # The model is data-independent
    def fit(self, *args, **kwargs):
        pass

    def predict_proba(self, xs, *args, **kwargs):
        if isinstance(xs, pd.DataFrame):
            xs = xs.values
        p = xs[:, 0] > self.threshold
        p = np.expand_dims(p, 1)
        return np.hstack([1-p, p])

ThreshClassifier().predict_proba(np.array([[0, 1], [1, 0]]))
model_zoo["threshold"] = lambda: ThreshClassifier(1)

# %%
total_size = 2500
size0 = 0.2
size1 = 0.8
gen = np.random.RandomState(seed=1)
minus0 = gen.multivariate_normal([0, -1], [[7, 1], [1, 7]],
                                 size=int(0.1 * size0 * total_size))
plus0 = gen.multivariate_normal([1, 2], [[5, 2], [2, 5]],
                                 size=int(0.9 * size0 * total_size))
minus1 = gen.multivariate_normal([-5, 0], [[5, 1], [1, 5]],
                                 size=int(0.5 * size1 * total_size))
plus1 = gen.multivariate_normal([2, 3], [[10, 1], [1, 4]],
                                 size=int(0.5 * size1 * total_size))
len(minus0), len(plus0), len(minus1), len(plus1)

data = pd.concat([
    pd.DataFrame(minus0).assign(z=0, y=0),
    pd.DataFrame(plus0).assign(z=0, y=1),
    pd.DataFrame(minus1).assign(z=1, y=0),
    pd.DataFrame(plus1).assign(z=1, y=1),
], axis=0, ignore_index=True)

data.head()

# %%
data.groupby(["z", "y"]).count()

# %%
sns.displot(data, x=0, hue="z", col="y")


# %%
def get_subgroup_vulns(clf, data_train, data_test,
                       sensitive_features=False, ys=None, zs=None,
                       ignore_y=False, visualize=False,
                       method="average_loss_threshold"):
    if ys is None: ys = [0, 1]
    if zs is None: zs = [0, 1]

    result = pd.DataFrame()
    for y, z in itertools.product(ys, zs):
        group_train = data_train.query(f"y == {y} and z == {z}")
        group_test = data_test.query(f"y == {y} and z == {z}")

        preds_train = infer_from(clf, group_train[[0, 1]])
        preds_test = infer_from(clf, group_test[[0, 1]])

        assert "threshold" in method

        vulns = run_threshold_estimator(
            group_train.y, preds_train, group_test.y, preds_test,
            microdata=False,
            method=method,

            # With False, all threshold estimators will have additional bias due
            # to unequal representations of in/out challenge examples in the subgroups,
            # which should be accounted by a change of baseline in the advantage computation
            # from 2 * (attack_acc - 0.5) to 2 * (attack_acc - skewed_group_baseline);
            # alternatively, sampling of train/test datasets should be stratified by subgroup.
            # Therefore, True is the right usage if no corrections are performed, and we use
            # True for these experiments; however, the arxiv preprint does not include enough
            # implementation details to tell how exactly the original paper implemented this.
            enforce_uniform_prior=True,
        )

        result = result.append(
            pd.DataFrame(dict(vuln=[vulns], y=y, z=z)),
            ignore_index=True
        )
    return result

methods = ["best_loss_threshold", "average_loss_threshold"]
sim_results = pd.DataFrame()
for rep in tqdm.trange(NUM_SHUFFLES):
    data_train, data_test = train_test_split(
        data, test_size=0.5, random_state=rep)
    X_train = data_train[[0, 1]].values
    y_train = data_train.y.values

    control_model = ThreshClassifier(0)
    control_model.fit(X_train, y_train)
#     control_model1 = ThreshClassifier(2)
#     control_model1.fit(X_train, y_train)
#     control_model2 = ThreshClassifier(10)
#     control_model2.fit(X_train, y_train)
    normal_model = MLPClassifier(hidden_layer_sizes=[8, 8, 8]).fit(X_train, y_train)
    fair_model = model_zoo["lr_eo_expgrad"]()
    fair_model.fit(X_train, y_train, sensitive_features=data_train.z)

    for method in methods:
        vulns_data = pd.concat([
            get_subgroup_vulns(control_model, data_train, data_test,
                               method=method) \
                .assign(model="control", method=method),
#             get_subgroup_vulns(control_model1, data_train, data_test,
#                                method=method) \
#                 .assign(model="control1", method=method),
#             get_subgroup_vulns(control_model2, data_train, data_test,
#                                method=method) \
#                 .assign(model="control2", method=method),
            get_subgroup_vulns(normal_model, data_train, data_test,
                               method=method) \
                .assign(model="nn", method=method),
            get_subgroup_vulns(fair_model, data_train, data_test,
                               sensitive_features=True,
                               method=method) \
                .assign(model="fair", method=method),
        ], axis=0, ignore_index=True).assign(rep=rep)
        sim_results = sim_results.append(vulns_data, ignore_index=True)

sim_results.head()

# %%
sim_results.groupby(["model", "method", "z", "y", "rep"]).vuln.mean().reset_index() \
           .groupby(["model", "method", "z", "y"]).agg(dict(vuln="mean"))

# %%
sim_results.groupby(["model", "method", "z", "y"]).vuln.mean()

# %%
sim_results["subgroup"] = list(f"{z}-{y}" for z, y in zip(sim_results["z"], sim_results["y"]))

# %% [markdown]
# Compute ANOVA F-test p-values to check if there is significant disparity between subgroup vulnerabilities

# %%
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

for model, method in itertools.product(sim_results.model.unique(), methods):
    df = sim_results.query(f"model == '{model}' and method == '{method}'")
    anova = AnovaRM(
        data=df,
        depvar="vuln",
        subject="rep",
        within=["subgroup"],
        aggregate_func=np.mean
    )
    res = anova.fit()
    f, p = (
        res.anova_table.loc["subgroup", "F Value"],
        res.anova_table.loc["subgroup", "Pr > F"]
    )

    print(f"{model=} {method=}")
    print(f"{p=} {f=}\n")
    sim_results.loc[df.index, "p"] = p
    sim_results.loc[df.index, "F"] = f

# %%
plot_df = sim_results.copy()
plot_df = plot_df.replace({
    "average_loss_threshold": "Avg. loss threshold",
    "best_loss_threshold": "Opt. loss threshold",
}).rename(columns={
    "subgroup": "Subgroup",
    "vuln": "Estimate of subgroup vuln.",
    "method": "Method",
})

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(
    data=plot_df.query("model == 'control'"),
    estimator=lambda vulns: (2 * vulns.mean() - 1) * 100,
    x="Subgroup", y="Estimate of subgroup vuln.", hue="Method",
    order=["0-0", "0-1", "1-0", "1-1"],
    ax=ax
)

fig.set_tight_layout(tight=True)

# plt.savefig("images/plot_estimation_bias_chang.pdf")

# %% [markdown]
# The following is the evidence that the advantage from the optimal threshold attack is not only due to small-sample bias, but also due to legitimate advantage, but it is not possible to tell which part of the estimate is bias (would have been there if the target model was independent of the data) and which one is not, as we cannot just, e.g., subtract the vulnerability of the control model.

# %%
plot_df = sim_results.copy()
plot_df
plot_df = plot_df.replace({
    "average_loss_threshold": "Avg. loss threshold",
    "best_loss_threshold": "Opt. loss threshold",
}).rename(columns={
    "model": "Model",
    "subgroup": "Subgroup",
    "vuln": "Est. of vuln.",
    "method": "Method",
})
sns.catplot(data=plot_df, x="Subgroup", y="Est. of vuln.",
            hue="Model", col="Method",
            estimator=lambda vulns: (2 * vulns.mean() - 1) * 100,
            kind="bar")
