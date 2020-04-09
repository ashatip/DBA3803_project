# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from statsmodels.formula.api import ols
from sklearn.linear_model import ElasticNetCV
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import re
# %%

df = pd.read_csv("data_static.csv")
# drop empty columns
df = df.dropna(axis=1, how='all')

# %%
# keep only certain columns because we don't want the model equation to be too big
# we choose 'Indicated Dividend Rate (Security)' to be our Y
newdf = df[['Indicated Dividend Rate (Security)',
        'Alpha (Security)', 'Beta (WS)',
       'Book Value Per Share - Current (Security)',
       'Cash Flow Per Share - Current (Security)',
         'Earnings Yield - Current (Security)',
        'Market Capitalization Current (U.S.$)',
        'Market Price - Current (Security)',
        'Price/Book Value Ratio - Current (Security)',
       'Price/Cash Flow Current (Security)',
       'Price/Earnings Ratio - Current (Security)',
        'Return On Equity - Per Share - Current']]
h = newdf
import re
h = h.rename(columns=lambda x: re.sub('\W', '_', x))

print('\nThe dataset has', len(h), 'rows and', h.shape[1], 'columns.')
# len(h) == h.shape[0]

print('\nColumns are:', list(h))
newdf2 = df[['Indicated Dividend Rate (Security)',
        'Alpha (Security)', 'Beta (WS)',
       'Book Value Per Share - Current (Security)',
       'Cash Flow Per Share - Current (Security)',
         'Earnings Yield - Current (Security)',
        'Market Capitalization Current (U.S.$)']]

newdf3 = df[['Indicated Dividend Rate (Security)',
            'Market Price - Current (Security)',
        'Price/Book Value Ratio - Current (Security)',
       'Price/Cash Flow Current (Security)',
       'Price/Earnings Ratio - Current (Security)',
        'Return On Equity - Per Share - Current']]

newdf2 = newdf2.rename(columns=lambda x: re.sub('\W', '_', x))
newdf3 = newdf3.rename(columns=lambda x: re.sub('\W', '_', x))

DF2 = newdf2
print("Null values", DF2.isnull().sum())
# %% [markdown]
# AHMET code
# %%


def add_powers(y_val, data, powers):
    """adding powers"""
    x_col = list(data)
    x_col.remove(y_val)
    for x_c in x_col:
        for power in powers:
            data[x_c+str(power)+"power"] = np.power(
                data[x_c].values, power)
            if power == 2:
                if all(nums >= 0 for nums in data[x_c].values):
                    data[x_c+str(power)+"root"] = np.sqrt(data[x_c])
            elif power == 3:
                data[x_c+str(power)+"root"] = np.cbrt(data[x_c])
    return data


def add_interactions(data, do_not_include):
    '''this function will add interaction variables to the dataframe'''
    cols = list(data)
    if do_not_include is not None:
        cols.remove(do_not_include)
    lens = len(cols)
    for fir in range(0, lens):
        for sec in range(fir, lens):
            name = cols[fir]+"_X_"+cols[sec]
            data[name] = data[cols[fir]] * data[cols[sec]]
    return data


# found this method online looks good
# https://planspace.org/20150423-forward_selection_with_statsmodels/
def forward_selection(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    AHMET: stops when no more features to be added or r2 is not improved
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


# https://datascience.stackexchange.com/questions/24405/
# how-to-do-stepwise-regression-using-sklearn/24447#24447
def stepwise_selection(data, y_val,
                       threshold_in=0.01,
                       threshold_out=0.05,
                       verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list()
    while True:
        x_col = list(data)
        x_col.remove(y_val)
        changed = False
        # forward step
        if len(included) == 0:
            excluded = x_col
        else:
            excluded = list(set(x_col) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model_eq = y_val + "~" + "+".join(included + [new_column])
            model = ols(model_eq, data).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(
                    best_feature, best_pval))
        # backward step
        model_eq = y_val + "~" + "+".join(included)
        model = ols(model_eq, data).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(
                    worst_feature, worst_pval))
        if not changed:
            break
    model_eq = y_val + "~" + "+".join(included)
    model = ols(model_eq, data).fit()
    return model


# %%
# training and testing using ELASTIC NET
# not sure if it will finish in time or not
DF2.dropna(inplace=True)
Y_VAL = DF2.columns[0]
DF2 = add_powers(Y_VAL, DF2, [2, 3])
DF2 = add_interactions(newdf2, newdf2.columns[0])
DF2_TRA = DF2.sample(frac=0.8, random_state=200)
# random state is a seed value
DF2_TES = DF2.drop(DF2_TRA.index)

# %%
# elastic fitting, took a bit too long
# ELA_CV = ElasticNetCV(cv=10, l1_ratio=[.1, .5, .7, .9,  1],
#                       alphas=[.1, .5, .7, .9,  1], max_iter=5000)
# FIT = ELA_CV.fit(DF2_TRA.drop(columns=[Y_VAL]), DF2_TRA[Y_VAL])
# FIT.get_params()
# %%
# %%

FOR = forward_selection(DF2_TRA, Y_VAL)
STEP = stepwise_selection(DF2_TRA, Y_VAL)
print("FORWARD")
print(FOR.summary2())
print("STEP")
print(STEP.summary2())


