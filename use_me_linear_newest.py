"""small file for some common linear methods"""
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
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
# %%

# %% [markdown]
# AHMET code
# %%

def view_df(df_v):
    """Function for viewing the columns of the data frame and deciding to keep
    or remove"""
    # remove na and duplicate rows
    df_v.dropna(axis=0, how="all", inplace=True)
    df_v.dropna(axis=1, how="all", inplace=True)
    # not sure but, I think duplicate data is useless
    df_v.drop_duplicates(inplace=True, ignore_index=True)
    df_v = df_v.loc[:, ~df_v.columns.duplicated()]
    print(df_v.shape)
    id_v = []
    keep = []
    remove = []
    location = []
    time = []
    for i in list(df_v):
        print("*****")
        print("COL: ", i)
        i_type = df_v[i].dtypes
        print("Type", i_type)
        print("% of NA", df_v[i].isna().sum()*100/len(df_v[i]))
        print("number of unique values: ", df_v[i].nunique())
        num_print = 5
        if num_print > df_v[i].nunique():
            num_print = df_v[i].nunique()
        print("some example values", df_v[i].unique()[:num_print])
        if np.issubdtype(df_v[i].dtype, np.number):
            plt.hist(df_v[i])
            plt.show()
        sub = input("Is the column a: \n l for location \n k for keep \n r fo"
                    + "remove \n i for id \n t for time")
        if sub[0] == "l":
            location.append(i)
        elif sub[0] == "k":
            keep.append(i)
        elif sub[0] == "r":
            remove.append(i)
        elif sub[0] == "i":
            id_v.append(i)
        elif sub[0] == "t":
            time.append(i)
        print("*****")
    return keep, id_v, time, location, remove


def add_powers(data, powers, do_not_include):
    """adding powers"""
    x_col = list(data)
    for rem in do_not_include:
        x_col.remove(rem)
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


# from class


#Regression_RandomForest_GradientBoosting_NeuralNetwork_SupportVector
def fitpred(model, modstr, x_train, x_test, y_train, y_test):
    """good for plotting most important values
    model is model for sklearn
    modstr is model name"""
    stm = time.time()
    print(f'\033[1m' + '\n\n' + modstr, 'Regression may take time...' + f'\033[0m') #in bold
    model.fit(x_train, y_train)
    try:
        im = model.feature_importances_
        nx = len(im)

        #plot in original order of columns
        pl.title("For '" + y_train.name + "', in " + str(nx) + " Variables' Original Order")

        #original x-order
        pl.plot(im)
        locs, labs = pl.xticks()
        pl.xticks(locs, [''] + [list(x_train)[i] for i in [int(x) for x in list(locs[1:])
                                                           if x <= len(list(x_train)) - 1]], rotation=90)

        pl.ylabel('Relative Importance')
        pl.show()

        #plot in the order of variable's importance
        pl.title("Ranked by Relative Importance for '" + y_train.name + "'")
        rs = sorted(im, reverse=True)
        i = min(9, nx)
        pl.plot(rs[:i])
        locs, labs = pl.xticks()
        xs = [z for _, z in sorted(zip(im, list(x_train)), reverse=True)]
        pl.xticks(locs[:i + 1], [''] + [xs[i] for i in [int(x) for x in list(locs[1:])
                                                        if x <= len(xs) - 1]][:i], rotation=90)
        pl.ylabel('Relative Importance')
        pl.show()
    except:
        print()

    #predict using test set

    predictmod = model.predict(x_test)
    print(modstr, 'validation  MAE =', mean_absolute_error(y_test, predictmod))
    print(modstr, 'validation RMSE =', np.sqrt(mean_squared_error(y_test, predictmod)))
    #r2_score(y_test, predictmod) is not reliable (may be < 0)
    r2 = pd.concat([y_test, pd.Series(predictmod, index=y_test.index)], 1).corr().iloc[0, 1] ** 2
    print(modstr, 'validation   R² =', r2)

    #plot y vs y-hat

    #plot pseudo regression line
    s, i = np.polyfit(predictmod, y_test, 1) #s=slope, i=intercept
    a, b = min(predictmod), max(predictmod)
    pl.plot([a, b], i + [s * a, s * b], 'red', linewidth=0.4) #increase linewidth for darker line
    del s, i, a, b

    #scatter y vs y-hat
    pl.scatter(predictmod, y_test, s=6, linewidths=0)
    xn = modstr + ' Prediction'
    yn = y_test.name
    pl.xlabel(xn)
    pl.ylabel(yn)
    pl.xticks(rotation=90)
    pl.title("Plot of '" + yn + "' vs '" + xn + "' for " + str(len(y_test)) + ' obs')
    pl.show()

    print('\n' + modstr, 'Regression for', len(x_train), 'rows and', x_train.shape[1], 'columns, and prediction for',
          len(x_test), 'rows, took', '%.2f' % ((time.time() - stm) / 60), 'mins.')

    return r2

