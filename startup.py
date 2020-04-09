# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]

# This is Ahmets script for looking at the files
# %%
"""File analyst"""
import os
import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup as bs4
# %%
# I make the data local on my machine will change it soon
plt.rc_context({'axes.edgecolor': 'orange',
                'xtick.color': 'red', 'ytick.color': 'green',
                'figure.facecolor': 'white'})
# import matplotlib.ticker as ticker
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams['figure.figsize'] = 11, 5
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 1


DIR = os.getcwd()
# DIR = os.path.dirname(os.path.realpath(__file__))
DF1 = pd.read_csv(DIR + "/DATA/Financial_Info_SG_Firms_Combined.csv")
# %%
# remove all nans
DF1.dropna(axis=0, how="all", inplace=True)
DF1.dropna(axis=1, how="all", inplace=True)
list(DF1)
UNI = DF1.nunique(dropna=True)
# %%

# this is for creating the defintion table
DEF1 = pd.read_excel(DIR + "/DATA/Financial_Info_SG_Firms.xlsx",
                     sheet_name="DEFINITION_WORLDSCOPE_STATIC")
DEF1.set_index("Name", inplace=True)
DEF2 = pd.read_excel(DIR + "/DATA/Financial_Info_SG_Firms.xlsx",
                     sheet_name="DEFINITION_WORLDSCOPE_TS")
DEF2.set_index("Name", inplace=True)
print(DEF2.head())


print(list(DEF1))
print(list(DEF2))

DEF = DEF1.append(DEF2)
# look it up like this:
print(DEF.loc["ADR CUSIP 1"])

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


# %%
# KEEP, ID_V, TIME, LOCATION, REMOVE = view_df(DF1)

# %%
print(DEF.loc["Market Price - Current (Security)"])

# %%
# cleaning
DF1.columns = [re.sub("\s", "_", x) for x in list(DF1)]

# %% [markdown]
# building models to predict Industry type

# %%
Y_VAL = DF1[["Thomson_Reuters_Business_Classification_Code", "serial_number",
         "Worldscope_Permanent_ID", "Date_Added_To_Product"]]
Y_VAL.index = Y_VAL["serial_number"]
Y_VAL = Y_VAL[Y_VAL.notnull()]
Y_VAL.columns = ["predict_me", "id", "company_id", "time"]
Y_VAL = Y_VAL.astype(str)
Y_VAL["predict_me"] = Y_VAL["predict_me"].str.replace(r"\.0", "")
print(Y_VAL)
# %%
# seeing if the company type changed
BOTH = Y_VAL["predict_me"] + Y_VAL["company_id"]
print(BOTH.nunique())
print(Y_VAL["company_id"].nunique())
NEED = BOTH.nunique()
# BOTH are equal

# %%
Y_VAL = Y_VAL.sort_values("time", ascending=False)
CURR_INDEX =
# %%
# import the information data on the types
WIKI = requests.get(url =
                "https://en.wikipedia.org/wiki/Thomson_Reuters_Business_Classification")
PARS = bs4(WIKI.text)
TAB = PARS.body.find("table")
Y_MAP = pd.read_html(str(TAB))
Y_MAP

# %%

# %%
# playground
df = Y_VAL["predict_me"]
pd.DataFrame([df.unique, df.nunique])
# %%
df.str.startswith("5").sum()
