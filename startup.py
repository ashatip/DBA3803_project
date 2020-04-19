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
import use_me_linear_newest
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
pd.set_option('display.max_rows', 1000)
pd.set_option('display.min_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
DIR = os.getcwd()
# %%

# this is for creating the defintion table
DEF1 = pd.read_excel(DIR + "/DATA/Financial_Info_SG_Firms.xlsx",
                     sheet_name="DEFINITION_WORLDSCOPE_STATIC")
DEF1.set_index("Name", inplace=True)
DEF2 = pd.read_excel(DIR + "/DATA/Financial_Info_SG_Firms.xlsx",
                     sheet_name="DEFINITION_WORLDSCOPE_TS")
DEF2.set_index("Name", inplace=True)
DEF = DEF1.append(DEF2)
DEF.sort_index(inplace=True)
print(DEF.index)

# look it up like this:
print(DEF.loc["ADR CUSIP 1"])

# %%
print("DEF1")
for def1 in DEF1.index.values:
    print(def1)
print("DEF2")
for def2 in DEF2.index.values:
    print(def2)
# %%


def clean_up(data):
    """I made this method because our professor gave us a dataframe that is
    neither clean or understandable"""
    print("OLD")
    print(data.shape)
    data.dropna(axis=0, how="all", inplace=True)
    data.dropna(axis=1, how="all", inplace=True)
    fews = data.nunique(dropna=True)
    dels_me = fews[fews < 3]
    data.drop(columns=dels_me.index, inplace=False)
    data.drop_duplicates(inplace=True)
    print("NEW")
    print(data.shape)
    data.columns = [re.sub(r"\s", "_", x) for x in list(data)]
    return data


# %%
DF10 = pd.read_excel(DIR + "/DATA/Financial_Info_SG_Firms.xlsx",
                     sheet_name="DATA_WORLDSCOPE_TS")
DF11 = pd.read_excel(DIR + "/DATA/Financial_Info_SG_Firms.xlsx",
                     sheet_name="DATA_WORLDSCOPE_STATIC")
DF10 = clean_up(DF10)
DF11 = clean_up(DF11)

# %%
UNI =  DF10["Worldscope_Permanent_ID"] + DF10["Year"].astype(str)
print(UNI.nunique())
# ok so I realized i was using the wrong data

# %%
NEW_Y = DF11[["Worldscope_Permanent_ID",
             "Thomson_Reuters_Business_Classification_Code"]].dropna(how =
                                                                      "any")
NEW_Y.shape
# %%
# clean DF10
DF00 = DF10.select_dtypes(exclude="object")
DF00["company_id"] = DF10["Worldscope_Permanent_ID"]
# see if the company_id match
print(DF00["company_id"].nunique())
print(NEW_Y["Worldscope_Permanent_ID"].nunique())
# does not
# %%
UNI_COM = NEW_Y["Worldscope_Permanent_ID"].unique()
DF00 = DF00.loc[DF00["company_id"].isin(UNI_COM), list(DF00)]
DF00 = clean_up(DF00)

# %%
COLS = pd.Series(list(DF00))
SPEC = COLS.str.contains("Total_Asset|Sales|company", regex=True)
YEARS = COLS.str.contains("%|//", regex=True)
COLS_IND = [spec or years for spec, years in zip(SPEC, YEARS)]
WANT = DF00.loc[:, COLS_IND].dropna(axis=0, how="all")
WANT = clean_up(WANT)
# WANT.dropna(inplace=True, how="any", axis=1)
print(WANT.isnull().sum())
# %%
# DIR = os.path.dirname(os.path.realpath(__file__))
DF0 = pd.read_csv(DIR + "/DATA/Financial_Info_SG_Firms_Combined.csv",
                  low_memory=False)
print(DF0.shape)
DF0["serial number"].nunique()

# %%
print(sorted(list(set(DEF.index) - set(list(DF0)))))
print(sorted(list(set(list(DF0)) - set(DEF.index))))
COLS = list(set(DEF.index).intersection(set(list(DF0))))
COLS = ["Worldscope Permanent ID"] + COLS
print(len(COLS))
# so confused the defintions do not match the table, not good at all I will
# just use the columns in the DEFintions because i know what they mean

# %%
# cleaning
# remove all nans
DF1_sub= DF0[COLS].copy()
DF1_sub = clean_up(DF1_sub)
DF1= DF1_sub.select_dtypes(exclude=["object"])
# %%
DF1["Date_Added_To_Product"] = DF1["Date_Added_To_Product"].astype(str)
DF1["Worldscope_Permanent_ID"] = DF1_sub["Worldscope_Permanent_ID"]
DF1.dropna(how = "any", axis = 0, subset = ["Thomson_Reuters_Business_Classification_Code"])
DF1.shape

# %%
FEW = DF1.nunique(dropna=True)
DELS = FEW[FEW < 50]
print(DELS)
print(len(DELS))
print(DF1.shape)
# %%
print(DF1.shape)
TIME = DF1["Date_Added_To_Product"].nunique()
COMP = DF1["Worldscope_Permanent_ID"].nunique()
print(TIME)
print(COMP)
# %%
UNI =  DF1["Worldscope_Permanent_ID"] + DF1["Date_Added_To_Product"]
EXA = DF1.query("Worldscope_Permanent_ID == 'C702H1440' and Date_Added_To_Product" +
                "== '19920609'")
EXA.dropna(how = "any", axis = 1, inplace = True)
print(UNI.nunique())

# %%
DF1["Worldscope_Permanent_ID"].head()
# %%
# KEEP, ID_V, TIME, LOCATION, REMOVE = view_df(DF1)

# %%
print(DEF.index[DEF.index.str.contains("Month")])
# %% [markdown]
# building models to predict Industry type
# %%
Y_VAL = DF1[["Thomson_Reuters_Business_Classification_Code", "serial_number",
             "Worldscope_Permanent_ID", "Date_Added_To_Product"]]
Y_VAL = Y_VAL.dropna(how="any")
Y_VAL.columns = ["predict_me", "id", "company_id", "time"]
Y_VAL = Y_VAL.astype(str)
Y_VAL["predict_me"] = Y_VAL["predict_me"].str.replace(r"\.0", "")
Y_VAL.sort_index(inplace=True)
print(Y_VAL)

# %%
# seeing if the company type changed
BOTH = Y_VAL["predict_me"] + Y_VAL["company_id"]
print(BOTH.nunique())
print(Y_VAL["company_id"].nunique())
NEED = BOTH.nunique()
# BOTH are equal

# %%
UNI_COM = Y_VAL["company_id"].unique()
ROWS = []
for uni in UNI_COM:
    subs = Y_VAL[Y_VAL["company_id"] == uni]
    time_max = max(subs["time"])
    ROWS.append(subs[subs["time"] == time_max].index)
print(len(ROWS))
print(ROWS)

# %% [markdown]
# So I will make a model that given a companies balance sheet and other stuff
# it will predict what type of company it is. I want my model to be flexible,
# so I will only use companies that are unique and their most recent time table
# even if they have multiple entries

# %%
# getting my indices for values
Y_VAL = Y_VAL.sort_values("time", ascending=True)
DIS = Y_VAL[["id", "predict_me", "company_id"]]
DIS.index = DIS["id"]
print(DIS["predict_me"])

# %%
# see the values
LENS = Y_VAL["predict_me"].str.len()
print(LENS.value_counts())
# %%
print(Y_VAL["time"])
print(Y_VAL["time"].value_counts())
# %%
SUB = Y_VAL["company_id"].drop_duplicates(keep="first")
CHO = pd.merge(SUB, Y_VAL[["predict_me", "time"]], left_index=True,
               right_index=True, how="left")

print(CHO.shape)
print("IDs", CHO.index)
print(CHO.head())
print("TIMES", CHO["time"].value_counts())

# %%
CHO["length"] = CHO["predict_me"].str.len()
LEN = CHO["length"]
# most of them have length of 10
print(LEN.value_counts())
TENTHS = CHO[LEN == 10]
EIGHTS = CHO[LEN == 8]
SIXTH = CHO[LEN == 6]
print(TENTHS["predict_me"].value_counts())
TENTHS["predict_me"] = TENTHS["predict_me"].map(lambda x: str(x)[:-6])
EIGHTS["predict_me"] = EIGHTS["predict_me"].map(lambda x: str(x)[:-4])
SIXTH["predict_me"] = SIXTH["predict_me"].map(lambda x: str(x)[:-2])
CHOS = pd.concat([TENTHS, EIGHTS, SIXTH])
CHOS["length"] = CHOS["predict_me"].str.len()
# print(CHOS)
# print(CHOS["length"].value_counts())
# print(CHOS["time"].value_counts())
# a bit diverse in time but that would be fine
UNI = CHOS["predict_me"].value_counts()
print(UNI)
print(UNI.sort_index())
print(CHOS["time"].value_counts().sort_index())
# I decided to make all the values in length 6 because it will use all the data
# and will group the types of Industry togehter better

# %% [markdown]
# Now the Thomson_Reuters_Business_Classification classifies each company with
# a unique value which determines which company type it is. I got the data from
# wikipeida and brought it in using python,

# %%
# import the information data on the types
WIKI = requests.get(
    "https://en.wikipedia.org/wiki/Thomson_Reuters_Business_Classification")
PARS = bs4(WIKI.text, features="lxml")
TAB = PARS.body.find("table")
Y_MAP = pd.read_html(str(TAB), header=0)[0]
print(Y_MAP)
Y_MAP = Y_MAP[Y_MAP.columns[len(list(Y_MAP)) - 1]]
SPL = Y_MAP.str.split(" ", n=1, expand=True)
Y_MAPS = pd.DataFrame({"code": SPL[0], "name": SPL[1]})
print(Y_MAPS)
# %% [markdown]
# choosing columns: I am planning to choose columns that are based on balance
# sheet and divide each number by the total value of assets and for the income
# statement I will divided it by the sales revenue and the cash flows. I will
# use the columns with Total Assets 5 year average and Total Sales 5 year
# average

# %%
PREC = DEF.index.str.contains("Sales 5")
FIVE = DEF.index.str.contains("Total Assets")
PRE_IND = [one or sec for one, sec in zip(FIVE, PREC)]
print("LEN", sum(PRE_IND))
PRE = DEF[PRE_IND]
PRE.drop(index=["Capital Expenditure % Total Sales 5 Year Average"],
         inplace=True)
for deff in PRE.index:
    print(deff)
# %%
COLS = [re.sub(r"\s", "_", x) for x in PRE.index]
COLS.remove("Total_Assets_GAAP")
COLS.append("Thomson_Reuters_Business_Classification_Code")
FIN = DF1.loc[CHOS.index][COLS]
print(FIN.isnull().sum())
