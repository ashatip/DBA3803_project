# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
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
import pandas as pd
# I make the data local on my machine will change it soon
DF = pd.read_csv("~/Desktop/DBA3803/group_assign/DBA3803_project/DATA/Financial_Info_SG_Firms_Combined.csv")

# %%
print("COLS Num", len(DF.columns))
#remove all nans
DF.dropna()
