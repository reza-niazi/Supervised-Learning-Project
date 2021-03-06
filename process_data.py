# Generates datasets needed by neural_network.py and perceptron.py
import pandas as pd
import numpy as np

rand_seed = 1234

# Load dataset from csv file
data = pd.read_csv("datasets/covid_confirmed_usafacts.csv")

# Calculate new cases for each day
new_cases = pd.DataFrame()
# Note: probably a better way to do this
for i, col in enumerate(data.columns):
    if i < 4:
        new_cases[col] = data[col]
    if i > 4:
        new_cases[col] = data[col] - data.iloc[:, i - 1]
# new_cases.to_csv("datasets/new_cases.csv", index=False)

pop_data = pd.read_csv("datasets/covid_county_population_usafacts.csv")

# Calculate daily new cases per 100k population
new_cases_per_100k = pd.DataFrame()
for i, col in enumerate(new_cases.columns):
    if i < 4:
        new_cases_per_100k[col] = new_cases[col]
    else:
        new_cases_per_100k[col] = 100000 * new_cases[col] / pop_data["population"]
new_cases_per_100k.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
# new_cases_per_100k.to_csv("datasets/new_cases_per_100k.csv", index=False)

# Calculate the running 7-day average of new cases per 100k population
avg_new_cases_per_100k = pd.DataFrame()
for i, col in enumerate(new_cases_per_100k.columns):
    if i < 4:
        avg_new_cases_per_100k[col] = new_cases_per_100k[col]
    else:
        avg_new_cases_per_100k[col] = new_cases_per_100k.iloc[:, max(4, i-6):i+1].mean(axis=1)

# Create smaller dataset of new cases per 100k for just April
avg_new_cases_per_100k_april = pd.concat([avg_new_cases_per_100k.loc[:, "countyFIPS":"stateFIPS"], avg_new_cases_per_100k.loc[:, "4/1/20":"4/30/20"]], axis=1, sort=False)
avg_new_cases_per_100k_april.loc[avg_new_cases_per_100k_april["4/14/20"] > avg_new_cases_per_100k_april["4/1/20"], "w2inc"] = 1
avg_new_cases_per_100k_april.loc[avg_new_cases_per_100k_april["4/14/20"] <= avg_new_cases_per_100k_april["4/1/20"], "w2inc"] = 0
avg_new_cases_per_100k_april.loc[avg_new_cases_per_100k_april["4/21/20"] > avg_new_cases_per_100k_april["4/8/20"], "w3inc"] = 1
avg_new_cases_per_100k_april.loc[avg_new_cases_per_100k_april["4/21/20"] <= avg_new_cases_per_100k_april["4/8/20"], "w3inc"] = 0
avg_new_cases_per_100k_april.loc[avg_new_cases_per_100k_april["4/28/20"] > avg_new_cases_per_100k_april["4/15/20"], "w4inc"] = 1
avg_new_cases_per_100k_april.loc[avg_new_cases_per_100k_april["4/28/20"] <= avg_new_cases_per_100k_april["4/15/20"], "w4inc"] = 0
# avg_new_cases_per_100k_april.to_csv("datasets/avg_new_cases_per_100k_april.csv", index=False)

# Create dataset with every week from Jan-Dec
# Note: this is probably really inefficient
row_list = []
for _, row in avg_new_cases_per_100k.iterrows():
    for week_start in range(4, len(avg_new_cases_per_100k.columns) - 13):
        weekDict = {}
        for i, day in enumerate(range(week_start, week_start + 7)):
            weekDict[i] = row[day]
        if row[week_start + 13] > row[week_start]:
            weekDict["inc"] = 1
        else:
            weekDict["inc"] = 0
        row_list.append(weekDict)
df_weeks = pd.DataFrame(row_list)
# Remove rows where every day is 0
df_weeks = df_weeks.loc[(df_weeks.iloc[:, 0:6] != 0).any(axis=1)]
# df_weeks.to_csv("datasets/avg_new_cases_per_100k_weeks.csv", index=False)

# Training, validation, and testing datasets for first two weeks of April
week1 = avg_new_cases_per_100k_april.loc[:, "4/1/20":"4/7/20"]
week1 = week1.loc[(week1.loc[:, "4/1/20":"4/7/20"] != 0).any(axis=1)]
week1["w2inc"] = avg_new_cases_per_100k_april["w2inc"]
training_data = week1
training_data.to_csv("datasets/project_training.csv", index=False)
week2 = avg_new_cases_per_100k_april.loc[:, "4/8/20":"4/14/20"]
week2 = week2.loc[(week2.loc[:, "4/8/20":"4/14/20"] != 0).any(axis=1)]
week2["w3inc"] = avg_new_cases_per_100k_april["w3inc"]
validation_data = week2.sample(frac=0.8, random_state=rand_seed)
testing_data = week2[~week2.index.isin(validation_data.index)]
validation_data.to_csv("datasets/project_validation.csv", index=False)
testing_data.to_csv("datasets/project_test.csv", index=False)

# Training and validation datasets for every set of 7 days
data = df_weeks
validation_data = data.sample(frac=0.2, random_state=rand_seed)
validation_data.to_csv("datasets/project_validation_2.csv", index=False)
training_data = data[~data.index.isin(validation_data.index)]
training_data.to_csv("datasets/project_training_2.csv", index=False)