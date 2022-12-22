import pandas as pd
import numpy as np

# formatting attendance data 

df1 = pd.read_csv(r'D:\Pupil Progress\Results _ Reception.csv', error_bad_lines=False)

df1 = df1.iloc[2: , :]
df1 = df1.iloc[: ,0:3]
df1 = df1.iloc[::2]
cols = ['Unnamed: 1', 'Unnamed: 2']
df1["Unnamed: 1"] = df1["Unnamed: 1"].str[:-1].astype(float)
df1["Unnamed: 2"] = df1["Unnamed: 2"].str[:-1].astype(float)
df1['attendance'] = df1[cols].sum(axis=1)/2
df1.rename(columns = {' ':'Name'}, inplace = True)
df1["Name"] = df1["Name"].str[:-11]
# need to swap name values before comma
df1 = df1.drop(["Unnamed: 1","Unnamed: 2"], axis=1)
names = list(df1["Name"])
formattedNames = []

# reformatting names in attendance so they match tapestry
for i in range(len(names)):
    newName = names[i].split(",")
    newName = newName[-1] + newName[0]
    newName = newName.strip()
    formattedNames.append(newName)

# formatting achievement data   
df1["Name"] = formattedNames


df2 = pd.read_csv(r"D:\Pupil Progress\Tapestry data copy.csv", error_bad_lines=False)

df2 = df2[df2["Child"].isin(df1["Name"])] 
df2.rename(columns = {'Child':'Name'}, inplace = True)

# merging dfs

merged_df = df1.merge(df2, on='Name')
#merged_df = pd.merge(df1, df2, on="Name")

# saving to file 
merged_df.to_csv(r'D:\Pupil Progress\formattedData.csv')

# replacing with 0's and 1's and 2's

for column in merged_df.columns[2::]:
    merged_df[column] = merged_df[column].replace(["Concerns","Review","No Concerns"], [0, 1, 2])

# build the multinomial regression model
# evaluate multinomial logistic regression model
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# define learning area to predict acheivement in 
y = merged_df["L"]
X = []

for attendance in merged_df["attendance"]:
    X.append([attendance])
# define the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))