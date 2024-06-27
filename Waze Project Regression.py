#!/usr/bin/env python
# coding: utf-8

# # **Waze Project**
# **Course 5 - Regression analysis: Simplify complex data relationships**

# Your team is more than halfway through their user churn project. Earlier, you completed a project proposal, used Python to explore and analyze Waze’s user data, created data visualizations, and conducted a hypothesis test. Now, leadership wants your team to build a regression model to predict user churn based on a variety of variables.
# You check your inbox and discover a new email from Ursula Sayo, Waze's Operations Manager. Ursula asks your team about the details of the regression model. You also notice two follow-up emails from your supervisor, May Santner. The first email is a response to Ursula, and says that the team will build a binomial logistic regression model. In her second email, May asks you to help build the model and prepare an executive summary to share your results.
# A notebook was structured and prepared to help you in this project. Please complete the following questions and prepare an executive summary.
# # **Course 5 End-of-course project: Regression modeling**
# In this activity, a binomial logistic regression model will be buily. Logistic regression helps you estimate the probability of an outcome. For data science professionals, this is a useful skill because it allows you to consider more than one variable against the variable you're measuring against. This opens the door for much more thorough and flexible analysis to be completed.
# **The purpose** of this project is to demostrate knowledge of exploratory data analysis (EDA) and a binomial logistic regression model.
# **The goal** is to build a binomial logistic regression model and evaluate the model's performance.

# *This activity has three parts:*
# **Part 1:** EDA & Checking Model Assumptions
# **Part 2:** Model Building and Evaluation
# **Part 3:** Interpreting Model Results

# Beginning of the program.
# ## **PACE: Plan**
# ### **Task 1. Imports and data loading**
# Packages for numerics + dataframes #
import pandas as pd
import numpy as np

# Packages for visualization #
import seaborn as sns
import matplotlib.pyplot as plt

# Packages for Logistic Regression & Confusion Matrix #
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Loading the dataset by running this cell #
data = pd.read_csv("waze_dataset.csv")

# ## **PACE: Analyze**
# ### **Task 2a. Explore data with EDA**

### Quick Analysis ###
data.shape, data.info()

## First 10 values ##
data.head(10)

### Deleting ID column since it is not necessary ###
data = data.drop("ID", axis=1)
data.head(10)

## Obtain number of retained and churned ##
data["label"].value_counts(dropna=False)

## Obtain percentage of retained and churned ##
data["label"].value_counts(normalize=True)

## Describe the data ##
data.describe()


# **Question:** Are there any variables that could potentially have outliers just by assessing at the quartile values, standard deviation, and max values?

# The variables with possible sessions based mostly on the difference between mean and max or min and using STD is sessions, drives,total_navigations_fav1, total_navigations_fav2 and driven_km_drives.

# ### **Task 2b. Create features**
# 1. Creating `km_per_driving_day` column
data["km_per_driving_day"] = data["driven_km_drives"]/data["driving_days"]
data.head(10)

# 2. Calling `describe()` on the new column
data["km_per_driving_day"].describe()

# 1. Converting infinite values to zero
data.loc[data["km_per_driving_day"] == np.inf,"km_per_driving_day"] = 0

# 2. Confirm that it worked
data["km_per_driving_day"].describe()

# Creating `professional_driver` column #
data["professional_driver"] = np.where((data["drives"] >= 60) & (data["driving_days"] >= 15), 1, 0)

# Demostration of the values for professional_driver #
columna = ["professional_driver","drives","driving_days"]
print(data[columna])

# 1. Check count of professionals and non-professionals
P = data["professional_driver"].value_counts(dropna=False)
PDrivers = P[1]
NPDrivers = P[0]
print("Number of Professional Drivers: ", PDrivers)
print("Number of Non Professional Drivers: ", NPDrivers)

# 2. Check in-class churn rate
data["label"]
churn_rate = data.groupby(["professional_driver"])["label"].value_counts(normalize = True)
churn_rate *= 100
churn_rate = churn_rate.apply(lambda x: f"{x:.2f}%")
churn_rate_a = churn_rate.reset_index(name="percentage")
churn_rate_a["professional_driver"] = churn_rate_a["professional_driver"].replace({0: "Non Professional", 1: "Professional"})
print(churn_rate_a)

### Verify missing values ###
data.info()

# Droping rows with missing data in `label` column
data.dropna(subset = ["label"])
data.info()
_drives`

# Imput outliers
columns = ["sessions", "drives", "total_sessions", "total_navigations_fav1","total_navigations_fav2", "driven_km_drives", "duration_minutes_drives"]
for column in columns:
    quantile95 = data[column].quantile(0.95)
    data.loc[data[column] > quantile95, column] = quantile95

### Quick Describe ###
data.describe()


# #### **Encode categorical variables**

# Create binary `label2` column
data["label2"] = np.where(data["label"] == "churned", 1, 0)
print(data[["label", "label2"]])


# ### **Task 3b. Determine whether assumptions have been met**
# The following are the assumptions for logistic regression:

# * Independent observations (This refers to how the data was collected.)
# * No extreme outliers
# * Little to no multicollinearity among X predictors
# * Linear relationship between X and the **logit** of y

# #### **Collinearity**

# Generating a correlation matrix
data.corr(method='pearson')

# Ploting correlation heatmap
plt.figure(figsize=(14,8))
sns.heatmap(data.corr(method='pearson'), vmin=-1, vmax=1, annot=True)
plt.title('Correlation heatmap')
plt.show();


# **Question:** Which variables are multicollinear with each other?

# Sessions and drives, activity days and driving days.

# ### **Task 3c. Create dummies **
 
# Create a new, binary column called `device2` that encodes user devices as follows:
# 
# * `Android` -> `0`
# * `iPhone` -> `1`

# Creating new `device2` variable
data["device2"] = np.where(data["device"] == "Android", 0, 1)

# Demostration of the values for professional_driver #
columna = ["device","device2"]
print(data[columna])


# ### **Task 3d. Model building**

# #### **Assign predictor variables and target**
# 
# To build your model you need to determine what X variables you want to include in your model to predict your target&mdash;`label2`.
# 
# Drop the following variables and assign the results to `X`:
# 
# * `label` (this is the target)
# * `label2` (this is the target)
# * `device` (this is the non-binary-encoded categorical variable)
# * `sessions` (this had high multicollinearity)
# * `driving_days` (this had high multicollinearity)
# 
# **Note:** Notice that `sessions` and `driving_days` were selected to be dropped, rather than `drives` and `activity_days`. The reason for this is that the features that were kept for modeling had slightly stronger correlations with the target variable than the features that were dropped.

# Isolating predictor variables
NoPV = data.drop(columns = ['label', 'label2', 'device', 'sessions', 'driving_days'])

# Isolate target variable
OnlyTV = data['label2']

# #### **Split the data**
# **Note 1:** It is important to do a train test to obtain accurate predictions.  You always want to fit your model on your training set and evaluate your model on your test set to avoid data leakage.
# 
# **Note 2:** Because the target class is imbalanced (82% retained vs. 18% churned), you want to make sure that you don't get an unlucky split that over- or under-represents the frequency of the minority class. Set the function's `stratify` parameter to `y` to ensure that the minority class appears in both train and test sets in the same proportion that it does in the overall dataset.

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(NoPV, OnlyTV, stratify=OnlyTV, random_state=14)

# Use .head()
X_train.head()


# Fit the model on `X_train` and `y_train`.

### Fitting on X and y train ###
model = LogisticRegression(penalty = "none", max_iter = 400)
model.fit(X_train, y_train)

### Using .coef_: change of logits ###
pd.Series(model.coef_[0], index = NoPV.columns)

### Intercept of the Model ###
model.intercept_

# Get the predicted probabilities of the training data
t_prob = model.predict_proba(X_train)
t_prob


# In logistic regression, the relationship between a predictor variable and the dependent variable does not need to be linear, however, the log-odds (a.k.a., logit) of the dependent variable with respect to the predictor variable should be linear. Here is the formula for calculating log-odds, where _p_ is the probability of response:

# 1. Copy the `X_train` dataframe and assign to `logit_data`
logit = X_train.copy()

# 2. Create a new `logit` column in the `logit_data` df
logit["logit"] = [np.log(prob[1] / prob[0]) for prob in t_prob]
logit["logit"]

# Plot regplot of `activity_days` log-odds
sns.regplot(x="activity_days", y="logit", data=logit, scatter_kws={"s": 2, "alpha": 0.05}, color="green")
sns.set_style("whitegrid")
plt.title("Logits of Predicted Probablities");


# ## **PACE: Execute**

# Generate predictions on X_test
YPredictions = model.predict(X_test)

# Score the model (accuracy) on the test data
model.score(X_test, y_test)

### Confusion Matrix Building ###
CMatrix = confusion_matrix(y_test, YPredictions)

### Displaying Confusion Matrix ###
disp = ConfusionMatrixDisplay(confusion_matrix = CMatrix, display_labels = ["Retained", "Churned"])
fig, ax = plt.subplots()
disp.plot(ax=ax)
ax.grid(False)

# Calculate PRECISION manually
precision = CMatrix[1,1] / (CMatrix[0, 1] + CMatrix[1, 1])
print("The Precision is:",precision)

# Calculate RECALL manually
recall = CMatrix[1,1] / (CMatrix[1, 0] + CMatrix[1, 1])
print("The Recall is:",recall)

# Creating a classification report
Labels = ["Retained", "Churned"]
print(classification_report(y_test, YPredictions, target_names=Labels))


# Creating a list of (column_name, coefficient) tuples
feature_importance = list(zip(X_train.columns, model.coef_[0]))

# Sort the list by coefficient value
feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
feature_importance

# Plot the feature importances
import seaborn as sns
sns.barplot(x=[x[1] for x in feature_importance], y=[x[0] for x in feature_importance],orient='h')
plt.title("Feature importances");


# . Conclusion*
# 1. What variable most influenced the model's prediction?
# 
# 2. Were there any variables that you expected to be stronger predictors than they were?
# 
# 3. Why might a variable you thought to be important not be important in the model?
# 
# 4. Would you recommend that Waze use this model? Why or why not?
# 
# 5. What could you do to improve this model?
# 
# 6. What additional features would you like to have to help improve the model?
# 

# 1. Activity_Days was the most important column for the prediction, as it helped analyze the negative correlation with churn between users.
# 2. Not really, just thought that the type of phone could affect in some way.
# 3. Phones, as there is not an actual reason to have other results. In any case, would use other methodology to analyze this.
# 4. Yes, as it helps making predictions and furthermore, analysis on what is the best and how to act.
# 5. Analyse more columns, as they may have a certain participation too. However, it is ok in my opinion.
# 6. Obtain more data since the begining and ask to the users how to improve.
