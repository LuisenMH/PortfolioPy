#!/usr/bin/env python
# coding: utf-8

# # **Waze Project**
# **Course 5 - Regression analysis: Simplify complex data relationships**

# Your team is more than halfway through their user churn project. Earlier, you completed a project proposal, used Python to explore and analyze Waze’s user data, created data visualizations, and conducted a hypothesis test. Now, leadership wants your team to build a regression model to predict user churn based on a variety of variables.
# 
# You check your inbox and discover a new email from Ursula Sayo, Waze's Operations Manager. Ursula asks your team about the details of the regression model. You also notice two follow-up emails from your supervisor, May Santner. The first email is a response to Ursula, and says that the team will build a binomial logistic regression model. In her second email, May asks you to help build the model and prepare an executive summary to share your results.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions and prepare an executive summary.

# # **Course 5 End-of-course project: Regression modeling**
# 
# In this activity, you will build a binomial logistic regression model. As you have learned, logistic regression helps you estimate the probability of an outcome. For data science professionals, this is a useful skill because it allows you to consider more than one variable against the variable you're measuring against. This opens the door for much more thorough and flexible analysis to be completed.
# <br/>
# 
# **The purpose** of this project is to demostrate knowledge of exploratory data analysis (EDA) and a binomial logistic regression model.
# 
# **The goal** is to build a binomial logistic regression model and evaluate the model's performance.
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** EDA & Checking Model Assumptions
# * What are some purposes of EDA before constructing a binomial logistic regression model?
# 
# **Part 2:** Model Building and Evaluation
# * What resources do you find yourself using as you complete this stage?
# 
# **Part 3:** Interpreting Model Results
# 
# * What key insights emerged from your model(s)?
# 
# * What business recommendations do you propose based on the models built?
# 
# <br/>
# 
# Follow the instructions and answer the question below to complete the activity. Then, you will complete an executive summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.

# # **Build a regression model**

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.

# ### **Task 1. Imports and data loading**
# Import the data and packages that you've learned are needed for building logistic regression models.

# In[128]:


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


# Import the dataset.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[129]:


# Load the dataset by running this cell #
data = pd.read_csv("waze_dataset.csv")


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.
# 
# In this stage, consider the following question:
# 
# * What are some purposes of EDA before constructing a binomial logistic regression model?

# ==> ENTER YOUR RESPONSE HERE

# ### **Task 2a. Explore data with EDA**
# 
# Analyze and discover data, looking for correlations, missing data, potential outliers, and/or duplicates.
# 
# 

# Start with `.shape` and `info()`.

# In[130]:


### Quick Analysis ###
data.shape, data.info()


# Label has less values, being 14299 instead of the expected 14999

# ==> ENTER YOUR RESPONSE HERE

# Use `.head()`.
# 
# 

# In[131]:


## First 10 values ##
data.head(10)


# Use `.drop()` to remove the ID column since we don't need this information for your analysis.

# In[132]:


### Deleting ID column since it is not necessary ###
data = data.drop("ID", axis=1)
data.head(10)


# Now, check the class balance of the dependent (target) variable, `label`.

# In[133]:


## Obtain number of retained and churned ##
data["label"].value_counts(dropna=False)


# In[134]:


## Obtain percentage of retained and churned ##
data["label"].value_counts(normalize=True)


# Call `.describe()` on the data.
# 

# In[135]:


## Describe the data ##
data.describe()


# **Question:** Are there any variables that could potentially have outliers just by assessing at the quartile values, standard deviation, and max values?

# The variables with possible sessions based mostly on the difference between mean and max or min and using STD is sessions, drives,total_navigations_fav1, total_navigations_fav2 and driven_km_drives.

# ### **Task 2b. Create features**
# 
# Create features that may be of interest to the stakeholder and/or that are needed to address the business scenario/problem.

# #### **`km_per_driving_day`**
# 
# You know from earlier EDA that churn rate correlates with distance driven per driving day in the last month. It might be helpful to engineer a feature that captures this information.
# 
# 1. Create a new column in `df` called `km_per_driving_day`, which represents the mean distance driven per driving day for each user.
# 
# 2. Call the `describe()` method on the new column.

# In[136]:


# 1. Creating `km_per_driving_day` column
data["km_per_driving_day"] = data["driven_km_drives"]/data["driving_days"]
data.head(10)


# In[137]:


# 2. Calling `describe()` on the new column
data["km_per_driving_day"].describe()


# Note that some values are infinite. This is the result of there being values of zero in the `driving_days` column. Pandas imputes a value of infinity in the corresponding rows of the new column because division by zero is undefined.
# 
# 1. Convert these values from infinity to zero. You can use `np.inf` to refer to a value of infinity.
# 
# 2. Call `describe()` on the `km_per_driving_day` column to verify that it worked.

# In[138]:


# 1. Converting infinite values to zero
data.loc[data["km_per_driving_day"] == np.inf,"km_per_driving_day"] = 0

# 2. Confirm that it worked
data["km_per_driving_day"].describe()


# #### **`professional_driver`**
# 
# Create a new, binary feature called `professional_driver` that is a 1 for users who had 60 or more drives <u>**and**</u> drove on 15+ days in the last month.
# 
# **Note:** The objective is to create a new feature that separates professional drivers from other drivers. In this scenario, domain knowledge and intuition are used to determine these deciding thresholds, but ultimately they are arbitrary.

# To create this column, use the [`np.where()`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) function. This function accepts as arguments:
# 1. A condition
# 2. What to return when the condition is true
# 3. What to return when the condition is false
# 
# ```
# Example:
# x = [1, 2, 3]
# x = np.where(x > 2, 100, 0)
# x
# array([  0,   0, 100])
# ```

# In[139]:


# Creating `professional_driver` column #
data["professional_driver"] = np.where((data["drives"] >= 60) & (data["driving_days"] >= 15), 1, 0)

# Demostration of the values for professional_driver #
columna = ["professional_driver","drives","driving_days"]
print(data[columna])


# Perform a quick inspection of the new variable.
# 
# 1. Check the count of professional drivers and non-professionals
# 
# 2. Within each class (professional and non-professional) calculate the churn rate

# In[140]:


# 1. Check count of professionals and non-professionals
P = data["professional_driver"].value_counts(dropna=False)
PDrivers = P[1]
NPDrivers = P[0]
print("Number of Professional Drivers: ", PDrivers)
print("Number of Non Professional Drivers: ", NPDrivers)


# In[141]:


# 2. Check in-class churn rate
data["label"]
churn_rate = data.groupby(["professional_driver"])["label"].value_counts(normalize = True)
churn_rate *= 100
churn_rate = churn_rate.apply(lambda x: f"{x:.2f}%")
churn_rate_a = churn_rate.reset_index(name="percentage")
churn_rate_a["professional_driver"] = churn_rate_a["professional_driver"].replace({0: "Non Professional", 1: "Professional"})
print(churn_rate_a)


# The churn rate for professional drivers is 7.6%, while the churn rate for non-professionals is 19.9%. This seems like it could add predictive signal to the model.

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# 
# After analysis and deriving variables with close relationships, it is time to begin constructing the model.
# 
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.
# 
# In this stage, consider the following question:
# 
# * Why did you select the X variables you did?

# Because it is easier to have a control.

# ### **Task 3a. Preparing variables**

# Call `info()` on the dataframe to check the data type of the `label` variable and to verify if there are any missing values.

# In[142]:


### Verify missing values ###
data.info()


# Because you know from previous EDA that there is no evidence of a non-random cause of the 700 missing values in the `label` column, and because these observations comprise less than 5% of the data, use the `dropna()` method to drop the rows that are missing this data.

# In[143]:


# Droping rows with missing data in `label` column
data.dropna(subset = ["label"])
data.info()


# #### **Impute outliers**
# 
# You rarely want to drop outliers, and generally will not do so unless there is a clear reason for it (e.g., typographic errors).
# 
# At times outliers can be changed to the **median, mean, 95th percentile, etc.**
# 
# Previously, you determined that seven of the variables had clear signs of containing outliers:
# 
# * `sessions`
# * `drives`
# * `total_sessions`
# * `total_navigations_fav1`
# * `total_navigations_fav2`
# * `driven_km_drives`
# * `duration_minutes_drives`
# 
# For this analysis, impute the outlying values for these columns. Calculate the **95th percentile** of each column and change to this value any value in the column that exceeds it.
# 

# In[144]:


# Imput outliers
columns = ["sessions", "drives", "total_sessions", "total_navigations_fav1","total_navigations_fav2", "driven_km_drives", "duration_minutes_drives"]
for column in columns:
    quantile95 = data[column].quantile(0.95)
    data.loc[data[column] > quantile95, column] = quantile95


# Call `describe()`.

# In[145]:


### YOUR CODE HERE ###
data.describe()


# #### **Encode categorical variables**

# Change the data type of the `label` column to be binary. This change is needed to train a logistic regression model.
# 
# Assign a `0` for all `retained` users.
# 
# Assign a `1` for all `churned` users.
# 
# Save this variable as `label2` as to not overwrite the original `label` variable.
# 
# **Note:** There are many ways to do this. Consider using `np.where()` as you did earlier in this notebook.

# In[146]:


# Create binary `label2` column
data["label2"] = np.where(data["label"] == "churned", 1, 0)
print(data[["label", "label2"]])


# ### **Task 3b. Determine whether assumptions have been met**
# 
# The following are the assumptions for logistic regression:
# 
# * Independent observations (This refers to how the data was collected.)
# 
# * No extreme outliers
# 
# * Little to no multicollinearity among X predictors
# 
# * Linear relationship between X and the **logit** of y
# 
# For the first assumption, you can assume that observations are independent for this project.
# 
# The second assumption has already been addressed.
# 
# The last assumption will be verified after modeling.
# 
# **Note:** In practice, modeling assumptions are often violated, and depending on the specifics of your use case and the severity of the violation, it might not affect your model much at all or it will result in a failed model.

# #### **Collinearity**
# 
# Check the correlation among predictor variables. First, generate a correlation matrix.

# In[147]:


# Generating a correlation matrix
data.corr(method='pearson')


# Now, plot a correlation heatmap.

# In[148]:


# Ploting correlation heatmap
plt.figure(figsize=(14,8))
sns.heatmap(data.corr(method='pearson'), vmin=-1, vmax=1, annot=True)
plt.title('Correlation heatmap')
plt.show();


# If there are predictor variables that have a Pearson correlation coefficient value greater than the **absolute value of 0.7**, these variables are strongly multicollinear. Therefore, only one of these variables should be used in your model.
# 
# **Note:** 0.7 is an arbitrary threshold. Some industries may use 0.6, 0.8, etc.
# 
# **Question:** Which variables are multicollinear with each other?

# Sessions and drives, activity days and driving days.

# ### **Task 3c. Create dummies (if necessary)**
# 
# If you have selected `device` as an X variable, you will need to create dummy variables since this variable is categorical.
# 
# In cases with many categorical variables, you can use pandas built-in [`pd.get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), or you can use scikit-learn's [`OneHotEncoder()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) function.
# 
# **Note:** Variables with many categories should only be dummied if absolutely necessary. Each category will result in a coefficient for your model which can lead to overfitting.
# 
# Because this dataset only has one remaining categorical feature (`device`), it's not necessary to use one of these special functions. You can just implement the transformation directly.
# 
# Create a new, binary column called `device2` that encodes user devices as follows:
# 
# * `Android` -> `0`
# * `iPhone` -> `1`

# In[149]:


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

# In[150]:


# Isolating predictor variables
NoPV = data.drop(columns = ['label', 'label2', 'device', 'sessions', 'driving_days'])


# Now, isolate the dependent (target) variable. Assign it to a variable called `y`.

# In[151]:


# Isolate target variable
OnlyTV = data['label2']


# #### **Split the data**
# 
# Use scikit-learn's [`train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function to perform a train/test split on your data using the X and y variables you assigned above.
# 
# **Note 1:** It is important to do a train test to obtain accurate predictions.  You always want to fit your model on your training set and evaluate your model on your test set to avoid data leakage.
# 
# **Note 2:** Because the target class is imbalanced (82% retained vs. 18% churned), you want to make sure that you don't get an unlucky split that over- or under-represents the frequency of the minority class. Set the function's `stratify` parameter to `y` to ensure that the minority class appears in both train and test sets in the same proportion that it does in the overall dataset.

# In[152]:


# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(NoPV, OnlyTV, stratify=OnlyTV, random_state=14)


# In[153]:


# Use .head()
X_train.head()


# Use scikit-learn to instantiate a logistic regression model. Add the argument `penalty = None`.
# 
# It is important to add `penalty = None` since your predictors are unscaled.
# 
# Refer to scikit-learn's [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) documentation for more information.
# 
# Fit the model on `X_train` and `y_train`.

# In[154]:


### Fitting on X and y train ###
model = LogisticRegression(penalty = "none", max_iter = 400)
model.fit(X_train, y_train)


# Call the `.coef_` attribute on the model to get the coefficients of each variable.  The coefficients are in order of how the variables are listed in the dataset.  Remember that the coefficients represent the change in the **log odds** of the target variable for **every one unit increase in X**.
# 
# If you want, create a series whose index is the column names and whose values are the coefficients in `model.coef_`.

# In[155]:


### Using .coef_: change of logits ###
pd.Series(model.coef_[0], index = NoPV.columns)


# Call the model's `intercept_` attribute to get the intercept of the model.

# In[156]:


### Intercept of the Model ###
model.intercept_


# #### **Check final assumption**
# 
# Verify the linear relationship between X and the estimated log odds (known as logits) by making a regplot.
# 
# Call the model's `predict_proba()` method to generate the probability of response for each sample in the training data. (The training data is the argument to the method.) Assign the result to a variable called `training_probabilities`. This results in a 2-D array where each row represents a user in `X_train`. The first column is the probability of the user not churning, and the second column is the probability of the user churning.

# In[157]:


# Get the predicted probabilities of the training data
t_prob = model.predict_proba(X_train)
t_prob


# In logistic regression, the relationship between a predictor variable and the dependent variable does not need to be linear, however, the log-odds (a.k.a., logit) of the dependent variable with respect to the predictor variable should be linear. Here is the formula for calculating log-odds, where _p_ is the probability of response:
# <br>
# $$
# logit(p) = ln(\frac{p}{1-p})
# $$
# <br>
# 
# 1. Create a dataframe called `logit_data` that is a copy of `df`.
# 
# 2. Create a new column called `logit` in the `logit_data` dataframe. The data in this column should represent the logit for each user.
# 

# In[158]:


# 1. Copy the `X_train` dataframe and assign to `logit_data`
logit = X_train.copy()

# 2. Create a new `logit` column in the `logit_data` df
logit["logit"] = [np.log(prob[1] / prob[0]) for prob in t_prob]
logit["logit"]


# Plot a regplot where the x-axis represents an independent variable and the y-axis represents the log-odds of the predicted probabilities.
# 
# In an exhaustive analysis, this would be plotted for each continuous or discrete predictor variable. Here we show only `driving_days`.

# In[159]:


# Plot regplot of `activity_days` log-odds
sns.regplot(x="activity_days", y="logit", data=logit, scatter_kws={"s": 2, "alpha": 0.05}, color="green")
sns.set_style("whitegrid")
plt.title("Logits of Predicted Probablities");


# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 4a. Results and evaluation**
# 
# If the logistic assumptions are met, the model results can be appropriately interpreted.
# 
# Use the code block below to make predictions on the test data.
# 

# In[160]:


# Generate predictions on X_test
YPredictions = model.predict(X_test)


# Now, use the `score()` method on the model with `X_test` and `y_test` as its two arguments. The default score in scikit-learn is **accuracy**.  What is the accuracy of your model?
# 
# *Consider:  Is accuracy the best metric to use to evaluate this model?*

# In[161]:


# Score the model (accuracy) on the test data
model.score(X_test, y_test)


# ### **Task 4b. Show results with a confusion matrix**

# Use the `confusion_matrix` function to obtain a confusion matrix. Use `y_test` and `y_preds` as arguments.

# In[165]:


### Confusion Matrix Building ###
CMatrix = confusion_matrix(y_test, YPredictions)


# Next, use the `ConfusionMatrixDisplay()` function to display the confusion matrix from the above cell, passing the confusion matrix you just created as its argument.

# In[183]:


### Displaying Confusion Matrix ###
disp = ConfusionMatrixDisplay(confusion_matrix = CMatrix, display_labels = ["Retained", "Churned"])
fig, ax = plt.subplots()
disp.plot(ax=ax)
ax.grid(False)


# You can use the confusion matrix to compute precision and recall manually. You can also use scikit-learn's [`classification_report()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) function to generate a table from `y_test` and `y_preds`.

# In[190]:


# Calculate PRECISION manually
precision = CMatrix[1,1] / (CMatrix[0, 1] + CMatrix[1, 1])
print("The Precision is:",precision)


# In[198]:


# Calculate RECALL manually
recall = CMatrix[1,1] / (CMatrix[1, 0] + CMatrix[1, 1])
print("The Recall is:",recall)


# In[200]:


# Creating a classification report
Labels = ["Retained", "Churned"]
print(classification_report(y_test, YPredictions, target_names=Labels))


# **Note:** The model has decent precision but very low recall, which means that it makes a lot of false negative predictions and fails to capture users who will churn.

# ### **BONUS**
# 
# Generate a bar graph of the model's coefficients for a visual representation of the importance of the model's features.

# In[201]:


# Creating a list of (column_name, coefficient) tuples
feature_importance = list(zip(X_train.columns, model.coef_[0]))

# Sort the list by coefficient value
feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
feature_importance


# In[202]:


# Plot the feature importances
import seaborn as sns
sns.barplot(x=[x[1] for x in feature_importance], y=[x[0] for x in feature_importance],orient='h')
plt.title("Feature importances");


# ### **Task 4c. Conclusion**
# 
# Now that you've built your regression model, the next step is to share your findings with the Waze leadership team. Consider the following questions as you prepare to write your executive summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.
# 
# **Questions:**
# 
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

# In[205]:


# 1. Activity_Days was the most important column for the prediction, as it helped analyze the negative correlation with churn between users.
# 2. Not really, just thought that the type of phone could affect in some way.
# 3. Phones, as there is not an actual reason to have other results. In any case, would use other methodology to analyze this.
# 4. Yes, as it helps making predictions and furthermore, analysis on what is the best and how to act.
# 5. Analyse more columns, as they may have a certain participation too. However, it is ok in my opinion.
# 6. Obtain more data since the begining and ask to the users how to improve.

