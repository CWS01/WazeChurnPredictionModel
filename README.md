# WazeChurnPredictionModel
How can Waze leverage existing data to minimize user churn rate?

## Plan
### Background
Waze’s free navigation app makes it easier for drivers around the world to get to where they want to go. Waze’s community of map editors, beta testers, translators, partners, and users helps make each drive better and safer. Waze partners with cities, transportation authorities, broadcasters, businesses, and first responders to help as many people as possible travel more efficiently and safely. Recently, Waze management has noticed that there is a nonnegligible number of users who churn form, or no longer use, the Waze app. Due to this finding, Waze management would like to explore the factors that appear to influence whether a user churns, ultimately leading to better user retention and growth for the Waze business.

### Stakeholders

#### Data Team
* Harriet Hadzic - Director of Data Analysis
* May Santner - Data Analysis Manager
* Chidi Ga - Senior Data Analyst
* Sylvester Esperanza - Senior Project Manager

#### Other Stakeholders
* Emmick Larson - Finance and Administartion Department Head
* Ursula Sayo - Operations Manager

### Business Task
Develop a machine learning model to prevent user churn, improve user retention, and grow Waze's business by allowing Waze to implement interventions for users at risk for churning.

### Key Questions
1. Who are the users most likely to churn?
2. Why do user churn?
3. When do users churn?
4. Is there a relationship between mean amount of rides and device type?

### Data Source
The data for this project was provided as part of the Google Advanced Data Analytics Certificate Program. The data set contained 14999 observations for 12 of the 13 columns proided in the data set. One column was missing 700 values, this was the `label` column. The data was synthetic and its only purpose was for use with this project.
#### Data Dictionary
![image](https://github.com/user-attachments/assets/e2e0ea5e-6f5f-4ebf-af27-3c14373f4dd6)


### Initial Data Inspection (See `Initial_Exploratory_Data_Analysis.ipynb`)
To begin, it was necessary to get familair with the data. Necessary python packages were imported and the data was imported to a Jupyter Notebook. The data was inspected to see what data was present in the file as well as the format of the data in the different cokumns in the file. One main factor was then inspected at this point of the analysis, this was device type and whether that had a correlation on whether or not a user was more likely to churn. 

![image](https://github.com/user-attachments/assets/eb9fd339-dbe8-4efc-aa2f-82ba325d0d2c)

Ultimately, it was found that on average there are more iPhone users that were retained as well as more iPhone users that churned. This can likely be attributed to the fact that there were more iPhone users present in the dataset and thus they were more likely to both churn and be retained. Additionally, some more brief EDA showed that churned users tended to have less driving days, but also tended to have more drives and more distance traveled in these days. This provides some initial insight into what a churned users profile may be like. Churned users may be people who are only using the app on a road trip or vacation or a similar event while not needing the app for their everyday life.

## Analyze
### Exploratory Data Analysis (See `Exploratory_Data_Analysis_and_Viz`)
After an initial screening of the data was complete, it was necessary to do a more in depth screening of the variables that were provided as part of the data set. This in depth screening is referred to as Exploratory Data Analysis (EDA) and involves the following components (if necessary): discovery, structuring, cleaning, joining, validating, and presenting. The main focus of the EDA completed as a part of this project was the creation of visualizations which explained the distribution of variables or the relationship between a pair of variables. Some of the visualizations will be shown below, but due to the plethora of existing visualizations, not all will be shown, please refer to the ipynb file referenced to see all the visualizations.

![image](https://github.com/user-attachments/assets/09f034bd-e907-4006-84d4-66b1edfe8e03)

In the above visualization, the percentage of churned users and retained users is looked at as a function of the number of days the user spent driving in the last month. What can be seen in is that the more days a user spends using the app the less likely they are to churn (40% churn rate with 0 driving days, ~17% churn rate with 15 driving days, and a 0% churn rate with 30 driving days). 

![image](https://github.com/user-attachments/assets/7edf09c4-d9fe-43ff-8d2e-b19980328ba9)

As alluded to earlier, there does not appear to be a discrepancy between device types for whether or not users churn. For both Androids and iPhones there appeared to be about 1 user who churned for every 4 users who were retained.

Additional work was done to create new variables that could be used later for additional analysis. One example is shown below where the `driven_km_drives` column and the `driving_days` column were used get a mean number of kilometers driven per day by each user.

  ```
    # 1. Create `km_per_driving_day` column
    df['km_per_driving_day'] = df['driven_km_drives']/df['driving_days']

    # 2. Call `describe()` on the new column
    df['km_per_driving_day'].describe()
```
All of this EDA work allowed for the early determination of which variables may be best suited for modeling, while eliminating others. For example, there was a relationship shown between the number of users who churned and how many days they spent using the app within a month, but it was also shown that a users device type likely does not have an influence on whether or not a user churns. Addiitonally, the EDA work allowed for the identification and removal of outliers.

### Statistical Testing (See `Data_Exploration_and_Hypothesis_Testing`)
As mentioned in the key questions section above, one area of interest was to determine if there was a relationship between mean amount of rides and device type, specifically a statistical analysis. As there are two samples in this dataset, mean amount of rides by iPhone users and mean amount of rides by Android users, and due to the fact that population statistics are not available, a two-sample t-test is the most appropriate statistical test to determine if there is a difference between the two samples.

To perform the analysis it was first necessary to transform the categorical variable device (iPhone or Android) to an integer using label encoding. The following code block demonstrates how the label encoding was done in Python:

```
# 1. Create `map_dictionary`
map_dictionary = {'iPhone': 1, 'Android': 2}

# 2. Create new `device_type` column
df['device_type'] = df['device'].map(map_dictionary)

# 3. Map the new column to the dictionary
df['device_type']
```

Thus, all iPhone users were assigned the integer 1, while all Android users were assigned the integer 2. The mean for each group was then found using the following code:

```
df.groupby('device_type')['drives'].mean()
```
```
device_type
1    67.859078
2    66.231838
Name: drives, dtype: float64
```

iPhone users tended to have more drives on average but as the difference can be seen to be small, the difference may arise from random sampling and not a true difference between the means of the two samples. To test whether this is a true difference between the mean of the two samples, it is necessary to perform a two-sample t-test. In this case the null and alternative hypotheses are the following:

  $H_0$: There is no difference in the average number of drives between drivers who use iPhone devices and drivers who use Android devices.

  $H_A$: There is a difference in the average number of drives between drivers who use iPhone devices and drivers who use Android devices.

The significance level chosen for the test is 5%. The t-test was performed using the `stats` module from the `Scipy` package in Python, to perform the test the variances of the two samples were assumed to be unequal.

```
# 1. Isolate the `drives` column for iPhone users.
iphone_drives = df[df['device_type']==1]['drives']

# 2. Isolate the `drives` column for Android users.
android_drives = df[df['device_type']==2]['drives']

# 3. Perform the t-test
stats.ttest_ind(a=iphone_drives, b=android_drives, equal_var=False)
```
```
Ttest_indResult(statistic=1.4635232068852353, pvalue=0.1433519726802059)
```

The t-statistic for the test was found to be 1.464 and the p-value was found to be 0.143. Since the p-value (0.143) is greater than our significance level (0.05 or 5%), the null hypothesis should fail to be rejected. There is not a statistically significant difference between the average drives completed by iPhone users and the average drives completed by Android users.

## Construct

After the completion of the EDA and statistical testing, the data was ready for modeling. The first type of modeling that explored in this project was regression modeling. As we are interested in whether or not a user churned, a binary variable, the most applicable regression model to apply to the data is a binomial logistic regression.

### Regression Modeling

#### Model Preparation
The target variable in this case was the varibale labeled `label` in the dataset. This is the variable which stated whether or not a user churned. All other variables, apart from `ID` were assumed to be able to be used as predictor variables. To begin with the binomial logistic regression, it was necessary to complete some further EDA. After the ID column had been dropped from the dataset, it was necessary to check the class balance of the target variable.

```
df['label'].value_counts(normalize=True)
```
```
retained    0.822645
churned     0.177355
Name: label, dtype: float64
```

Here it can be seen that there is approximately an 80/20 split in the target variable, thus, when creating the training and testing data it will be necessary to stratify the data on the target variable to ensure the class balance is maintained. All the variables were then looked at and assessed for outliers. The following columns were found to have outliers: `sessions`, `drives`, `total_sessions`, `total_navigations_fav_1`, `total_navigations_fav2`, `driven_km_drives`, and `duration_minutes_drives.` The outliers in these columns were imputed by setting all values that exceeded the 95th quantile of the column equal to that 95th quantile value.

```
# Impute outliers
for column in ['sessions', 'drives', 'total_sessions', 'total_navigations_fav1', 'total_navigations_fav2', 'driven_km_drives', 'duration_minutes_drives']:
    threshold = df[column].quantile(0.95)
    df.loc[df[column] > threshold, column] = threshold
```

There will be two variables created for this analysis. The first variable is `km_per_driving_day` which was discussed above. This variable was shown to correlate with churn rate as shown during the earlier EDA.

```
# 1. Create `km_per_driving_day` column
df['km_per_driving_day'] = df['driven_km_drives']/df['driving_days']

# 2. Call `describe()` on the new column
df['km_per_driving_day'].describe()
```
```
count    1.499900e+04
mean              inf
std               NaN
min      3.022063e+00
25%      1.672804e+02
50%      3.231459e+02
75%      7.579257e+02
max               inf
Name: km_per_driving_day, dtype: float64
```

Here it can be seen that there are infinite values that exist in the newly formed column, to deal with this, all infinity values were set to 0.

```
# 1. Convert infinite values to zero
df.loc[df['km_per_driving_day'] == np.inf, 'km_per_driving_day'] = 0

# 2. Confirm that it worked
df['km_per_driving_day'].describe()
```
```
count    14999.000000
mean       578.963113
std       1030.094384
min          0.000000
25%        136.238895
50%        272.889272
75%        558.686918
max      15420.234110
Name: km_per_driving_day, dtype: float64
```
The replacement of the inifinty values with 0 fixed the distrubution of the data in the column. The second variable that was created was a `professional_driver` variable. This variable was defined as users who had 60 or more drives in the last month and drove on 15 or more days. The goal here is to separate individuals who might be completing drives for work related reasons from individuals who are completing drives casually.

```
# Create `professional_driver` column
df['professional_driver'] = np.where((df['drives'] > 59) & (df['driving_days'] > 14), 1, 0)
```
```
# 1. Check count of professionals and non-professionals
print(df['professional_driver'].value_counts())

# 2. Check in-class churn rate
print(df.groupby(['professional_driver'])['label'].value_counts(normalize=True))
```
```
0    12405
1     2594
Name: professional_driver, dtype: int64
professional_driver  label   
0                    retained    0.801202
                     churned     0.198798
1                    retained    0.924437
                     churned     0.075563
Name: label, dtype: float64
```
Here we can see that the churn rate for professional drivers (7.6%) is lower than the churn rate for non-professional drivers (19.9%) indicative of a potential predictive signal. The next step for creating the binomial logistic regression model is to encode the categorical variables, specifically the `label` variable. To do this, we will set all churned users equal to 1 and all reatined users equal to 0.

```
# Create binary `label2` column
df['label2'] = np.where(df['label'] == 'churned', 1, 0)
df[['label', 'label2']].tail()
```
This allows the variable to be used for modeling. The final step before creating the model was to check all model assumptions. As a reminder these are the assumptions in place for a logistic regression model:
* Independent Observations
* No extreme outliers
* Little to no multicollinearity among X predictors
* Linear relationship between X and the logit of y

For this project, all observations are assumed to be independent. Extreme outliers have already been dealt with. Whether or not there is a linear relationship between X and the logit of y will be explored and verified after the model has been created. Thus, the remaining assumption to verify is little to no multicollinearity among X predictors. To do this, a correlation matrix will first be genmerated and a correlation heatmap will be plotted.

```
# Generate a correlation matrix
df.corr(method='pearson')
```
```
# Plot correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(method='pearson'), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title("Correlation heatmap", fontsize = 18)
plt.show()
```

![image](https://github.com/user-attachments/assets/9252d45b-747b-4686-8197-abcc81b2e04b)

Using 0.7 as the multicollinearity threshold shows that `sessions` and `drives` are multicolliear as well as `driving_days` and `activity_days.` The variables `drives` and `activity_days` can be observed to have slightly stronger correlations with the target variable compared with the other variables, thus, `sessions` and `driving_days` will be dropped from the analysis. The final task to complete before modeling is to create dummy variables for any categorical predictor variables. There is only one categorical predictor variable, `device`, which was already shown to likely not have an effect on the predictor variable but will be withheld for the robustness of analysis.

```
# Create new `device2` variable
df['device2'] = np.where(df['device'] == 'iPhone', 1, 0)
df[['device', 'device2']].tail()
```

iPhone users were set equal to 1, while Android users were set equal to 0.

#### Model Construction

To begin with model construction, we first isolate our predictor and target variables.

```
# Isolate predictor variables
X = df.drop(columns = ['label', 'label2', 'device', 'sessions', 'driving_days'])
```
```
# Isolate target variable
y = df['label2']
```

Next, we will split the data into training and testing sets, as mentioned earlier, there is a class imbalance that exists in the target variable and thus it it necessary to set the parameter `stratify` equal to y to maintain the class balance.

```
# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
```
Now that thjis is complete, the model is ready to be created and fitted using the training data. Since the predictor variables are unscaled, it is necessary to set the `penalty` parameter to none.

```
model = LogisticRegression(penalty='none', max_iter=400)

model.fit(X_train, y_train)
```
The coefficients of the model were found to be the following:

```
array([[ 1.91336945e-03,  3.27070888e-04, -4.06477637e-04,
         1.23175474e-03,  9.31478651e-04, -1.48604534e-05,
         1.09093436e-04, -1.06031965e-01,  1.82230940e-05,
        -1.52850416e-03, -1.04121752e-03]])
```
The intercept was found to the following:

```
array([-0.00170675])
```
The next step is to circle back and check the final logistic regression assumption. To do this, we must first create a variable which outlines the probability of response for each sample in the training data, meaning the probabiliy that each user has for churning.

```
# Get the predicted probabilities of the training data
training_probabilities = model.predict_proba(X_train)
training_probabilities
```
Next, we will find the logit of the probability data using the following formula, reminder the relationship of the logit of the probability values should be linear with the predictor varaible:
<br>
logit(p) = ln(\frac{p}{1-p})
<br>

```
# 1. Copy the `X_train` dataframe and assign to `logit_data`

logit_data = X_train.copy()

# 2. Create a new `logit` column in the `logit_data` df

logit_data['logit'] = [np.log(prob[1]/prob[0]) for prob in training_probabilities]
```
Now that we have the logit data, we can create a regplot to see if the relation between a single predictor variable and log-odds of the predicted probabilities is linear (Note: we are only doing this for one predictor variable when in reality it should be done for all predictor variables).

```
# Plot regplot of `activity_days` log-odds
sns.regplot(x = "activity_days", y = "logit", data = logit_data, scatter_kws={'s':2, 'alpha': 0.5})
plt.title('Log-odds vs. Activity Days')
plt.show()
```
![image](https://github.com/user-attachments/assets/8bde8991-c4d6-4551-a84b-1fbdfaf4cbd7)

Here we can see that a linear relationship does in fact exist, validating the final assumption of the model. The model has now been created, the final step is to interpret the model results (See Regression Modeling Results).


## Execute

### Regression Modeling Results
To evaluate the performance of the binomial logistic regression model, predictions will be made on `X_test` data generated earlier.

```
# Generate predictions on X_test
y_pred = model.predict(X_test)
```

The accuracy of the model is then scrutinized.

```
# Score the model (accuracy) on the test data
model.score(X_test, y_test)
```
```
0.8237762237762237
```
The model was found to have an accuracy of 82.4%, indicating that the model performs pretty well overall in classifying churned and retained users but there is further analysis that must be completed. To gain a better understanding of how the model is predicting a confusion matrix will be created and displayed.

```
cm = metrics.confusion_matrix(y_test, y_pred, labels=model.classes_)
```
```
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=model.classes_)

disp.plot()
```

![image](https://github.com/user-attachments/assets/0dfaf17b-2566-43cc-8d2d-a90709124c7d)

Here we can see that the model has a lot of False negatives compared with false positives. This means that the model is more likely to say a user was retained when in fact the user churned. This is likely not ideal for a model that is interested in predicting whether a user will churn or not. The precision and recall scors of the model will now be scrutinized. Based on the high number of false negatives, it is likely that a lower recall score than precision score will be seen.

```
# Calculate precision manually
precision = cm[1,1] / (cm[0, 1] + cm[1, 1])
precision
```
```
0.5178571428571429
```
```
# Calculate recall manually
recall = cm[1,1] / (cm[1, 0] + cm[1, 1])
recall
```
```
0.0914826498422713
```
The model was in fact found to have a much higher precision score (0.52) than a recall score (0.09) meaning that the model makes a lot of false negative predictions and fails to capture users who will churn.

### Other Modeling Results
