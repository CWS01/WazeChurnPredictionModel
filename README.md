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

## Construct

## Execute
