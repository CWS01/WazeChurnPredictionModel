# WazeChurnPredictionModel
How can Waze leverage existing data to minimize user churn rate?

## Plan
### Background
Waze’s free navigation app makes it easier for drivers around the world to get to where they want to go. Waze’s community of map editors, beta testers, translators, partners, and users helps make each drive better and safer. Waze partners with cities, transportation authorities, broadcasters, businesses, and first responders to help as many people as possible travel more efficiently and safely. Recently, Waze management has noticed that there is a nonnegligible number of users who churn, or no longer use, from the Waze app. Due to this finding, Waze management would like to explore the factors that appear to influence whether a user churns, ultimately leading to better user retention and growth for the Waze business.

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

## Analyze
### Initial Data Inspection (See `Initial_Exploratory_Data_Analysis.ipynb`)
To begin, it was necessary to get familair with the data. Necessary python packages were imported and the data was imported to a Jupyter Notebook. The data was inspected to see what data was present in the file as well as the format of the data in the different cokumns in the file. One main factor was then inspected at this point of the analysis, this was device type and whether that had a correlation on whether or not a user was more likely to churn. 

![image](https://github.com/user-attachments/assets/eb9fd339-dbe8-4efc-aa2f-82ba325d0d2c)

Ultimately, it was found that on average there are more iPhone users that were retained as well as more iPhone users that churned. This can likely be attributed to the fact that there were more iPhone users present in the dataset and thus they were more likely to both churn and be retained. Additionally, some more brief EDA showed that churned users tended to have less driving days, but also tended to have more drives and more distance traveled in these days. This provides some initial insight into what a churned users profile may be like. Churned users may be people who are only using the app on a road trip or vacation or a similar event while not needing the app for their everyday life.


## Construct

## Execute
