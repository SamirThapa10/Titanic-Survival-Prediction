# Titanic-Survival-Prediction
It is a simple prediction model which predict whether the passenger is survived or not.


Process in Titanic-Survival-Prediction
1.LOAD TRAIN AND TEST DATA
 a.Removing data with nan in Embarked
 
 b.Selecting Features
 c.Barplot between Sex and Survived
 d.Barplot between Embarked and Survived

2.CONVERTING THE DATA
  a.Get list of categorical variables
  b.Unique string present in Cols
  c.Apply one-hot encoder to each column with categorical data

3.COMPLETING THE MISSING DATA
  a.Identify the missing data
  b.Imputation fills in the missing values with mean value

4.CREATING MODEL
  a.Specifying the Model
  b.Fiting Model
  c.Make validation predictions and calculate mean absolute error
  d.Selecting the best tree size
  e.Write loop to find the ideal tree size from candidate_max_leaf_nodes
  f.Creating model using Best tree size
  g.Making the prediction
  h.Converting the prediction value into categories(i.e 1 or 0)
  i.Exporting predicted values to submission.csv file


