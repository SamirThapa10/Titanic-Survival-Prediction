# Titanic-Survival-Prediction
It is a simple prediction model which predict whether the passenger is survived or not.


# Process in Titanic-Survival-Prediction
# LOAD TRAIN AND TEST DATA
    Removing data with nan in Embarked 
    Selecting Features
    Barplot between Sex and Survived
    Barplot between Embarked and Survived

# CONVERTING THE DATA
    Get list of categorical variables
    Unique string present in Cols
    Apply one-hot encoder to each column with categorical data

# COMPLETING THE MISSING DATA 
    Identify the missing data
    Imputation fills in the missing values with mean value

# CREATING MODEL
    Specifying the Model
    Fiting Model
    Make validation predictions and calculate mean absolute error
    Selecting the best tree size
    Write loop to find the ideal tree size from candidate_max_leaf_nodes
    Creating model using Best tree size
    Making the prediction
    Converting the prediction value into categories(i.e 1 or 0)
    Exporting predicted values to submission.csv file


