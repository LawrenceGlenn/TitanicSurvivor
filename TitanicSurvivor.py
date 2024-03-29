
%matplotlib tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


#create training data from csv from kaggle using pandas
data_train = pd.read_csv('Downloads/titanic/train.csv')
#create testing dataset from kaggle csv using pandas
data_test = pd.read_csv('Downloads/titanic/test.csv')


#display a few pieces of data
data_train.sample(3)


#create plots to examine data
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);

#turn ages into usable groups, based on human catigories
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

#exract the letter from the front of the cabin data, and fill in an N for Na
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

#group the fares into quartiles
def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

#split the last name into name and prefix as seperate columns
def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
#remove unwanted features
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

#apply transformations
def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)
data_train.head()

#plot values of some of the transformed data
sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train);

sns.barplot(x="Cabin", y="Survived", hue="Sex", data=data_train);

sns.barplot(x="Fare", y="Survived", hue="Sex", data=data_train);


#function to use LabelEncoder to convert tables into more usable form
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test


#apply the label encoder
data_train, data_test = encode_features(data_train, data_test)
data_train.head()


#split the training data into the survived value and everything else
X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']


#using scikit to shuffle the data using 80% to train and 20% to test
num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


# select random forest as the classifier
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)


predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))


#function to verify efficacy with KFold, selecting 10 buckets and using a different buck to test a new iteration each time
def run_kfold(clf):
	kf = KFold(n_splits=10)
	outcomes = []
	fold = 0
	for train_index, text_index in kf.split(features):
		fold +=1
		X_train, X_test = X_all.values[train_index], X_all.values[test_index]
		y_train, y_test = y_all.values[train_index], y_all.values[test_index]
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		accuracy = accuracy_score(y_test, predictions)
		outcomes.append(accuracy)
		print("Fold {0} accuracy: {1}".format(fold,accuracy))
	mean_outcome = np.mean(outcomes)
	print("Mean Accuracy: {0}".format(mean_outcome))


run_kfold(model)

#predict with the actual test data
ids = data_test['PassengerId']
predictions = clf.predict(data_test.drop('PassengerId', axis=1))


#look at final result
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
# output.to_csv('titanic-predictions.csv', index = False)
output.head()
