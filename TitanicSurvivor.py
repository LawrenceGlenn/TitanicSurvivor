import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline



#create training data from csv from kaggle using pandas
data_train = pd.read_csv('../input/train.csv')
#create testing dataset from kaggle csv using pandas
data_test = pd.read_csv('../input/test.csv')


#display a few pieces of data
data_train.sample(3)


#create plots to examine data
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);

