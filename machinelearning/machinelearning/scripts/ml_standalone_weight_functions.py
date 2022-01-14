import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import constants
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def loadDataset():
    df = pd.read_csv("weight-height.csv")
    print(df.head())
    return df

def transformUsPropertiesIntoEu(df):
    df['Weight'] = np.around(df['Weight'] * constants.pound, 1)
    df['Height'] = np.around(df['Height'] * constants.inch * 100)
    df['Height'] = df['Height'].astype(np.int64, errors='ignore') 
    print(df.head())
    return df

def checkIfGendersIsBalanced(df):
    genders = df['Gender'].value_counts()
    print(genders)

def plotGendersDependingOnWeightAndHeight(df):
    sns.scatterplot(x='Height', y='Weight', data=df, hue='Gender')
    # import matplotlib.pyplot as plt
    # plt.show()
    return df

def replaceStringGenderMaleFemale(df):
    df.Gender = df.Gender.map({"Male" : 0, "Female" : 1})
    print(df.sample(n=10))
    return df

def trainModelWithMachineLearningPatterns(df):
    X = df[ ["Gender", "Height"] ] # definir input sur l'axe X 
    y = df[ ["Weight"] ] # definir output sur l'axe Y
    # Trouver des correlations entre X et Y afin de definir des modeles
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Entrainer le dataset avec un modele de regression lineaire.
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    print(lin_reg.score(X_test, y_test))
    return lin_reg

def queryAfterTheModelIsTrained(lin_reg, gender, height):
    """gender : 0 for male, 1 for female"""
    test = np.round(lin_reg.predict([[gender, height]])[0][0],1)
    print(test)

def saveTrainedModel(lin_reg):
    #arg name ?
    name = "WeightPredictionLinRegModel.joblib"
    joblib.dump(lin_reg, name)

