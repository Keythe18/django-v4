# -------------------------------------------------------

print(f'step 1 - imports')

# Import all dependencies
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import constants
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


print() # add return into logging
# -------------------------------------------------------

print(f'step 2 - Load Dataset')

# Load dataset
df = pd.read_csv("weight-height.csv")
print(df.head())

print(f'--- --- ---')


print() # add return into logging
# -------------------------------------------------------

print(f'step 3 - Transform US properties into EU ')

# Transform US properties into EU 
df['Weight'] = np.around(df['Weight'] * constants.pound, 1)
df['Height'] = np.around(df['Height'] * constants.inch * 100)
df['Height'] = df['Height'].astype(np.int64, errors='ignore') 
print(df.head())

print(f'--- --- ---')


print() # add return into logging
# -------------------------------------------------------

print(f'step 4 - Check if Genders is Balanced')

genders = df['Gender'].value_counts()
print(genders)

print(f'--- --- ---')


print() # add return into logging
# -------------------------------------------------------


print(f'step 5 - Plot Genders depending on Weight and Height')

sns.scatterplot(x='Height', y='Weight', data=df, hue='Gender')
# import matplotlib.pyplot as plt
# plt.show()
print(f'--- --- ---')


print() # add return into logging
# -------------------------------------------------------


print(f'step 6 - Replace string gender Male by Int:0 & Female by Int:1')

df.Gender = df.Gender.map({"Male" : 0, "Female" : 1})
print(df.sample(n=10))

print(f'--- --- ---')


print() # add return into logging
# -------------------------------------------------------

print(f'step 7 - Train model with Machine Learning Patterns')

X = df[ ["Gender", "Height"] ] # definir input sur l'axe X 
y = df[ ["Weight"] ] # definir output sur l'axe Y
# Trouver des correlations entre X et Y afin de definir des modeles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Entrainer le dataset avec un modele de regression lineaire.
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(lin_reg.score(X_test, y_test))

print(f'--- --- ---')


print() # add return into logging
# -------------------------------------------------------


print(f'step 8 - After the model is trained we can query')
# Donne moi le poids moyen d'un male de 1.80M
test = np.round(lin_reg.predict([[0, 180]])[0][0],1)
# Print le poids retournee
print(test)

print(f'--- --- ---')


print() # add return into logging
# -------------------------------------------------------


print(f'step 9 - Save the trained modele that can be used later')

# Save the model
joblib_file = "WeightPredictionLinRegModel.joblib"
joblib.dump(lin_reg, joblib_file) 

print(f'--- --- ---')

print() # add return into logging
# -------------------------------------------------------
