from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Dataset to a dataframe
df = pd.read_csv('datasets/IRIS.csv')
df = pd.DataFrame(df)

# Data frame splitting
X = df.drop(columns=['species'])
Y = df['species']
le_species = LabelEncoder()
Y = le_species.fit_transform(Y)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y)

# Model Creation
model = SVC(C=10)
model.fit(X_Train, Y_Train)

# Model Score
score = model.score(X_Test, Y_Test)
print(score)