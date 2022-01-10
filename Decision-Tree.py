import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Reading Data
df = pd.read_csv('datasets/titanic.csv')
Y = df['Survived']
X = df.drop(columns=['PassengerId', 'Survived', 'Name', 'SibSp',
                     'Parch', 'Ticket', 'Cabin', 'Embarked'])

# Data Preprocessing
le_sex = LabelEncoder()
X['Sex_n'] = le_sex.fit_transform(X['Sex'])
X['Age'] = X['Age'].fillna(X['Age'].median())
X = X.drop(columns=['Sex'])
X_Train, X_Test, Y_Train , Y_Test = train_test_split(X, Y, )

# Model Instance
model = tree.DecisionTreeClassifier()
model.fit(X_Train, Y_Train)

# Model Score
score = model.score(X_Test, Y_Test)
print(score)








