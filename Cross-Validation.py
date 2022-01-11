from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Data Preprocessing
df = pd.read_csv('datasets/IRIS.csv')
df = pd.DataFrame(df)

# Data frame splitting
X = df.drop(columns=['species'])
Y = df['species']
le_species = LabelEncoder()
Y = le_species.fit_transform(Y)

# Model Declaration
model1 = SVC()
model2 = DecisionTreeClassifier()
model3 = RandomForestClassifier()


# Score Testing
def getScore(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


# K Fold testing
kf = KFold(n_splits=4)
for train_index, test_index in kf.split(X):
    # print(train_index)
    # print(X.shape)

    X_Train, X_Test, Y_Train, Y_Test = X[train_index, :], X[test_index, :], Y[train_index], Y[test_index]
    print(getScore(model1, X_Train, X_Test, Y_Train, Y_Test))
    print(getScore(model2, X_Train, X_Test, Y_Train, Y_Test))
    print(getScore(model3, X_Train, X_Test, Y_Train, Y_Test))
