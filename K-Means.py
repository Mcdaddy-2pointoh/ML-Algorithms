from sklearn.cluster import k_means
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Reading Dataset
data = pd.read_csv('datasets/IRIS.csv')
df = pd.DataFrame(data)

# Processing Dataset
X = df.drop(columns=['species'])
Y = df['species']
le_species = LabelEncoder()
Y = le_species.fit_transform(Y)

# Model Creation
Y_pred = k_means(n_clusters=3, X = X)
print(Y_pred[1])





