from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

FILE_PATH = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data'

df = pd.read_csv(FILE_PATH)
df.columns = ['english', 'instructor', 'course',
              'semester', 'size', 'attribute']

y = df['attribute'].values
X = df.drop('attribute', 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# random forests
rf_clf = RandomForestClassifier(n_estimators=25)
rf_clf.fit(X_train, y_train)

# k neighbors
kn_clf = KNeighborsClassifier(n_neighbors=2)
kn_clf.fit(X_train, y_train)

rf_accuracy = rf_clf.score(X_test, y_test)
kn_accuracy = kn_clf.score(X_test, y_test)
print("Random Forest accuracy: {:.3f}".format(rf_accuracy))
print("K Neighbors accuracy: {:.3f}".format(kn_accuracy))
