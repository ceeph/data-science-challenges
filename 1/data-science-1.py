from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

FILE_PATH = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data'

df = pd.read_csv(FILE_PATH)
df.columns = ['english', 'instructor', 'course',
              'semester', 'size', 'attribute']

y = df['attribute']
X = df.drop('attribute', 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

rf_clf = RandomForestClassifier(n_estimators=25)
rf_clf.fit(X_train, y_train)
rf_accuracy = rf_clf.score(X_test, y_test)
print("Random Forest accuracy: {:.3f}".format(rf_accuracy))
