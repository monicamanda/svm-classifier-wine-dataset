import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

test_size = 0.2

# Red Wine
datasetR = pd.read_csv('winequality-red.csv', sep=';') 

X = datasetR.iloc[:, :11]
Y = datasetR.iloc[:, 11:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=False)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

model = SVC(kernel='linear')
model.fit(x_train, y_train.values.ravel())
predict = model.predict(x_test)

print('\n-RED WINE')
print(' accuracy: ', round((accuracy_score(predict, y_test)*100), 2), '%')
print('\n', classification_report(y_test, predict))

# White Wine
datasetW = pd.read_csv('winequality-white.csv', sep=';') 

X = datasetW.iloc[:, :11]
Y = datasetW.iloc[:, 11:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=False)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

model = SVC(kernel='linear')
model.fit(x_train, y_train.values.ravel())
predict = model.predict(x_test)

print('\n-WHITE WINE')
print(' accuracy: ', round((accuracy_score(predict, y_test)*100), 2), '%')
print('\n', classification_report(y_test, predict))
