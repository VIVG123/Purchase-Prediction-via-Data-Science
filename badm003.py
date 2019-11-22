import pandas as pd
from sklearn import preprocessing, model_selection, neighbors

# dfv = pd.read_excel('Downloads/RetailSales Train.xlsx')
dft = pd.read_excel('../Downloads/RetailSales Validate.xlsx')
dft = dft[dft['Status']=='Delivered']
dft = dft[dft['ItemID']!='POST']
dft['InvoiceDate'] = dft['InvoiceDate'].apply(pd.to_datetime)
dft['month'] = dft.InvoiceDate.dt.month
dft['date'] = dft.InvoiceDate.dt.day
dft['year'] = dft.InvoiceDate.dt.year
dft['hour'] = dft.InvoiceDate.dt.hour
dft = dft[dft['Quantity']>0]
dft.reset_index()
dft['hour'].unique()
def hour_to_factor(x):
    m = 5
#     print(m)
    for i in range(1,m+1):
        if x<i*(24/m):
            return i
    return m
dft['hour'] = dft['hour'].apply(hour_to_factor)

c = 0
item = {}
for i in (dft['ItemID'].unique()):
    item[i] = c
    c+=1
dft['ItemID'] = list(map(lambda x: item[x],dft['ItemID']))
X = dft[['CustomerID']]
y = dft[['hour']]
# X,y?                dft['hour'].loc[i] = j
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
X = preprocessing.scale(X)
y = np.array(y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
clf = neighbors.KNeighborsClassifier()
len(X_train)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
# print(X_train,y_train)
# print(X_test,y_test)
print(confidence)