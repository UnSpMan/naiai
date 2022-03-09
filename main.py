from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
cancer = load_breast_cancer()
df_train = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df_target = pd.DataFrame(cancer.target)

X_train,X_test,y_train,y_test = train_test_split(df_train,df_target,test_size=0.3,random_state=0)
from sklearn.tree import export_graphviz
import pydot
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
preds = tree.predict(X_test)
print(preds)
print(type(y_test))
y_test['preds'] = preds
y_test['if_correct'] = y_test.apply(lambda x:x[0] == x['preds'],axis=1)
# print('分类正确的有{}/{}'.format(y_test.loc[y_test['if_correct'] == True].count().preds,len(y_test)))
export_graphviz(tree,out_file="tree.dot",class_names=load_iris.target)
