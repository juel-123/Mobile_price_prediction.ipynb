# Mobile_price_prediction.ipynb

Problem Statement
Now a days mobile phones are very much neccessary to the indiviuals with best specification.

Mobile phone come in all sort of prices, features ,specifications and all.price estimation and prediction is an important part of consumer strategy. Deciding on the correct price of the product is very important for the market success of a product. A new prooduct that has to be launched must have the correct price so that consumers find it appropriate to buy the product.

In the competitive mobile phone market companies want to understand the sales data of mobile phones and factors that drives the prices.

The main objective of this project is to build a model which will classify the price range of the mobile phones based on the specification of the mobile phone






Data description
The given features description:

battery_power :Battery acpacvity in mAh
blue : Has Bluetooth or not
clock_speed : Speed at which microprocessor executes instructions
dual_sim :Has dual sim support or not
fc :Front camera megapixel
four_g : Has 4G or not
int_memory :Internal memory capacity
m_dep : Mobile depth in cm
mobile_wt :Weight of mobile phones
n_cores : Nuber of cores in processor
pc :Primary camera in megapixels
px_height :Pixel resolution height
px_width : Pixel resolution width
ram :Random access memory in MB
sc_h :screen height
sc_w :screen width
talk_time : Longest that a single memory last over a call
three_g :Has 3G or not
touch_screen : Touch screen or not
wifi :Has wifi or not
price_range :This is the target variable with value 0 (low cost),1(medium cost),2(high cost),3(very high cost)





In case of pixel resolution height(px_height): The price_range(0) has the lowest px_height and the price_range(3) has highest px_height.

In case of pixel resolution width(px_width): The price_range(0) has the lowest px_width and the price_range(3) has highest px_width.



Mobile of low cost supporting dual sim has less internal memory storage and mobile of very high cost supporting dual sim has more internal storage memory.
Mobile of high cost that not supporting dual sim has less internal memory storage and mobile of very high cost that not supporting dual sim has more internal storage memory than the mobile supporting dual sim


Observation of Decision Tree Classifier

before tunning

Trainning accuracy= 100%
test accuracy = 85%
The model is overfitted the training data and does not generalised so we tuned.

After tunning

Trainning accuracy= 95%
test accuracy = 83%
In general, achieving 100% training accuracy is often a sign of overfitting. The goal of hyperparameter tuning is to strike a balance between training accuracy and test accuracy by finding the optimal set of hyperparameters that prevent overfitting and lead to better generalization. While the test accuracy decreased slightly, the decrease in training accuracy suggests that the tuning process is helping the model to better generalize, which is a step in the right direction.

Observation of Random forest

before tuning

Training accuracy= 100%
test accuracy = 87%
The model is overfitted the training data and does not generalised so we tuned.

After tuning

Training accuracy= 96%

test accuracy = 85%

The drop in training accuracy and the modest improvement in test accuracy after tuning indicate that the model is likely to perform better on unseen data and that the tuning process is helping to address overfitting.



Project Title : Mobile price Prediction
Project type :Classification

Contribution: Indiviual

Github link :https://github.com/juel-123/Mobile_price_prediction.ipynb/blob/main/Mobile_price_prediction.ipynb
Problem Statement
Now a days mobile phones are very much neccessary to the indiviuals with best specification.

Mobile phone come in all sort of prices, features ,specifications and all.price estimation and prediction is an important part of consumer strategy. Deciding on the correct price of the product is very important for the market success of a product. A new prooduct that has to be launched must have the correct price so that consumers find it appropriate to buy the product.

In the competitive mobile phone market companies want to understand the sales data of mobile phones and factors that drives the prices.

The main objective of this project is to build a model which will classify the price range of the mobile phones based on the specification of the mobile phones.

Importing libraries
[ ]
# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report,precision_score,recall_score,f1_score,roc_curve,auc, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
Importing Dataset
[ ]
#importing the dataset
df = pd.read_csv('/content/drive/MyDrive/data_mobile_price_range.csv')
[ ]
# head of the dataset
df.head()

[ ]
# tail of the dataset
df.tail()

[ ]
# shape of the dataset
df.shape
(2000, 21)
There are 2000 rows and 21 columns

[ ]
# information of the dataset
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 21 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   battery_power  2000 non-null   int64  
 1   blue           2000 non-null   int64  
 2   clock_speed    2000 non-null   float64
 3   dual_sim       2000 non-null   int64  
 4   fc             2000 non-null   int64  
 5   four_g         2000 non-null   int64  
 6   int_memory     2000 non-null   int64  
 7   m_dep          2000 non-null   float64
 8   mobile_wt      2000 non-null   int64  
 9   n_cores        2000 non-null   int64  
 10  pc             2000 non-null   int64  
 11  px_height      2000 non-null   int64  
 12  px_width       2000 non-null   int64  
 13  ram            2000 non-null   int64  
 14  sc_h           2000 non-null   int64  
 15  sc_w           2000 non-null   int64  
 16  talk_time      2000 non-null   int64  
 17  three_g        2000 non-null   int64  
 18  touch_screen   2000 non-null   int64  
 19  wifi           2000 non-null   int64  
 20  price_range    2000 non-null   int64  
dtypes: float64(2), int64(19)
memory usage: 328.2 KB
Data description
The given features description:

battery_power :Battery acpacvity in mAh
blue : Has Bluetooth or not
clock_speed : Speed at which microprocessor executes instructions
dual_sim :Has dual sim support or not
fc :Front camera megapixel
four_g : Has 4G or not
int_memory :Internal memory capacity
m_dep : Mobile depth in cm
mobile_wt :Weight of mobile phones
n_cores : Nuber of cores in processor
pc :Primary camera in megapixels
px_height :Pixel resolution height
px_width : Pixel resolution width
ram :Random access memory in MB
sc_h :screen height
sc_w :screen width
talk_time : Longest that a single memory last over a call
three_g :Has 3G or not
touch_screen : Touch screen or not
wifi :Has wifi or not
price_range :This is the target variable with value 0 (low cost),1(medium cost),2(high cost),3(very high cost)
Checking whether the data is clean or not
[ ]
# checking the duplicate data
df[df.duplicated].sum()
battery_power    0.0
blue             0.0
clock_speed      0.0
dual_sim         0.0
fc               0.0
four_g           0.0
int_memory       0.0
m_dep            0.0
mobile_wt        0.0
n_cores          0.0
pc               0.0
px_height        0.0
px_width         0.0
ram              0.0
sc_h             0.0
sc_w             0.0
talk_time        0.0
three_g          0.0
touch_screen     0.0
wifi             0.0
price_range      0.0
dtype: float64
There are no duplicated values in the dataset

[ ]
# checking the null values
df.isnull().sum()
battery_power    0
blue             0
clock_speed      0
dual_sim         0
fc               0
four_g           0
int_memory       0
m_dep            0
mobile_wt        0
n_cores          0
pc               0
px_height        0
px_width         0
ram              0
sc_h             0
sc_w             0
talk_time        0
three_g          0
touch_screen     0
wifi             0
price_range      0
dtype: int64
There are no null values in the Dataset

Statistical Information
[ ]
# getting statistical  information of the features
df.describe(include = 'all')

[ ]
#  Let's see the correlation of each feature
correlation =  df.corr()
plt.figure(figsize=(17,16))
sns.heatmap(abs(correlation),annot=True)
plt.show()

Here after seeing Heat map,The correlation of independent features are not that highly correlated with each other which would lead to multicollinearity, which can affect the performance and interpretability of our model.

[ ]
#  columns in the dataset
df.columns
Index(['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi', 'price_range'],
      dtype='object')
[ ]
#no.of unique features
for i in df.columns:
  unique_count = df[i].nunique()
  print(f"The number of items in '{i}':{unique_count}")
The number of items in 'battery_power':1094
The number of items in 'blue':2
The number of items in 'clock_speed':26
The number of items in 'dual_sim':2
The number of items in 'fc':20
The number of items in 'four_g':2
The number of items in 'int_memory':63
The number of items in 'm_dep':10
The number of items in 'mobile_wt':121
The number of items in 'n_cores':8
The number of items in 'pc':21
The number of items in 'px_height':1137
The number of items in 'px_width':1109
The number of items in 'ram':1562
The number of items in 'sc_h':15
The number of items in 'sc_w':19
The number of items in 'talk_time':19
The number of items in 'three_g':2
The number of items in 'touch_screen':2
The number of items in 'wifi':2
The number of items in 'price_range':4
Visualization of Dependent variable
[ ]
# dependent feature
dependent_feature=['price_range']
[ ]
df['price_range'].value_counts().plot.pie(explode = [0.05,0.05,0.05,0.05],autopct= '%d%%', startangle= 90, shadow = True, figsize=(8,8))
plt.title('Pie chart for price_range')
plt.show()

[ ]
x = pd.DataFrame(df.groupby('price_range').count())
x

The count of each item in 'price_range' is 500

Relation between price_range and battery_power
[ ]
# visualizing price_range and battery power
plt.figure(figsize=(8,7))
sns.barplot(x= df['price_range'] ,y= df['battery_power'], data=df)
plt.xlabel('price_range')
plt.ylabel('battery_power')
plt.title('Relation between price_range and battery_power')
plt.show()

From the visualization we can infer that price_range having (0) has the lowest battery life and the price_range having (3) has the highest battery life.

Relation between px_width,px_height and price_range
[ ]
# visualizing Relation between px_width,px_height and price_range
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
sns.barplot(x= 'price_range',y='px_height',data  =df,palette='Reds')
plt.xlabel('price_range')
plt.ylabel('px_height')
plt.title('Relation between price_range and px_height')
plt.subplot(1,2,2)
sns.barplot(x= 'price_range',y='px_width',data=df,palette='Blues')
plt.xlabel('price_range')
plt.ylabel('px_width')
plt.title('Relation between price_range and px_width')
plt.tight_layout()
plt.show()

In case of pixel resolution height(px_height): The price_range(0) has the lowest px_height and the price_range(3) has highest px_height.

In case of pixel resolution width(px_width): The price_range(0) has the lowest px_width and the price_range(3) has highest px_width.

Relation between price_range and ram
[ ]
#visualizing relation between price_range and ram
plt.figure(figsize=(8,7))
sns.barplot(x= df['price_range'] ,y= df['ram'], data=df)
plt.xlabel('price_range')
plt.ylabel('ram')
plt.title('Relation between price_range and ram')
plt.show()

From the above visualization we can infer that as the price_range is increasing the ram also increasing.

Relation between price_range and 3G/4G
[ ]
#visualizing three_g column
plt.figure(figsize= (8,6))
sns.countplot(x=df['three_g'], hue = df['price_range'],palette='pink')
plt.show()

From the visualization we can infer that

Low cost price_range mobiles are more that are not having three_g feature
High cost price_range mobiles are more that are having three_g feature
[ ]
#visualizing four_g column
plt.figure(figsize= (8,8))
sns.countplot(x=df['four_g'], hue = df['price_range'],palette='Greens')
plt.show()

From the visualization we can infer that

High cost price_range mobiles are more that are not having four_g feature
Very High cost price_range mobiles are more that are having four_g feature
Relation between price_range , dual_sim and internal memory
[ ]
#Relation between price_range , dual_sim and internal memory
plt.figure(figsize=(6,5))
sns.lineplot(x=  'price_range',y = 'int_memory',hue='dual_sim', data=df)
plt.title('Relation between price_range , dual_sim and internal memory')
plt.show()

From the above visualization we can infer that:

Mobile of low cost supporting dual sim has less internal memory storage and mobile of very high cost supporting dual sim has more internal storage memory.
Mobile of high cost that not supporting dual sim has less internal memory storage and mobile of very high cost that not supporting dual sim has more internal storage memory than the mobile supporting dual sim
Relation between primary camera megapixel and front camera megapixel
[ ]
# Relation between pc and fc
plt.figure(figsize=(7,6))
sns.scatterplot(x = 'pc',y='fc',data =df)
plt.title('Relation between pc and fc')
plt.show()

From the above visualization we can infer that as primary megapixel increases the front camera megapixel also increases.

[ ]
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
sns.lineplot(x = 'price_range',y='fc',data =df)

plt.subplot(1,2,2)
sns.lineplot(x = 'price_range',y='pc',data =df)
plt.suptitle('Relation between price_range , pc and fc')
plt.tight_layout()
plt.show()

From the above visualization we can infer that:

Mobile of high price range has the highest front camera megapixels.
Mobile of very high price range has the highest primary camera megapixels.
low price range mobile have both less pc and less fc
[ ]
plt.figure(figsize=(6,5))
sns.lineplot(x=  'price_range',y = 'ram',hue='touch_screen', data=df)
plt.title('Relation between price_range , ram and touch_screen')
plt.show()

From the above visualization we can infer that as the price_range increases the ram also increases in case of both mobile having touch_screen or not.

Detecting outliers
[ ]
# Outliers
num_var = ['battery_power','clock_speed','fc', 'int_memory','m_dep', 'mobile_wt','n_cores', 'pc', 'px_height','px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
[ ]
# plotting the boxplot and distribution
for i in num_var:
  plt.figure(figsize=(15,6))
  plt.subplot(1,2,1)
  fig= sns.boxplot(df[i], color = 'red')
  fig.set_title('')
  fig.set_ylabel(i)


  plt.subplot(1,2,2)
  fig = sns.distplot(df[i], color='red')
  fig.set_xlabel(i)
  plt.show()

From the above visulaization we can see that px_height and fc has outlier and we need to fix it and the data is well distributed.

Treatment of outliers
[ ]
# treatment of outliers
Q1 = df['fc'].quantile(0.25)
Q3 = df['fc'].quantile(0.991)
IQR = Q3 -Q1
df = df[(df['fc'] <= Q3)]
[ ]
Q1 = df['px_height'].quantile(0.25)
Q3 = df['px_height'].quantile(0.991)
IQR = Q3 -Q1
df = df[(df['px_height'] <= Q3)]
[ ]
# visualizing whether utliers are removed or not
for i in ['fc','px_height']:
  plt.figure(figsize=(15,6))
  plt.subplot(1,2,1)
  fig= sns.boxplot(df[i], color = 'green')
  fig.set_title('')
  fig.set_ylabel(i)


  plt.subplot(1,2,2)
  fig= sns.distplot(df[i], color = 'green')
  fig.set_xlabel(i)
  plt.show()


As we can see from above visualization that there no outlier after fixing it.

Data Preprocessing
[ ]
x= df.drop(['price_range'],axis = 1)
y= df['price_range']
[ ]
# splitting into x_train , y_train, x_test, y_test
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
[ ]
# standardizing the data points
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x_train_scaled = std.fit_transform(x_train)
x_test_scaled= std.transform(x_test)
ML Models
Decision Tree Classifier
[ ]
# importing library
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state= 0)
[ ]
# fitting/ training the train set
tree.fit(x_train,y_train)

[ ]
# predicting the values
y_train_pred = tree.predict(x_train)
y_test_pred = tree.predict(x_test)
[ ]
# checking the train set accuracy
accuracy_score(y_train_pred,y_train)
1.0
[ ]
# checking the testset accuracy
accuracy_score(y_test_pred,y_test)
0.8575063613231552
[ ]
# confusion matrix for the test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[80, 11,  0,  0],
       [ 7, 82,  8,  0],
       [ 0, 12, 79, 10],
       [ 0,  0,  8, 96]])
[ ]
# classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       403
           1       1.00      1.00      1.00       395
           2       1.00      1.00      1.00       390
           3       1.00      1.00      1.00       383

    accuracy                           1.00      1571
   macro avg       1.00      1.00      1.00      1571
weighted avg       1.00      1.00      1.00      1571

let's tune some hyperparameters of Decision tree classifier

[ ]
tree = DecisionTreeClassifier(random_state = 0)
[ ]
parameters = {'criterion':['gini','entropy'],'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15],'splitter':['best','ramdom'],'min_samples_split':[3,5,10],'max_features':['auto','sqrt','log2',None]}
[ ]
# fitting model with hyperparameter
from sklearn. model_selection import GridSearchCV
tree_tune = GridSearchCV(tree,parameters,cv=5,scoring = 'accuracy',verbose= 3)
tree_tune.fit(x_train,y_train)

[ ]
print(tree_tune.best_params_)
{'criterion': 'entropy', 'max_depth': 7, 'max_features': None, 'min_samples_split': 5, 'splitter': 'best'}
[ ]
# using best parameter trainning the the data
tree_best_p = DecisionTreeClassifier(criterion= 'entropy', max_depth= 7,  min_samples_split= 5, random_state = 0)
tree_best_p.fit(x_train,y_train)

[ ]
#predicting y values on test and train data.
y_train_pred = tree_best_p.predict(x_train)
y_test_pred = tree_best_p.predict(x_test)
[ ]
# checking the train set accuracy
accuracy_score(y_train_pred,y_train)
0.9503500954805856
[ ]
# checking the test set accuracy
accuracy_score(y_test_pred,y_test)
0.8396946564885496
[ ]
# confusion matrix for the test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[76, 15,  0,  0],
       [ 6, 85,  6,  0],
       [ 0, 14, 80,  7],
       [ 0,  0, 15, 89]])
[ ]
# classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       0.99      0.98      0.98       403
           1       0.93      0.97      0.95       395
           2       0.91      0.93      0.92       390
           3       0.99      0.92      0.95       383

    accuracy                           0.95      1571
   macro avg       0.95      0.95      0.95      1571
weighted avg       0.95      0.95      0.95      1571

[ ]
#feature importance
def plot_feature_importance(algo, feature_names):
    importances = algo.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(x.shape[1]), importances[indices], align="center")
    plt.xticks(range(x.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, x.shape[1]])
    plt.tight_layout()
    plt.show()
[ ]
# Call the function to plot feature importance
plot_feature_importance(tree_best_p, x.columns)

Observation of Decision Tree Classifier

before tunning

Trainning accuracy= 100%
test accuracy = 85%
The model is overfitted the training data and does not generalised so we tuned.

After tunning

Trainning accuracy= 95%
test accuracy = 83%
In general, achieving 100% training accuracy is often a sign of overfitting. The goal of hyperparameter tuning is to strike a balance between training accuracy and test accuracy by finding the optimal set of hyperparameters that prevent overfitting and lead to better generalization. While the test accuracy decreased slightly, the decrease in training accuracy suggests that the tuning process is helping the model to better generalize, which is a step in the right direction.

Random forest classifier
[ ]
#creating object of the classifier
rfc = RandomForestClassifier(random_state = 0)
[ ]
#fitting /training the model
rfc.fit(x_train,y_train)

[ ]
#predicying y values on train and test set
y_train_pred =rfc.predict(x_train)
y_test_pred = rfc.predict(x_test)
[ ]
# checking the accuracy score of train set
accuracy_score(y_train, y_train_pred)
1.0
[ ]
# checking the accuracy score of test set
accuracy_score(y_test, y_test_pred)
0.8702290076335878
[ ]
# confusion matrix for test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[79, 12,  0,  0],
       [ 6, 81, 10,  0],
       [ 0, 10, 83,  8],
       [ 0,  0,  5, 99]])
[ ]
 #classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       403
           1       1.00      1.00      1.00       395
           2       1.00      1.00      1.00       390
           3       1.00      1.00      1.00       383

    accuracy                           1.00      1571
   macro avg       1.00      1.00      1.00      1571
weighted avg       1.00      1.00      1.00      1571

let's tune some hyperparameters of Random forest classifier

[ ]
parameters = {'n_estimators':[100,200,300],'max_depth': [10,20,40],'min_samples_split':[2,6,10],'max_leaf_nodes':[5, 10, 20, 50],'criterion':['entropy','gini'],'max_features':['log2','sqrt']}
[ ]
rfc = RandomForestClassifier(random_state=0)
[ ]
# applying GridSearchCV
rfc_tune =  GridSearchCV(rfc,parameters, cv=5, scoring='accuracy',verbose=3)
rfc_tune.fit(x_train,y_train)

[ ]
print(rfc_tune.best_params_)
{'criterion': 'entropy', 'max_depth': 20, 'max_features': 'log2', 'max_leaf_nodes': 50, 'min_samples_split': 6, 'n_estimators': 300}
[ ]
# fitting the model with best parameters
rfc_best_p = RandomForestClassifier(criterion= 'entropy', max_depth= 20, max_features= 'log2', max_leaf_nodes= 50, min_samples_split= 6, n_estimators= 300)
rfc_best_p .fit(x_train,y_train)

[ ]
#predicying y values on train and test set
y_train_pred =rfc_best_p.predict(x_train)
y_test_pred = rfc_best_p.predict(x_test)
[ ]
# checking the accuracy score of train set
accuracy_score(y_train, y_train_pred)
0.9694462126034373
[ ]
# checking the accuracy score of train set
accuracy_score(y_test, y_test_pred)
0.8524173027989822
[ ]
# confusion matrix for test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[85,  6,  0,  0],
       [ 9, 72, 16,  0],
       [ 0, 15, 79,  7],
       [ 0,  0,  5, 99]])
[ ]
 #classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       403
           1       0.94      0.95      0.94       395
           2       0.97      0.93      0.95       390
           3       0.99      1.00      0.99       383

    accuracy                           0.97      1571
   macro avg       0.97      0.97      0.97      1571
weighted avg       0.97      0.97      0.97      1571

[ ]
#feature importance
def plot_feature_importance(algo, feature_names):
    importances = algo.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(x.shape[1]), importances[indices], align="center")
    plt.xticks(range(x.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, x.shape[1]])
    plt.tight_layout()
    plt.show()
[ ]
# Call the function to plot feature importance
plot_feature_importance(rfc_best_p, x.columns)

Observation of Random forest

before tuning

Training accuracy= 100%
test accuracy = 87%
The model is overfitted the training data and does not generalised so we tuned.

After tuning

Training accuracy= 96%

test accuracy = 85%

The drop in training accuracy and the modest improvement in test accuracy after tuning indicate that the model is likely to perform better on unseen data and that the tuning process is helping to address overfitting.

Gradient Boosting Classifier
[ ]
# creating object of the classifier
gbc = GradientBoostingClassifier(random_state = 0)
gbc.fit(x_train,y_train)

[ ]
#predicting values of y train and test set
y_train_pred =gbc.predict(x_train)
y_test_pred =gbc.predict(x_test)

[ ]
# checking the accuracy score of the train set
accuracy_score(y_train,y_train_pred)
1.0
[ ]
# checking the accuracy score of the test set
accuracy_score(y_test,y_test_pred)
0.9007633587786259
[ ]
# confusion matrix for test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[82,  9,  0,  0],
       [ 5, 85,  7,  0],
       [ 0,  8, 88,  5],
       [ 0,  0,  5, 99]])
[ ]
 #classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       403
           1       1.00      1.00      1.00       395
           2       1.00      1.00      1.00       390
           3       1.00      1.00      1.00       383

    accuracy                           1.00      1571
   macro avg       1.00      1.00      1.00      1571
weighted avg       1.00      1.00      1.00      1571

let's tune some hyperparameters of Gradient Boosting classifier

[ ]
parameters = {'learning_rate':[0.005,1,2,3],'min_samples_split':range(10,26)}
[ ]
# let's apply GridSearchCV
gbc_tune = GridSearchCV(gbc,parameters, cv= 5,scoring='accuracy',verbose=1)
gbc_tune.fit(x_train,y_train)

[ ]
gbc_tune.best_params_
{'learning_rate': 1, 'min_samples_split': 17}
[ ]
# fitting with the best parameters
gbc_best_p = GradientBoostingClassifier(learning_rate= 1, min_samples_split= 17,random_state=0)
gbc_best_p.fit(x_train,y_train)

[ ]
#predicting values of y train and test set
y_train_pred =gbc_best_p.predict(x_train)
y_test_pred =gbc_best_p.predict(x_test)
[ ]
# checking the accuracy score of the train set
accuracy_score(y_train,y_train_pred)
1.0
[ ]
# checking the accuracy score of the test set
accuracy_score(y_test,y_test_pred)
0.9287531806615776
[ ]
# confusion matrix for test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[ 82,   9,   0,   0],
       [  3,  89,   5,   0],
       [  0,   6,  94,   1],
       [  0,   0,   4, 100]])
[ ]
#classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       403
           1       1.00      1.00      1.00       395
           2       1.00      1.00      1.00       390
           3       1.00      1.00      1.00       383

    accuracy                           1.00      1571
   macro avg       1.00      1.00      1.00      1571
weighted avg       1.00      1.00      1.00      1571

[ ]
#feature importance
def plot_feature_importance(algo, feature_names):
    importances = algo.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(x.shape[1]), importances[indices], align="center")
    plt.xticks(range(x.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, x.shape[1]])
    plt.tight_layout()
    plt.show()
[ ]
# Call the function to plot feature importance
plot_feature_importance(gbc_best_p, x.columns)

Observation of the Gradient Boosting Classifier

Before tunning:

Trainning accuracy= 100%
test accuracy = 90%
The model is overfitted the training data and does not generalised so we tuned.

After tunning

Trainning accuracy= 100%
test accuracy = 92%
we have slightly improved the model performance but the model is not best.

K Nearest Neighbors
[ ]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)

[ ]
#predicting values of y train and test set
y_train_pred =knn.predict(x_train)
y_test_pred =knn.predict(x_test)
[ ]
# checking the accuracy score of the train set
accuracy_score(y_train,y_train_pred)
0.9567154678548695
[ ]
# checking the accuracy score of the test set
accuracy_score(y_test,y_test_pred)
0.9185750636132316
[ ]
# confusion matrix for test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[85,  6,  0,  0],
       [ 4, 89,  4,  0],
       [ 0,  3, 92,  6],
       [ 0,  0,  9, 95]])
[ ]
#classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       0.99      0.98      0.98       403
           1       0.94      0.96      0.95       395
           2       0.94      0.93      0.93       390
           3       0.97      0.96      0.96       383

    accuracy                           0.96      1571
   macro avg       0.96      0.96      0.96      1571
weighted avg       0.96      0.96      0.96      1571

let's tune some hyperparameters of K Nearest Neighbors

[ ]
parameters ={'n_neighbors':list(range(1,31))}
[ ]
# let's apply GridSearchCV
knn_tune = GridSearchCV(knn,parameters, cv= 5,scoring='accuracy',verbose=3)
knn_tune.fit(x_train,y_train)

[ ]
knn_tune.best_params_
{'n_neighbors': 7}
[ ]
# fitting model with the best parameters
knn_best_p = KNeighborsClassifier(n_neighbors= 7)
knn_best_p.fit(x_train,y_train)

[ ]
#predicting values of y train and test set
y_train_pred =knn_best_p.predict(x_train)
y_test_pred =knn_best_p.predict(x_test)
[ ]
# checking the accuracy score of the train set
accuracy_score(y_train,y_train_pred)
0.9458943348185869
[ ]
# checking the accuracy score of the test set
accuracy_score(y_test,y_test_pred)
0.926208651399491
[ ]
# confusion matrix for test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[86,  5,  0,  0],
       [ 5, 89,  3,  0],
       [ 0,  3, 92,  6],
       [ 0,  0,  7, 97]])
[ ]
#classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       403
           1       0.93      0.95      0.94       395
           2       0.91      0.92      0.91       390
           3       0.96      0.93      0.95       383

    accuracy                           0.95      1571
   macro avg       0.95      0.95      0.95      1571
weighted avg       0.95      0.95      0.95      1571

Observation of the K Nearest Neighbors

Before tuning:

Training accuracy= 95%
test accuracy = 91%
The model is overfitted the training data and does not generalised so we tuned.

After tunning

Training accuracy= 94%
test accuracy = 92%
Overall, the tuning of hyperparameters seems to have improved the generalization performance of the KNN classifier. The drop in training accuracy, accompanied by a rise in test accuracy, indicates that the model's performance is becoming more balanced and less prone to overfitting. This is a positive outcome, as models that generalize well on new data are more likely to perform better in real-world scenarios.

XGBoost Classifier
[ ]
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train,y_train)


[ ]
#predicting values of y train and test set
y_train_pred =xgb.predict(x_train)
y_test_pred =xgb.predict(x_test)
[ ]
# checking the accuracy score of the train set
accuracy_score(y_train,y_train_pred)
1.0
[ ]
# checking the accuracy score of the test set
accuracy_score(y_test,y_test_pred)
0.910941475826972
[ ]
# confusion matrix for test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[ 82,   9,   0,   0],
       [  4,  88,   5,   0],
       [  0,   8,  88,   5],
       [  0,   0,   4, 100]])
[ ]
#classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       403
           1       1.00      1.00      1.00       395
           2       1.00      1.00      1.00       390
           3       1.00      1.00      1.00       383

    accuracy                           1.00      1571
   macro avg       1.00      1.00      1.00      1571
weighted avg       1.00      1.00      1.00      1571

let's tune some hyperparameters of xgboost

[ ]
parameters = {'learning_rate':[0.6,1],'n_estimators':[500,1000],'gamma':[0.2],'subsample':[0.5,0.6]}
[ ]
xgb_tune = GridSearchCV(xgb,parameters,cv=3,verbose =4)
xgb_tune.fit(x_train,y_train)

[ ]
xgb_tune.best_params_
{'gamma': 0.2, 'learning_rate': 0.6, 'n_estimators': 500, 'subsample': 0.6}
[ ]
#Fitting the model with the best parameter
xgb_best_p = XGBClassifier(gamma= 0.2, learning_rate= 0.6, n_estimators= 500, subsample= 0.6)
xgb_best_p.fit(x_train,y_train)

[ ]
#predicting values of y train and test set
y_train_pred =xgb_best_p.predict(x_train)
y_test_pred =xgb_best_p.predict(x_test)
[ ]
# checking the accuracy score of the train set
accuracy_score(y_train,y_train_pred)
1.0
[ ]
# checking the accuracy score of the test set
accuracy_score(y_test,y_test_pred)
0.910941475826972
[ ]
# confusion matrix for test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[85,  6,  0,  0],
       [ 4, 88,  5,  0],
       [ 0,  9, 87,  5],
       [ 0,  0,  6, 98]])
[ ]
#classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       403
           1       1.00      1.00      1.00       395
           2       1.00      1.00      1.00       390
           3       1.00      1.00      1.00       383

    accuracy                           1.00      1571
   macro avg       1.00      1.00      1.00      1571
weighted avg       1.00      1.00      1.00      1571

[ ]
#feature importance
def plot_feature_importance(algo, feature_names):
    importances = algo.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(x.shape[1]), importances[indices], align="center")
    plt.xticks(range(x.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, x.shape[1]])
    plt.tight_layout()
    plt.show()
[ ]
# Call the function to plot feature importance
plot_feature_importance(xgb_best_p, x.columns)

Observation of the XGBoost Classifier

Before tuning:

Training accuracy= 100%
test accuracy = 91%
The model is overfitted the training data and does not generalised so we tuned.

After tuning

Training accuracy= 100%

test accuracy = 91%

achieving a training accuracy of 100% could be a sign of overfitting, where the model has memorized the training data and may not generalize well to new, unseen data. The fact that the test accuracy is not much lower than the training accuracy is a positive sign, as it indicates that the model is not severely overfitting. However, the goal is to achieve a good balance between training and test accuracy while avoiding overfitting.

SVM
[ ]
from sklearn.svm import SVC
svc = SVC(random_state= None)
svc.fit(x_train,y_train)

[ ]
#predicting values of y train and test set
y_train_pred =svc.predict(x_train)
y_test_pred =svc.predict(x_test)
[ ]
# checking the accuracy score of the train set
accuracy_score(y_train,y_train_pred)
0.9567154678548695
[ ]
# checking the accuracy score of the test set
accuracy_score(y_test,y_test_pred)
0.9440203562340967
[ ]
# confusion matrix for test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[ 89,   2,   0,   0],
       [  2,  94,   1,   0],
       [  0,   8,  88,   5],
       [  0,   0,   4, 100]])
[ ]
#classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       403
           1       0.94      0.97      0.95       395
           2       0.95      0.91      0.93       390
           3       0.96      0.97      0.96       383

    accuracy                           0.96      1571
   macro avg       0.96      0.96      0.96      1571
weighted avg       0.96      0.96      0.96      1571

let's tune some hyperparameters of SVC

[ ]
parameters = {'C':[0.1,1,10],'kernel':['linear','rbf']}
[ ]
svc_tune = GridSearchCV(svc,parameters, cv= 5, verbose=2)
svc_tune.fit(x_train,y_train)

[ ]
svc_tune.best_params_
{'C': 0.1, 'kernel': 'linear'}
[ ]
# fitting model with best parameters
svc_best_p= SVC(C= 0.1, kernel= 'linear')
svc_best_p.fit(x_train,y_train)

[ ]
#predicting values of y train and test set
y_train_pred =svc_best_p.predict(x_train)
y_test_pred =svc_best_p.predict(x_test)
[ ]
# checking the accuracy score of the train set
accuracy_score(y_train,y_train_pred)
0.9853596435391471
[ ]
# checking the accuracy score of the test set
accuracy_score(y_test,y_test_pred)
0.9796437659033079
[ ]
# confusion matrix for test set
cf_matrix= confusion_matrix(y_test,y_test_pred)
cf_matrix
array([[ 88,   3,   0,   0],
       [  0,  97,   0,   0],
       [  0,   3,  98,   0],
       [  0,   0,   2, 102]])
[ ]
#classification report for the train set
print(classification_report(y_train,y_train_pred))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       403
           1       0.98      0.98      0.98       395
           2       0.97      0.98      0.98       390
           3       0.99      0.99      0.99       383

    accuracy                           0.99      1571
   macro avg       0.99      0.99      0.99      1571
weighted avg       0.99      0.99      0.99      1571

Observation of the SVM Classifier

Before tuning:

Training accuracy= 95%
test accuracy = 94%
The model is overfitted the training data and does not generalised so we tuned.

After tuning

Training accuracy= 98%
test accuracy = 97%
Overall, the SVM classifier seems to be performing well both before and after tuning. The tuning process has resulted in a slight improvement in both training and test accuracy, which is a positive outcome. It's important to strike a balance between training and test accuracy to ensure the model generalizes well to new data.


Conclusion
We started with Data understanding ,Data wrangling, basic EDA where we found the trends between prices range and other independent features.

Implemented various classification algorithms , out of which SVM(Support vector machine) algothrithm gave the best performance after hyper-tuning.

KNN is the second best good model gave good performance after hyper-tunning.

We fine-tuned the chosen models' hyperparameters to optimize their performance. This involved using techniques like GridSearchCV to find the best combination of hyperparameters.

We evaluated the models using various metrics such as accuracy, precision, recall, F1-score, and confusion matrices. These metrics helped us assess the models' performance on both training and test data.
