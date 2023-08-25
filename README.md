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



Observation of Decision Tree Classifier

before tunning

Trainning accuracy= 100%
test accuracy = 85%
The model is overfitted the training data and does not generalised so we tuned.

After tunning

Trainning accuracy= 95%
test accuracy = 83%
In general, achieving 100% training accuracy is often a sign of overfitting. The goal of hyperparameter tuning is to strike a balance between training accuracy and test accuracy by finding the optimal set of hyperparameters that prevent overfitting and lead to better generalization. While the test accuracy decreased slightly, the decrease in training accuracy suggests that the tuning process is helping the model to better generalize, which is a step in the right direction.



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
