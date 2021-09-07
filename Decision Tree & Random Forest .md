Objective
Decision Tree & Random Forest 
We are going to predict once again if a passenger on the Titanic is going to survive or not using decision trees and random forests this time: 

1. Read your Titanic dataset as usual: A training set and Testing set Apply decision tree. 

2. Plot your decision tree and try to read the tree branches and conclude a prediction manually.

3. Change the decision tree parameters(change at least two parameters), 

4. Calculate the new accuracy and compare it with the previous results. 

5. Use random forest then change the number of estimators

6. Calculate the new accuracy and compare it with the previous result.
 

# 1. Read your Titanic dataset as usual: A training set and Testing set Apply decision tree.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

```


```python
df = pd.read_csv('titanic-passengers.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>343</td>
      <td>No</td>
      <td>2</td>
      <td>Collander, Mr. Erik Gustaf</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>248740</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76</td>
      <td>No</td>
      <td>3</td>
      <td>Moen, Mr. Sigurd Hansen</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>348123</td>
      <td>7.6500</td>
      <td>F G73</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>641</td>
      <td>No</td>
      <td>3</td>
      <td>Jensen, Mr. Hans Peder</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>350050</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>568</td>
      <td>No</td>
      <td>3</td>
      <td>Palsson, Mrs. Nils (Alma Cornelia Berglund)</td>
      <td>female</td>
      <td>29.0</td>
      <td>0</td>
      <td>4</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>672</td>
      <td>No</td>
      <td>1</td>
      <td>Davidson, Mr. Thornton</td>
      <td>male</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
      <td>F.C. 12750</td>
      <td>52.0000</td>
      <td>B71</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cabin</th>
      <td>687</td>
      <td>77.1</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>177</td>
      <td>19.9</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>2</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>PassengerId</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop rows that has missing values 
df = df.dropna(subset=['Embarked']) 
```


```python
embarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_one_hot], axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>343</td>
      <td>No</td>
      <td>2</td>
      <td>Collander, Mr. Erik Gustaf</td>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>248740</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76</td>
      <td>No</td>
      <td>3</td>
      <td>Moen, Mr. Sigurd Hansen</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>348123</td>
      <td>7.6500</td>
      <td>F G73</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>641</td>
      <td>No</td>
      <td>3</td>
      <td>Jensen, Mr. Hans Peder</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>350050</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>568</td>
      <td>No</td>
      <td>3</td>
      <td>Palsson, Mrs. Nils (Alma Cornelia Berglund)</td>
      <td>female</td>
      <td>29.0</td>
      <td>0</td>
      <td>4</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>672</td>
      <td>No</td>
      <td>1</td>
      <td>Davidson, Mr. Thornton</td>
      <td>male</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
      <td>F.C. 12750</td>
      <td>52.0000</td>
      <td>B71</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
str(df)
```




    '     PassengerId Survived  Pclass  \\\n0            343       No       2   \n1             76       No       3   \n2            641       No       3   \n3            568       No       3   \n4            672       No       1   \n..           ...      ...     ...   \n886           10      Yes       2   \n887           61       No       3   \n888          535       No       3   \n889          102       No       3   \n890          428      Yes       2   \n\n                                                  Name     Sex   Age  SibSp  \\\n0                           Collander, Mr. Erik Gustaf    male  28.0      0   \n1                              Moen, Mr. Sigurd Hansen    male  25.0      0   \n2                               Jensen, Mr. Hans Peder    male  20.0      0   \n3          Palsson, Mrs. Nils (Alma Cornelia Berglund)  female  29.0      0   \n4                               Davidson, Mr. Thornton    male  31.0      1   \n..                                                 ...     ...   ...    ...   \n886                Nasser, Mrs. Nicholas (Adele Achem)  female  14.0      1   \n887                              Sirayanian, Mr. Orsen    male  22.0      0   \n888                                Cacic, Miss. Marija  female  30.0      0   \n889                   Petroff, Mr. Pastcho ("Pentcho")    male   NaN      0   \n890  Phillips, Miss. Kate Florence ("Mrs Kate Louis...  female  19.0      0   \n\n     Parch      Ticket     Fare  Cabin Embarked  Embarked_C  Embarked_Q  \\\n0        0      248740  13.0000    NaN        S           0           0   \n1        0      348123   7.6500  F G73        S           0           0   \n2        0      350050   7.8542    NaN        S           0           0   \n3        4      349909  21.0750    NaN        S           0           0   \n4        0  F.C. 12750  52.0000    B71        S           0           0   \n..     ...         ...      ...    ...      ...         ...         ...   \n886      0      237736  30.0708    NaN        C           1           0   \n887      0        2669   7.2292    NaN        C           1           0   \n888      0      315084   8.6625    NaN        S           0           0   \n889      0      349215   7.8958    NaN        S           0           0   \n890      0      250655  26.0000    NaN        S           0           0   \n\n     Embarked_S  \n0             1  \n1             1  \n2             1  \n3             1  \n4             1  \n..          ...  \n886           0  \n887           0  \n888           1  \n889           1  \n890           1  \n\n[889 rows x 15 columns]'




```python
# Filling missing values in Cabin column with 'Unknown'
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['Cabin'] = df['Cabin'].apply(lambda x: x[0])
```


```python
# cabin into one-hot
cabin_one_hot = pd.get_dummies(df['Cabin'], prefix='Cabin')
df = pd.concat([df, cabin_one_hot], axis=1)
print(df.columns)
```

    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Embarked_C',
           'Embarked_Q', 'Embarked_S', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D',
           'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_U'],
          dtype='object')
    


```python
# sex into one-hot
sex_one_hot = pd.get_dummies(df['Sex'], prefix='Sex')
df = pd.concat([df, sex_one_hot], axis=1)
```


```python
# function to extract title from Name column
def get_title(x):
    return x.split(',')[1].split('.')[0].strip()

df['Title'] = df['Name'].apply(get_title)

print(df['Title'].unique())
title_one_hot = pd.get_dummies(df['Title'], prefix='Title')
df = pd.concat([df, title_one_hot], axis=1)
```

    ['Mr' 'Mrs' 'Miss' 'Dr' 'Major' 'Don' 'Master' 'Rev' 'Col' 'Mlle' 'Lady'
     'Jonkheer' 'Mme' 'Sir' 'Capt' 'the Countess' 'Ms']
    


```python
# age median of each title
age_median = df.groupby('Title')['Age'].median()
print(age_median)

def fill_age(x):
    for index, age in zip(age_median.index, age_median.values):
        if x['Title'] == index:
            return age

df['Age'] = df.apply(lambda x: fill_age(x) if np.isnan(x['Age']) else x['Age'], axis=1)

```

    Title
    Capt            70.0
    Col             58.0
    Don             40.0
    Dr              46.5
    Jonkheer        38.0
    Lady            48.0
    Major           48.5
    Master           3.5
    Miss            21.0
    Mlle            24.0
    Mme             24.0
    Mr              30.0
    Mrs             35.0
    Ms              28.0
    Rev             46.5
    Sir             49.0
    the Countess    33.0
    Name: Age, dtype: float64
    


```python
# Drop all columns with categorical values
df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1)
```


```python

```


```python

```


```python
# target (y) , features(X)
y = df['Survived'].values
X = df.iloc[:,1:].values
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21, test_size=0.2)
```


```python
clf = LogisticRegression()
```


```python
# Training model
clf.fit(X_train, y_train)
```

    C:\Users\arokh\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    LogisticRegression()




```python
# accuracy score
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
```

    0.8537271448663853
    0.7865168539325843
    

# Decision Tree


```python
decision_tree = DecisionTreeClassifier() decision_tree.fit(X_train, Y_train)  Y_pred = decision_tree.predict(X_test)  acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
```


      File "<ipython-input-58-b0e59598a3a3>", line 1
        decision_tree = DecisionTreeClassifier() decision_tree.fit(X_train, Y_train)  Y_pred = decision_tree.predict(X_test)  acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
                                                 ^
    SyntaxError: invalid syntax
    



```python
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-37-e91ae4a13054> in <module>
          4               'Stochastic Gradient Decent',
          5               'Decision Tree'],
    ----> 6     'Score': [acc_linear_svc, acc_knn, acc_log, 
          7               acc_random_forest, acc_gaussian, acc_perceptron,
          8               acc_sgd, acc_decision_tree]})
    

    NameError: name 'acc_linear_svc' is not defined



```python
# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
y_train = train['Survived']
x_train = train.drop(['Survived'], axis=1).values 
x_test = test.values

# Create Decision Tree with max_depth = 3
decision_tree = tree.DecisionTreeClassifier(max_depth = 3)
decision_tree.fit(x_train, y_train)

# Predicting results for test dataset
y_pred = decision_tree.predict(x_test)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)

# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(train.drop(['Survived'], axis=1)),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])
# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          '"Title <= 1.5" corresponds to "Mr." title', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('sample-out.png')
PImage("sample-out.png")

# Code to check available fonts and respective paths
# import matplotlib.font_manager
# matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-38-8c56103fe62b> in <module>
          1 # Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
    ----> 2 y_train = train['Survived']
          3 x_train = train.drop(['Survived'], axis=1).values
          4 x_test = test.values
          5 
    

    NameError: name 'train' is not defined



```python
# Load in the R package  
#install.packages('rpart')
require(rpart)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-39-b71eff305f60> in <module>
          1 # Load in the R package
          2 #install.packages('rpart')
    ----> 3 require(rpart)
    

    NameError: name 'require' is not defined



```python

```


```python
# Build the decision tree
my_tree_two <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train, method = "class")

# Visualize the decision tree using plot() and text()
#plot(my_tree_two)
#text(my_tree_two)

# Load in the packages to build a fancy plot
#install.packages('rattle')
#install.packages('rpart.plot')
#install.packages('RColorBrewer')
library(rattle)
```


      File "<ipython-input-41-323e6f6a3def>", line 2
        my_tree_two <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train, method = "class")
                                      ^
    SyntaxError: invalid syntax
    



```python
my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
my_tree_one = my_tree_one.fit(X, y)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-42-f98a8d328fe5> in <module>
    ----> 1 my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
          2 my_tree_one = my_tree_one.fit(X, y)
    

    NameError: name 'tree' is not defined



```python
y = targets = labels = df["Survived"].values

columns = ["Pclass", "Age", "SibSp"]
features = df[list(columns)].values
features
```




    array([[ 2., 28.,  0.],
           [ 3., 25.,  0.],
           [ 3., 20.,  0.],
           ...,
           [ 3., 30.,  0.],
           [ 3., 30.,  0.],
           [ 2., 19.,  0.]])




```python

\

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-52-a95c4aeedbe3> in <module>
          1 import matplotlib.pyplot as plt
          2 
    ----> 3 train_df = pd.read_csv(url+"train.csv")
          4 test_df = pd.read_csv(url+"test.csv")
    

    NameError: name 'url' is not defined



```python
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(features)
X
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-47-418cb99f2a86> in <module>
    ----> 1 imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
          2 X = imp.fit_transform(features)
          3 X
    

    NameError: name 'Imputer' is not defined



```python
my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
my_tree_one = my_tree_one.fit(X, y)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-48-f98a8d328fe5> in <module>
    ----> 1 my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
          2 my_tree_one = my_tree_one.fit(X, y)
    

    NameError: name 'tree' is not defined



```python
# Outcome variable
(DV <- "Survived")

# Predictor variables Model [1]
(IVs <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", 
          "Title", "Master.Male", "Female.Group"))

# Create the formula string 
(FMLA <- paste(DV, "~", paste(IVs, collapse = " + ")))

# Create recipe of formula
FMLA.Recipe <- recipe(Survived ~ Pclass + Sex + Age + SibSp + 
                      Parch + Fare + Embarked +  Title + Master.Male +
                      Female.Group, 
                      data = train.cv)
```


      File "<ipython-input-55-3a03817a97ae>", line 12
        FMLA.Recipe <- recipe(Survived ~ Pclass + Sex + Age + SibSp +
                                       ^
    SyntaxError: invalid syntax
    



```python
#Print Confusion matrix 
pred = .predict(X)
df_confusion = metrics.confusion_matrix(y, pred)
df_confusion
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-57-b0841aed0a23> in <module>
          1 #Print Confusion matrix
    ----> 2 pred = df.predict(X)
          3 df_confusion = metrics.confusion_matrix(y, pred)
          4 df_confusion
    

    ~\anaconda3\lib\site-packages\pandas\core\generic.py in __getattr__(self, name)
       5463             if self._info_axis._can_hold_identifiers_and_holds_name(name):
       5464                 return self[name]
    -> 5465             return object.__getattribute__(self, name)
       5466 
       5467     def __setattr__(self, name: str, value) -> None:
    

    AttributeError: 'DataFrame' object has no attribute 'predict'



```python

```


```python

```


```python

```

# 2. Plot your decision tree and try to read the tree branches and conclude a prediction manually.


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

# 3. Change the decision tree parameters(change at least two parameters), 


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

# 4. Calculate the new accuracy and compare it with the previous results. 


```python

```


```python

```


```python

```


```python

```

# 5. Use random forest then change the number of estimators


```python

```


```python

```


```python

```


```python

```

# 6. Calculate the new accuracy and compare it with the previous result.


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
