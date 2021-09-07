#!/usr/bin/env python
# coding: utf-8

# Objective
# Decision Tree & Random Forest 
# We are going to predict once again if a passenger on the Titanic is going to survive or not using decision trees and random forests this time: 
# 
# 1. Read your Titanic dataset as usual: A training set and Testing set Apply decision tree. 
# 
# 2. Plot your decision tree and try to read the tree branches and conclude a prediction manually.
# 
# 3. Change the decision tree parameters(change at least two parameters), 
# 
# 4. Calculate the new accuracy and compare it with the previous results. 
# 
# 5. Use random forest then change the number of estimators
# 
# 6. Calculate the new accuracy and compare it with the previous result.
#  

# # 1. Read your Titanic dataset as usual: A training set and Testing set Apply decision tree.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv('titanic-passengers.csv')


# In[3]:


df.head()


# In[4]:


total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# In[5]:


# Drop rows that has missing values 
df = df.dropna(subset=['Embarked']) 


# In[6]:


embarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_one_hot], axis=1)
df.head()


# In[10]:


str(df)


# In[16]:


# Filling missing values in Cabin column with 'Unknown'
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['Cabin'] = df['Cabin'].apply(lambda x: x[0])


# In[17]:


# cabin into one-hot
cabin_one_hot = pd.get_dummies(df['Cabin'], prefix='Cabin')
df = pd.concat([df, cabin_one_hot], axis=1)
print(df.columns)


# In[18]:


# sex into one-hot
sex_one_hot = pd.get_dummies(df['Sex'], prefix='Sex')
df = pd.concat([df, sex_one_hot], axis=1)


# In[19]:


# function to extract title from Name column
def get_title(x):
    return x.split(',')[1].split('.')[0].strip()

df['Title'] = df['Name'].apply(get_title)

print(df['Title'].unique())
title_one_hot = pd.get_dummies(df['Title'], prefix='Title')
df = pd.concat([df, title_one_hot], axis=1)


# In[20]:


# age median of each title
age_median = df.groupby('Title')['Age'].median()
print(age_median)

def fill_age(x):
    for index, age in zip(age_median.index, age_median.values):
        if x['Title'] == index:
            return age

df['Age'] = df.apply(lambda x: fill_age(x) if np.isnan(x['Age']) else x['Age'], axis=1)


# In[21]:


# Drop all columns with categorical values
df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1)


# In[ ]:





# In[ ]:





# In[22]:


# target (y) , features(X)
y = df['Survived'].values
X = df.iloc[:,1:].values


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21, test_size=0.2)


# In[33]:


clf = LogisticRegression()


# In[34]:


# Training model
clf.fit(X_train, y_train)


# In[35]:


# accuracy score
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))


# # Decision Tree

# In[58]:


decision_tree = DecisionTreeClassifier() decision_tree.fit(X_train, Y_train)  Y_pred = decision_tree.predict(X_test)  acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


# In[37]:


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


# In[38]:


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


# In[39]:


# Load in the R package  
#install.packages('rpart')
require(rpart)


# In[ ]:





# In[41]:


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


# In[42]:


my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
my_tree_one = my_tree_one.fit(X, y)


# In[46]:


y = targets = labels = df["Survived"].values

columns = ["Pclass", "Age", "SibSp"]
features = df[list(columns)].values
features


# In[52]:





# In[47]:


imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(features)
X


# In[48]:


my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
my_tree_one = my_tree_one.fit(X, y)


# In[55]:


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


# In[57]:


#Print Confusion matrix 
pred = .predict(X)
df_confusion = metrics.confusion_matrix(y, pred)
df_confusion


# In[ ]:





# In[ ]:





# In[ ]:





# # 2. Plot your decision tree and try to read the tree branches and conclude a prediction manually.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 3. Change the decision tree parameters(change at least two parameters), 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 4. Calculate the new accuracy and compare it with the previous results. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 5. Use random forest then change the number of estimators

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 6. Calculate the new accuracy and compare it with the previous result.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




