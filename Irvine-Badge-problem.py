## import packages

import numpy as np
import pandas as pd
import string

name = pd.read_csv('namelabel.csv', header=None)

name.columns = name.columns.astype(str)

name.columns = ['names']

def firstname(allname):
    firstname = allname.split()[1]
    return firstname

def midname(allname):
    if allname.split()[2][1] ==  '.':
        middlename = allname.split()[2]
        return middlename
    
def lastname(allname):
    if allname.split()[2][1] ==  '.':
        lastname = allname.split()[3]    
    else:
        lastname = allname.split()[2]
    return lastname

def label(allname):
    label = allname.split()[0]
    return label

lastname('+ Bradley L. Whitehall')

name['first name'] = name['names'].apply(lambda x: firstname(x))
name['middle name'] = name['names'].apply(lambda x: midname(x))
name['last name'] = name['names'].apply(lambda x: lastname(x))
name['label'] = name['names'].apply(lambda x: label(x))

def remove_none(name):
    
    if name is None:
        return ''
    else:
        return name

name['middle name'] = name['middle name'].apply(remove_none)
name['length of first'] = name['first name'].apply(len)
name['length of last'] = name['last name'].apply(len)

names = name
names.head()

def countVowels(name):
    namevowels = []
    vowels = ['A','a','E','e','I','i','O','o','U','u']
    for letter in name:
        if letter in vowels:
            namevowels.append(letter)
    return len(namevowels)

def countCons(name):
    namecons = []
    vowels = ['A','a','E','e','I','i','O','o','U','u']
    for letter in name:
        if letter not in vowels:
            namecons.append(letter)
    return len(namecons)

vowels = ['A','a','E','e','I','i','O','o','U','u']

names['firstvowels'] = names['first name'].apply(countVowels)
names['firstCons'] = names['first name'].apply(countCons)
names['lastvowels'] = names['last name'].apply(countVowels)
names['lastCons'] = names['last name'].apply(countCons)

def middlenamedummy(name):
    
    if name == '':
        return 0
    else:
        return 1

names['middledummy'] = names['middle name'].apply(middlenamedummy) # is there a middle name?

names['totalVowels'] = (names['first name'] + names['middle name'] + names['last name']).apply(countVowels) # count the number of vowels in the name

names['totalCons'] = (names['first name'] + names['middle name'] + names['last name']).apply(countCons) # count the number of consonants in the name

names['VtFratio'] = names['totalVowels']/(names['totalVowels']+names['totalCons']) # ratio of number of vowels to total length

names['CtFratio'] = names['totalCons']/(names['totalVowels']+names['totalCons']) # ratio of number of consonants to total length

names['VtCratio'] = names['totalVowels']/names['totalCons'] # ratio of number of vowels to number of consonants

names['FtVratio'] = round((names['totalVowels']+names['totalCons'])/names['totalVowels'],0) # ratio of total length to number of vowels

names['FtCratio'] = round((names['totalVowels']+names['totalCons'])/names['totalCons'],0) # ratio of total length to number of consonants

names['CtVratio'] = round(names['totalCons']/names['totalVowels'],0) # ratio of consonants to vowels

# DUMMIES BASED ON EVEN

# Is the total length an even number?
names['tot_lengthdummy'] = (names['totalVowels']+names['totalCons']).apply(lambda x: 1 if x%2 == 0 else 0)

# Is the length of first name an even number?
names['LoFdummy'] = names['length of first'].apply(lambda x: 1 if x%2 == 0 else 0)

# Is the length of last name an even number?
names['LoLdummy'] = names['length of last'].apply(lambda x: 1 if x%2 == 0 else 0)

# Is the number of vowels in first name even?
names['FVdummy'] = names['firstvowels'].apply(lambda x: 1 if x%2 == 0 else 0)

# Is the number of cons in first name even?
names['FCdummy'] = names['firstCons'].apply(lambda x: 1 if x%2 == 0 else 0)

# Is the number of vowels in last name even?
names['LVdummy'] = names['lastvowels'].apply(lambda x: 1 if x%2 == 0 else 0)

# Is the number of cons in last name even?
names['LCdummy'] = names['lastCons'].apply(lambda x: 1 if x%2 == 0 else 0)

# Is the total number of vowels in name even?
names['TVdummy'] = names['totalVowels'].apply(lambda x: 1 if x%2 == 0 else 0)

# Is the total number of cons in name even?
names['TCdummy'] = names['totalCons'].apply(lambda x: 1 if x%2 == 0 else 0)

# Is the ratio of length of full name to number of vowels even?
names['FtVdummy'] = names['FtVratio'].apply(lambda x: 1 if x%2 == 0 else 0)

# Is the ratio of length of full name to number of consonants even?
names['FtCdummy'] = names['FtCratio'].apply(lambda x: 1 if x%2 == 0 else 0)

# Is the ratio of consonants to number of vowels even?
names['CtVdummy'] = names['FtCratio'].apply(lambda x: 1 if x%2 == 0 else 0)

names['total_length'] = (names['totalVowels']+names['totalCons']) # total length of name

names['Vowel_Fdummy'] = names['first name'].apply(lambda x: 1 if x[0] in vowels else 0) # Is first letter in name a vowel?

names['Vowel_Sdummy'] = names['first name'].apply(lambda x: 1 if x[1] in vowels else 0) # Is second a letter in name a vowel?


# # As you can see, I threw in everything including the kitchen sink 
# 
# I ended up with 33 unique predictor variables
names.columns

names1 = names[['total_length', 'middledummy', 'tot_lengthdummy',
       'Vowel_Fdummy', 'Vowel_Sdummy', 'LoFdummy', 'LoLdummy', 'FVdummy', 'FCdummy', 'LVdummy', 'LCdummy',
       'TVdummy', 'TCdummy', 'FtVdummy', 'FtCdummy', 'CtVdummy', 'label']]

names1.head()

from sklearn.model_selection import train_test_split

X = names1.drop('label', axis=1)
y = names1['label']

X.head()


# # As we will see, a single tree worked great, with 100% accuracy. I didn't need  a whole forest :)
# 
# Nevertheless, I have another model below with the forest.
# 
# One might rightly wonder if all these variables were useful, I went all in at first. Now that I realize my model is working great, I decided to find the best variables. You will see the results further down.
# 
# I used the RFE from sklearn.feature_selection to select the top six variables:
# 
# 1. What is the total length of the person's name? ('total_length') 
# 
# 2. Is the first letter of the person's name a vowel? ('Vowel_Fdummy') 
# 
# 3. Is the second letter of the name a vowel? ('Vowel_Sdummy') 
# 
# 4. Is the length of last name an even number? ('LoLdummy') 
# 
# 5. Is the number of vowels in first name an even numer? ('FVdummy')
# 
# 6. Is there a middle name? ('middledummy')

# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print("These are the results for the decision tree classifier")
print(classification_report(y_test,predictions))
print('\n')
print (confusion_matrix(y_test,predictions))


print('\n')
print("Let's run for Random Forest")
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(y_test,rfc_pred))
print('\n')
print (confusion_matrix(y_test,rfc_pred))

print('\n')
print("Now, let's trim and identify the most useful predictors")

from sklearn.feature_selection import RFE

rfe = RFE(rfc, 5)
fit = rfe.fit(X_train, y_train)
print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)

print("\n Run a new model from most 6 most important predictors from our ranking ranking above")

X = names1[['total_length', 'tot_lengthdummy', 'Vowel_Fdummy', 'Vowel_Sdummy', 'LoLdummy', 'LVdummy']]
y = names1['label'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rfc = RandomForestClassifier(n_estimators=10)

rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(y_test,rfc_pred))
print('\n')
print (confusion_matrix(y_test,rfc_pred))

