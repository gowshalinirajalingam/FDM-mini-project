

#mini project
#classification
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from pandas import DataFrame
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#df =  pd.read_excel('CBC.xls',sheetname="DATA")
#print(df)


df = pd.read_csv("G:\\sliit DS\\3rd year 2nd seme\\FDM\\5.after mid\\mini project\\updatedCSV.csv")
print(df)

#pre processing
df.dropna()

df.dtypes





correlations=df[['Seq#','ID#','Gender','M','R','F','FirstPurch','Related Purchase','Florence','Mcode','Rcode','Fcode']].corr()    
print(correlations)

df[['Gender']].hist()
df[['M']].hist()
df[['R']].hist()
df[['F']].hist()
df[['FirstPurch']].hist()
df[['Related Purchase']].hist()
df[['Florence']].hist()


#segmentation of data set


temp_df=df[['Gender','M','R','F','FirstPurch','Related Purchase','Florence']]
temp_df.dtypes
y=temp_df.iloc[:,6]
x=temp_df.iloc[:,0:6]
#xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
#xtrain,xval,ytrain,yval=train_test_split(xtrain, ytrain, test_size=0.4375)
#

#segmenting data in balanced form
#reference :http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
from sklearn.model_selection import StratifiedShuffleSplit    

sss = StratifiedShuffleSplit(train_size=0.8, n_splits=1, 
                             test_size=0.2, random_state=0)  

for train_index, test_index in sss.split(x, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
    ytrain, ytest = y.iloc[train_index], y.iloc[test_index]


sss = StratifiedShuffleSplit(train_size=0.5625, n_splits=1, 
                             test_size=0.4375, random_state=0)  

for train_index, val_index in sss.split(xtrain, ytrain):
    print("TRAIN:", train_index, "VAL:", test_index)
    xtrain, xval = x.iloc[train_index], x.iloc[val_index]
    ytrain, yval = y.iloc[train_index], y.iloc[val_index]


#modeling classification models

#normalizing data
xtrain =xtrain.values #returns a numpy array
xtest=xtest.values
min_max_scaler = preprocessing.MinMaxScaler()

#normalize traindata
x_scaled = min_max_scaler.fit_transform(xtrain)
xtrain = pd.DataFrame(x_scaled)

#normalize test data
x_scaled = min_max_scaler.fit_transform(xtest)
xtest = pd.DataFrame(x_scaled)


#build model

#from sklearn.neighbors import KNeighborsClassifier
#model4=KNeighborsClassifier(n_neighbors=5).fit(xtrain,ytrain) # default value for n_neighbors is 5
#yhat4 = model4.predict(xval)
#accuracy_score(yval, yhat4)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0,class_weight='balanced').fit(xtrain,ytrain)
yhat = model.predict(xval)
len(yhat)
accuracy_score(yval, yhat)

#from sklearn.naive_bayes import GaussianNB
#model1=GaussianNB().fit(xtrain,ytrain)
#yhat1 = model1.predict(xval)
#accuracy_score(yval, yhat1)

#from sklearn import svm
#model2=svm.SVC().fit(xtrain,ytrain)
#yhat2 = model2.predict(xval)
#accuracy_score(yval, yhat2)

#from sklearn import tree
#model3 = tree.DecisionTreeClassifier(criterion='gini').fit(xtrain,ytrain)
#yhat3 = model3.predict(xval)
#accuracy_score(yval, yhat3)



#
#from sklearn.ensemble import RandomForestClassifier
#model5= RandomForestClassifier().fit(xtrain,ytrain) # default value for n_neighbors is 5
#yhat5 = model5.predict(xval)
#accuracy_score(yval, yhat5)


from sklearn.ensemble import GradientBoostingClassifier
model6= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(xtrain,ytrain) # default value for n_neighbors is 5
yhat6 = model6.predict(xval)
accuracy_score(yval, yhat6)








#implemented the probability score
#https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/




#log Loss


#Log loss, also called “logistic loss,” “logarithmic loss,” or “cross entropy” 
#can be used as a measure for evaluating predicted probabilities.
#Each predicted probability is compared to the actual class 
#output value (0 or 1) and a score is calculated that penalizes the 
#probability based on the distance from the expected value. 
#The penalty is logarithmic, offering a small score for small 
#differences (0.1 or 0.2) and enormous score 
#for a large difference (0.9 or 1.0).

#log loss can be (0 to infinity).if log loss is greater than 1 predicted
#values are deviates more from actual y.
#A model with perfect skill has a log loss score of 0.0.
#
#In order to summarize the skill of a model using log loss, the log loss is calculated 
#for each predicted probability, and the average loss is reported.
#from sklearn.metrics import log_loss





from sklearn.metrics import log_loss

probs = model6.predict_proba(x)       # predict probabilities
# keep the predictions for class 1 only
probs = probs[:, 1]
len(probs)
pd.DataFrame(probs)  
# calculate log loss
loss = log_loss(y, probs)



#reference:https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
#https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/


#The optimal cut off point would be where true positive rate is high and the 
#false positive rate is low. 
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds =roc_curve(y, pd.DataFrame(probs))
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

#ploted the area under the curve(ROC curve) fpr VS tpr
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr)
# show the plot
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.ix[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'])
pl.plot(roc['1-fpr'], color = 'red')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])


#references:https://groups.bme.gatech.edu/groups/biml/resources/useful_documents/Test_Statistics.pdf