#simple case illustration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from sklearn import datasets
X, y = datasets.make_hastie_10_2(n_samples=20000, random_state=1)
df=pd.DataFrame(data=X,columns=['X'+str(i) for i in range(1,11)])
df['y']=pd.Series(y)
df.head()
i=1
plt.figure(figsize=(20,20))
for c in df.columns[:-1]:
    plt.subplot(4,3,i)
    plt.title(f"Boxplot of {c}",fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    sns.boxplot(y=df[c],x=df['y'])
    i+=1
plt.show()
df_sample=df.sample(frac=0.01)
sns.set(style="ticks")
g=sns.pairplot(df_sample,vars=["X1","X2","X3"],
               hue="y",markers=["o", "s"],
               diag_kind="kde",diag_kws=dict(shade=True),plot_kws=dict(s=100,alpha=0.75))
from sklearn.model_selection import train_test_split
X=df.drop('y',axis=1)
y=df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50)
print("Shape of validation set:", X_val.shape)
print("Shape of test set:", X_test.shape)
print("Shape of training set:", X_train.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#Run the classification over a range of trees in the AdaBoost meta-estimator and record accuracy and compute time
import time
val_acc_num_trees=[]
val_f1_num_trees=[]
train_acc_num_trees=[]
train_f1_num_trees=[]
time_adaboost=[]
val_range=(1,21,1)
for i in range(val_range[0],val_range[1],val_range[2]):
    t1=time.time()
    # Fitting
    adaboost = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=20,max_depth=2),
                            n_estimators=i,learning_rate=0.2)
    adaboost.fit(X_train,y_train)
    pred_train = adaboost.predict(X_train)
    pred_val = adaboost.predict(X_val)
    # Accuracy and F1 score
    acc_train=accuracy_score(y_train,pred_train)
    f1_train = f1_score(y_train,pred_train,average='micro')
    acc_val=accuracy_score(y_val,pred_val)
    f1_val = f1_score(y_val,pred_val,average='micro')
    # Appending to the lists
    train_acc_num_trees.append(acc_train)
    train_f1_num_trees.append(f1_train)
    val_acc_num_trees.append(acc_val)
    val_f1_num_trees.append(f1_val)
    t2=time.time()
    time_adaboost.append(t2-t1)
    print(f"Done for number of trees: {i}")
#### Plots of accuracy and compute time
plt.plot(range(val_range[0],val_range[1],val_range[2]),train_acc_num_trees,c='red')
plt.plot(range(val_range[0],val_range[1],val_range[2]),val_acc_num_trees,c='blue')
plt.legend(["Training","Validation"],fontsize=15)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Number of trees in the meta-estimator", fontsize=16)
plt.ylabel("Accuracy",fontsize=16)
plt.ylim(0.5,1.05)
plt.show()

plt.plot(range(val_range[0],val_range[1],val_range[2]),time_adaboost,c='red')
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Number of trees in the meta-estimator", fontsize=16)
plt.ylabel("Model training time (seconds)",fontsize=16)
#plt.ylim(0.7,1.05)
plt.show()
#A simple linear objective function for the business-centric optimization
v = np.array(val_acc_num_trees)
t = np.array(time_adaboost)
v_th = 0.5
v=(v-v_th)/v.max()
t=t/t.max()
alpha = 3
beta = 1
o = alpha*v-beta*(t)
plt.figure(figsize=(7,4))
plt.plot(range(val_range[0],val_range[1],val_range[2]),o,color='k',marker='o',lw=3,markersize='10')
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Number of trees in the meta-estimator", fontsize=16)
plt.ylabel("Linear objective function",fontsize=16)
plt.xticks([i for i in range(2,21,2)])
plt.show()
def objective(x, v_th=0.5,alpha=3,beta=1):
    """
    Objective function computing a overall cost function
    involving the running time and performance of the AdaBoost classifier fitting
    x: number of trees to be used by the meta-estimator (AdaBoost)
    V_th: Minimum accuracy threshold
    alpha: Cost factor of the accuracy (indicator of profit)
    beta: Cost factor of running time (indicator of expense)
    """ 
    x = int(x)
    if x<1:
        x=1
    t1=time.time()
    # Fitting
    adaboost = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=20,max_depth=2),
                            n_estimators=x,learning_rate=0.2)
    adaboost.fit(X_train,y_train)
    t2=time.time()
    pred_train = adaboost.predict(X_train)
    pred_val = adaboost.predict(X_val)
    # Accuracy and F1 score
    acc_train=accuracy_score(y_train,pred_train)
    f1_train = f1_score(y_train,pred_train,average='micro')
    acc_val=accuracy_score(y_val,pred_val)
    f1_val = f1_score(y_val,pred_val,average='micro')
    
    v = acc_val # Validation set accuracy
    t = t2-t1 # Time taken for fitting and calculating the accuracy
    
    # Normalize the accuracy and time measures
    #v=(v-v_th)/v.max()
    #t=t/t.max()
    
    # Objective function (a profit measure)
    obj = alpha*v-beta*(t)
    
    return -float(obj)
for i in range(1,21):
    print(objective(i,0.5,5,1),end=', ')
n=np.arange(1,21)
plt.plot(n,list(map(objective,n)),color='k',marker='o',lw=3,markersize='10')
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Number of trees in the meta-estimator", fontsize=16)
plt.ylabel("Linear objective function",fontsize=16)
plt.xticks([i for i in range(2,21,2)])
plt.show()
# Calling minimize_scalar from Scipy
from scipy.optimize import minimize_scalar
r=minimize_scalar(objective,bounds=(2,21),options={'disp':True},method='Bounded')
r.items()
def objective_dataset(x, X_train,y_train,X_val,y_val,v_th=0.5,alpha=3,beta=1):
    """
    Objective function computing a overall cost function
    involving the running time and performance of the AdaBoost classifier fitting
    x: number of trees to be used by the meta-estimator (AdaBoost)
    V_th: Minimum accuracy threshold
    alpha: Cost factor of the accuracy (indicator of profit)
    beta: Cost factor of running time (indicator of expense)
    """ 
    x = int(x)
    if x<1:
        x=1
    t1=time.time()
    # Fitting
    adaboost = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=20,max_depth=2),
                            n_estimators=x,learning_rate=0.2)
    adaboost.fit(X_train,y_train)
    t2=time.time()
    pred_train = adaboost.predict(X_train)
    pred_val = adaboost.predict(X_val)
    # Accuracy and F1 score
    acc_train=accuracy_score(y_train,pred_train)
    f1_train = f1_score(y_train,pred_train,average='micro')
    acc_val=accuracy_score(y_val,pred_val)
    f1_val = f1_score(y_val,pred_val,average='micro')
    
    v = acc_val # Validation set accuracy
    t = t2-t1 # Time taken for fitting and calculating the accuracy
    
    # Normalize the accuracy and time measures
    #v=(v-v_th)/v.max()
    #t=t/t.max()
    
    # Objective function (a profit measure)
    obj = alpha*v-beta*(t)
    
    return -float(obj)
for _ in range(5):
    X, y = datasets.make_hastie_10_2(n_samples=2000, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50)
    result=minimize_scalar(objective_dataset,bounds=(2,20),args=(X_train,y_train,X_val,y_val),method='Bounded')
    print(result['x'])
    print("-"*15)


#########################################################################

