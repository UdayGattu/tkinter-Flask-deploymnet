import tkinter as tk 
from tkinter import messagebox,simpledialog,filedialog
from tkinter import *
import tkinter
from imutils import paths
from tkinter.filedialog import askopenfilename


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import cufflinks as cf
cf.go_offline()
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve 
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split



root= tk.Tk() 
root.title("Identify the common skills and qualifications of the top-performing employees in a company")
root.geometry("1300x1200")

global df
def upload():
	global Data
	Data = askopenfilename(initialdir ="Dataset")
	text.insert(END,"Dataset loaded\n\n")



def data():
    text.delete("1.0",END)
    df = pd.read_csv(Data)
    text.insert(END,'\n\nTop FIVE rows of the Dataset\n\n')
    text.insert(END,df.head())
    text.insert(END,'\n\nNumber Columns\n\n')
    inf = df.columns
    text.insert(END,inf)
    return df


def group():
    text.delete("1.0",END)
    df =pd.read_csv(Data)
    text.insert(END,"\n\nGrouping Categorical Variables\n\n")
    sal = df.groupby("sales").mean()
    text.insert(END,'\n\n'+str(sal)+'\n\n')
    salar =df.groupby("salary").mean()
    text.insert(END,'\n\n'+str(salar)+'\n\n')
    return df





def statistics():
	text.delete('1.0',END)
	global df
	df = pd.read_csv(Data)
	df_stats=df.describe()
	text.insert(END,"\n\nStatistical Measurements for Data\n\n")
	text.insert(END,df_stats)
	null=df.isnull().sum()
	text.insert(END,"\n\nDisplaying Number of Missing Values in all Independent Attributes\n\n")
	text.insert(END,null)




def preprocess():
    global df,x,y
    text.delete("1.0",END)
    text.insert(END,"\t\t\tHandled Missing Values based on value_counts\n\n")
    text.insert(END,"\t\t\t\t\t&\n")
    text.insert(END,"\t\tEncoded Categorical Variables\n\n")
    df=pd.read_csv(Data)
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    ct = ColumnTransformer
    ct = ct([('oh' , OneHotEncoder(),[7,8])],remainder='passthrough')
    x=ct.fit_transform(x)
    text.insert(END,"\n\nEncoding the Categorical Variables\n\n")
    text.insert(END,x)
    sc = StandardScaler()
    x = sc.fit_transform(x)
    text.insert(END,'\n\nPreprocessing\n\n')
    text.insert(END,x)
    return x,y







def train_test():
	global df,x,y,x_train,x_test,y_train,y_test
	text.delete("1.0",END)
	df=pd.read_csv(Data)
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7)
	text.insert(END,"Train and Test Model Generated\n\n")
	text.insert(END,"Total Dataset Size : "+str(len(df))+"\n")
	text.insert(END,"Training Size : "+str(len(x_train))+"\n")
	text.insert(END,"Test Size :"+str(len(x_test))+"\n")
	return x_train,x_test,y_train,y_test



def Rf():
    text.delete("1.0",END)
    global df
    global x_train,x_test,y_train,y_test
    rf =RandomForestClassifier()
    rf=rf.fit(x_train,y_train)
    pred=rf.predict(x_test)
    rf_accuracy=metrics.accuracy_score(y_test,pred)%100
    cr = classification_report(y_test,pred)
    cm = metrics.confusion_matrix(y_test,pred)
    text.insert(END,"\n\nAccuracy score for target:\n\n"+str(rf_accuracy)+'%'+"\n\n")
    text.insert(END,"\n\nPredicted values of input data:\n\n"+str(pred)+"\n\n")
    text.insert(END,"\n\nClassification report:\n\n"+str(cr)+"\n\n")
    text.insert(END,"\n\nconfusion matrix:\n\n"+str(cm)+"\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")



def DT():
    text.delete("1.0",END)
    global df
    global x_train,x_test,y_train,y_test
    dt =DecisionTreeClassifier()
    dt=dt.fit(x_train,y_train)
    pred=dt.predict(x_test)
    dt_accuracy=metrics.accuracy_score(y_test,pred)%100
    cr = classification_report(y_test,pred)
    cm = metrics.confusion_matrix(y_test,pred)
    text.insert(END,"\n\nAccuracy score for target:\n\n"+str(dt_accuracy)+'%'+"\n\n")
    text.insert(END,"\n\nPredicted values of input data:\n\n"+str(pred)+"\n\n")
    text.insert(END,"\n\nClassification report:\n\n"+str(cr)+"\n\n")
    text.insert(END,"\n\nconfusion matrix:\n\n"+str(cm)+"\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")



def LR():
    text.delete("1.0",END)
    global df
    global x_train,x_test,y_train,y_test
    lr =LogisticRegression()
    lr=lr.fit(x_train,y_train)
    pred=lr.predict(x_test)
    lr_accuracy=metrics.accuracy_score(y_test,pred)%100
    cr = classification_report(y_test,pred)
    cm = metrics.confusion_matrix(y_test,pred)
    text.insert(END,"\n\nAccuracy score for target:\n\n"+str(lr_accuracy)+'%'+"\n\n")
    text.insert(END,"\n\nPredicted values of input data:\n\n"+str(pred)+"\n\n")
    text.insert(END,"\n\nClassification report:\n\n"+str(cr)+"\n\n")
    text.insert(END,"\n\nconfusion matrix:\n\n"+str(cm)+"\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")




def AB():
    text.delete("1.0",END)
    global df
    global x_train,x_test,y_train,y_test
    abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
    abc= abc.fit(x_train, y_train)
    pred = abc.predict(x_test)
    abc_accuracy=metrics.accuracy_score(y_test,pred)%100
    cr = classification_report(y_test,pred)
    cm = metrics.confusion_matrix(y_test,pred)
    text.insert(END,"\n\nAccuracy score for target:\n\n"+str(abc_accuracy)+'&'+"\n\n")
    text.insert(END,"\n\nPredicted values of input data:\n\n"+str(pred)+"\n\n")
    text.insert(END,"\n\nClassification report:\n\n"+str(cr)+"\n\n")
    text.insert(END,"\n\nconfusion matrix:\n\n"+str(cm)+"\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")


def roc_graph():
    global df
    global x_train,x_test,y_train,y_test
    rf =RandomForestClassifier()
    rf=rf.fit(x_train,y_train)
    pred=rf.predict(x_test)
    rf_accuracy=metrics.accuracy_score(y_test,pred)%100
    cr = classification_report(y_test,pred)
    cm = metrics.confusion_matrix(y_test,pred)


    dt =DecisionTreeClassifier()
    dt=dt.fit(x_train,y_train)
    pred=dt.predict(x_test)
    dt_accuracy=metrics.accuracy_score(y_test,pred)%100
    cr = classification_report(y_test,pred)
    cm = metrics.confusion_matrix(y_test,pred)


    lr =LogisticRegression()
    lr=lr.fit(x_train,y_train)
    pred=lr.predict(x_test)
    lr_accuracy=metrics.accuracy_score(y_test,pred)%100
    cr = classification_report(y_test,pred)
    cm = metrics.confusion_matrix(y_test,pred)


    abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
    abc= abc.fit(x_train, y_train)
    pred = abc.predict(x_test)
    abc_accuracy=metrics.accuracy_score(y_test,pred)%100
    cr = classification_report(y_test,pred)
    cm = metrics.confusion_matrix(y_test,pred)
    
    
    
    fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:,1])
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(x_test)[:,1])
    dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dt.predict_proba(x_test)[:,1])
    abc_fpr, abc_tpr, abc_thresholds = roc_curve(y_test, abc.predict_proba(x_test)[:,1])

    plt.figure()

    # Plot Logistic Regression ROC
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % lr_accuracy)

    # Plot Random Forest ROC
    plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_accuracy)

    # Plot Decision Tree ROC
    plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_accuracy)

    # Plot AdaBoost ROC
    plt.plot(abc_fpr, abc_tpr, label='AdaBoost (area = %0.2f)' % abc_accuracy)

    plt.xlim([0.0, 1.1])
    plt.ylim([0.0, 1.2])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Graph')
    plt.legend(loc="lower right")
    plt.show()
    return



font = ('times', 14, 'bold')
title = Label(root, text='Identify the common skills and qualifications of the top-performing employees in a company')  
title.config(font=font)           
title.config(height=2, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
button1 = tk.Button(root, text="Upload Dataset",width=13, command=upload)
button1.config(font=font1)
button1.place(x=60,y=100)

pathlabel = Label(root)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

button2 =tk.Button(root, text="Data",width=13, command=data)
button2.config(font=font1)
button2.place(x=60,y=150)


button3 =tk.Button(root, text="Categorical Group" ,width=15,command=group)
button3.config(font=font1)
button3.place(x=60,y=200)

button4 = tk.Button(root, text="statistics",width=13, command=statistics)
button4.config(font=font1) 
button4.place(x=60,y=250)

button5 = tk.Button(root, text="preprocess",width=13, command=preprocess)
button5.config(font=font1)
button5.place(x=60,y=300)

button6= tk.Button(root, text="Train & Test",width=13, command=train_test)
button6.config(font=font1)
button6.place(x=60,y=350) 



title = Label(root, text='Application of ML models')
#title.config(bg='RoyalBlue2', fg='white')  
title.config(font=font1)           
title.config(width=25)       
title.place(x=250,y=70)


button6 = tk.Button (root, text='Random Forest',width=15,bg='pale green',command=Rf)
button6.config(font=font1) 
button6.place(x=300,y=100)



button7 = tk.Button (root, text='Decision Tree',width=15,bg='violet',command=DT)
button7.config(font=font1) 
button7.place(x=300,y=150)


button8 = tk.Button (root, text='Linear',width=15,bg='orange',command=LR)
button8.config(font=font1) 
button8.place(x=300,y=200)



button9 = tk.Button (root, text='AdaBoost',width=15,bg='indian red',command=AB)
button9.config(font=font1) 
button9.place(x=300,y=250)















graph= tk.Button(root, text="roc_Graph", command=roc_graph)
graph.place(x=60,y=400)
graph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(root,height=32,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set,xscrollcommand=scroll.set)
text.place(x=550,y=70)
text.config(font=font1)

root.mainloop()




    


   




	



