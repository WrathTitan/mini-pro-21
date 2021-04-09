from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 

import pandas as pd
import numpy as np
import requests
import random
import pickle


app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///user.db'
app.config['SQLALCHEMY_BINDS']={'datadb':'sqlite:///data.db','trainedmodeldb':'sqlite:///trainedmodel.db','projectinfodb':'sqlite:///projectinfo.db','projectsdb':'sqlite:///projects.db'}

db=SQLAlchemy(app)

class UserDB(db.Model):
	userID=db.Column(db.Integer,primary_key=True)
	username=db.Column(db.String(150),nullable=False)
	password=db.Column(db.String(150),nullable=False)
	def __init__(self,userID,username,password):
		self.userID=userID
		self.username=username
		self.password=password

class ProjectInfoDB(db.Model):
	__bind_key__='projectinfodb'
	projectUserID=db.Column(db.Integer,primary_key=True,nullable=False)
	userID=db.Column(db.Integer,nullable=False)
	def __init__(self,projectUserID,userID):
		self.projectUserID=projectUserID
		self.userID=userID
	
class ProjectsDB(db.Model):
	__bind_key__='projectsdb'
	projectID=db.Column(db.Integer,primary_key=True,nullable=False)
	dataID=db.Column(db.Integer,nullable=False)
	projectname=db.Column(db.String(100))
	projectdata=db.Column(db.PickleType)		#This is PickleType - Can be useful
	def __init__(self,projectID,dataID,projectname,projectdata):
		self.projectID=projectID
		self.dataID=dataID
		self.projectname=projectname
		self.projectdata=projectdata	

class TrainedModelsDB(db.Model):
	__bind_key__='trainedmodeldb'
	modelname=db.Column(db.String(100),primary_key=True,nullable=False)
	parameters=db.Column(db.String(100),nullable=False)
	weights=db.Column(db.String(100))
	modelmetrics=db.Column(db.String(100))
	def __init__(self,modelname,parameters,weights,modelmetrics):
		self.modelname=modelname
		self.parameters=parameters
		self.weights=weights
		self.modelmetrics=modelmetrics

class DataDB(db.Model):
	__bind_key__='datadb'
	dataID=db.Column(db.String(100),primary_key=True,nullable=False)
	traineddata=db.Column(db.PickleType,nullable=False)
	validationdata=db.Column(db.PickleType)
	def __init__(self,dataID,traineddata,validationdata):
		self.dataID=dataID
		self.traineddata=traineddata
		self.validationdata=validationdata
db.create_all()

myUserObject=UserDB(15,"UserTheGreat","PasswordTheStrong")
myMLObject=TrainedModelsDB("KNN","Params","Weights","Metrics")

	
infile = open('pickle_folder/modellr.pkl','rb')
model_classifier = pickle.load(infile)
infile.close()
#type(model_classifier)
testdbobject=DataDB(45,model_classifier,model_classifier)


X_train=pd.read_csv('csvfiles/dftraincls.csv')
X_test=pd.read_csv('csvfiles/dftestcls.csv')
Y_train=pd.read_csv('csvfiles/ytraincls.csv')
Y_test=pd.read_csv('csvfiles/ytestcls.csv')

modeltype="Regression" #Give Classification Or Regression
#if modeltype=="Classification":
y_pred=model_classifier.predict(X_train)
model_accuracy=metrics.accuracy_score(Y_train,y_pred.round())
model_f1_score=metrics.f1_score(Y_train,y_pred.round())
model_precision=metrics.precision_score(Y_train,y_pred.round())
model_recall=metrics.recall_score(Y_train,y_pred.round())
#elif modeltype=="Regression":
y_pred=model_classifier.predict(X_train)
model_rmse=metrics.mean_squared_error(Y_train,y_pred)
model_r2score=metrics.r2_score(Y_train,y_pred)

@app.route("/")
def myfunction():
	return render_template("index.html",myUserObject=myUserObject,myMLObject=myMLObject,testdbobject=testdbobject,modeltype=modeltype,model_accuracy=model_accuracy,model_f1_score=model_f1_score,model_precision=model_precision,model_recall=model_recall,model_rmse=model_rmse,model_r2score=model_r2score)

@app.route('/second',methods=['POST'])
def secondfunction():
	'''myUserObject=UserDB(9999,"User","Pwd")
	db.session.add(myUserObject)
	db.session.commit()
	
	myMLObject=TrainedModelsDB("RF3","Params2","Weights2","Metrics2")
	db.session.add(myMLObject)
	db.session.commit()
	
	testdbobject=DataDB(4545454,model_classifier,model_classifier)
	db.session.add(testdbobject)
	db.session.commit()
	
	myprojectobject=ProjectsDB(4545,45,'KNN Pickle',model_classifier)
	db.session.add(myprojectobject)
	db.session.commit()
	'''
	testobjectshowUserDB=ProjectsDB.query.get_or_404(4545)
	
	return f"This is the second page {testobjectshowUserDB.projectID} and now I'll show the pickle stuff {testobjectshowUserDB.projectdata}"



'''
@app.route("/update",methods=['POST'])
def myDBfunction():
	#mycurrentuser=UserDB.query.get_or_404(userID)
	if request.method=='POST':
		try:
			db.session.add(myUserObject)
			db.session.commit()
			return redirect('/')
		except:
			return 'There was an error updating that task'
	else:
		return render_template('index.html',myUserObject=myUserObject,myMLObject=myMLObject,testdbobject=testdbobject)
'''

if __name__=='__main__':
	app.run(debug=True)
