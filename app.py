from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from werkzeug import secure_filename
import os

import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

from os.path import relpath

app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def index():
	if request.method == 'POST' and 'train_data' in request.files and 'test_data' in request.files:
		basedir = os.path.abspath(os.path.dirname(__file__))

		#save train data on local machine
		file = request.files['train_data']
		filename = secure_filename(file.filename)
		file.save(os.path.join(basedir,'static/uploadsDB',filename))
		fullfile_train = os.path.join(basedir,'static/uploadsDB',filename)

		#save test data on local machine
		file = request.files['test_data']
		filename = secure_filename(file.filename)
		file.save(os.path.join(basedir,'static/uploadsDB',filename))
		fullfile_test = os.path.join(basedir,'static/uploadsDB',filename)

		#separate target & features
		dfTrain = pd.read_csv(fullfile_train)
		df_nrows = len(dfTrain.columns)
		x_train = dfTrain.iloc[:,0:df_nrows-1]
		y_train = dfTrain.iloc[:,df_nrows-1]

		#normalize the features
		x_train = preprocessing.scale(x_train)

		#feature selection
		#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
		#sel.fit_transform(x_train)

		#create knowledge model
		node1 = request.form['first_hidden_layer']
		node2 = request.form['second_hidden_layer']
		if int(node2) == 0:
			nodes = (int(node1),)
		else:
			nodes = (int(node1),int(node2))
		nn = MLPRegressor(learning_rate_init=0.01,hidden_layer_sizes=nodes,activation=request.form['activation_function'],solver=request.form['solver'])
		nn.fit(x_train,y_train)

		#predict test data
		dfTest = pd.read_csv(fullfile_test)
		df_ncols = len(dfTest.columns)
		x_test = dfTest.iloc[:,0:df_ncols-1]
		y_test = dfTest.iloc[:,df_ncols-1]

		#normalize the features
		x_test = preprocessing.scale(x_test)

		y_pred = nn.predict(x_test)

		#evaluation
		r2 = r2_score(y_test,y_pred)
		rmse = np.mean((y_pred-y_test)**2)

		dfResult = dfTest;
		#dfResult['Prediction'] = y_pred

		plt.scatter(range(y_pred.size),y_pred-y_test,color="b")
		plt.axhline()
		filename = secure_filename('new_plot2.png')
		chart_url = os.path.join(basedir,'static/images',filename)
		plt.savefig(chart_url)
		chart_url_2 = relpath(chart_url, basedir)

		return render_template('index.html',fullfile_train=fullfile_train,r2=r2,rmse=rmse,dfTrain=dfTrain,dfTest=dfTest,dfResult=dfResult,y=zip(y_pred,y_test),plot=chart_url_2,req=request.args,node1=node1,node2=node2,nn=nn,nodes=nodes)
	else:
		return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
