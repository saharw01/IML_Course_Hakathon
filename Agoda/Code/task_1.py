import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
	dataset = pd.read_csv('../Data/agoda_cancellation_train.csv')
	dataset['Mycol'] = pd.to_datetime(dataset['Mycol'], format='%d%b%Y:%H:%M:%S.%f')
	X = dataset.drop('cancellation_datetime', axis=1).to_numpy()
	y = dataset.loc[:, 'cancellation_datetime'].to_numpy()

	# Splitting the dataset into the Training set and Test set
	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
	# Feature Scaling
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	x_train = sc.fit_transform(x_train)
	x_test = sc.transform(x_test)
	return x_train, x_test, y_train, y_test


def fit_lgbm_classifier(X, y):
	from lightgbm import LGBMClassifier
	from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
	# evaluate the model
	model = LGBMClassifier()
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
	# fit the model on the whole dataset
	model = LGBMClassifier()
	model.fit(X, y)
	return model


if __name__ == '__main__':
	x_train, x_test, y_train, y_test = load_data()
	fitted_model = fit_lgbm_classifier(x_train, y_train)
	# fitted_model.predict(x_test)
