from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size=0.8)

rf = RandomForestRegressor(n_estimators=1000)

rf.fit(x_train, y_train)

categorical_features = np.argwhere(np.array([len(set(boston.data[:,x])) for x in range(boston.data.shape[1])]) <= 10).flatten()
explainer = LimeTabularExplainer(x_train, categorical_features=categorical_features, feature_names=boston.feature_names, class_names=['price'], verbose=True, mode='regression')

exp = explainer.explain_instance(x_test[0], rf.predict, num_features=5)

print(exp.as_list())