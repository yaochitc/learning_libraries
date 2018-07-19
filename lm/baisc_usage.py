import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)

rf = RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)

c = make_pipeline(vectorizer, rf)

explainer = LimeTextExplainer(class_names=class_names)

idx = 81
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])

fig = exp.as_pyplot_figure()

plt.show()