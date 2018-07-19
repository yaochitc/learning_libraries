from sklearn.preprocessing import Binarizer, LabelEncoder, OneHotEncoder

onehot_encoder = OneHotEncoder()
label_encoder = LabelEncoder()

x = ['a', 'b', 'c']

label_x = label_encoder.fit_transform(x).reshape([len(x), 1])
print(label_x)
print(onehot_encoder.fit_transform(label_x).toarray())

binarizer = Binarizer(threshold=1.0).fit(label_x)
print(binarizer.transform(label_x))