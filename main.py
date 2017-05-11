
import pandas as pd # Need to work with database
import matplotlib.pyplot as plt # For Plotting fatures

# Some of the functions are imported from SciKit Learn library as needed
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'classes']
dataset = pd.read_csv(url, names=features)

# class distribution
print(dataset.groupby('classes').size())

n_samples, n_features = dataset.shape
print("No. of Sample: " + str(n_samples))
print("No. of Features: " + str(n_features))
print(dataset.describe()) # Statastical description about the daataset to understand more

# plotting features
x_index = 3
colors = ['blue', 'red', 'green']

##for label, color in zip(range(len(colors)), colors):
##    plt.hist(dataset[dataset.classes==label, x_index],label=iris.target_names[label],color=color)
##plt.xlabel(dataset.classes[x_index])
##plt.legend(loc='upper_right')
##dataset.hist()

# Scatter Diagram to understand relation between petal length and sepal length
for indx in range(3):
    plt.scatter(dataset.petal_length[(indx*50):(indx+1)*50],dataset.sepal_length[(indx*50):(indx+1)*50], color = colors[indx])
plt.xlabel(features[2])
plt.ylabel(features[3])
plt.show()

# Histogram diagram to get visual understanding of feature distribution in 2 dimension space
for indx in range(3):
    plt.hist(dataset.petal_width[(indx*50):(indx+1)*50], color = colors[indx])
plt.xlabel(features[3])
plt.ylabel("Size in (cm)")
plt.show()

##plt.scatter(dataset.sepal_length[(indx*50):(indx+1)*50],dataset.sepal_width[(indx*50):(indx+1)*50], color = colors[indx])
##dataset.sepal_length[:50].hist()
##dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# Classification Starts from Here:
array = dataset.values 
x = array[:,:4] # To get only features from whole dataset
y = array[:,4] # Array of classes in the same sequence as features
val_size = 0.2 #  Validation size: 20% -> 80% Training Dataset, 20% Testing Datatset
no_seed = 7 # To get some randomness in traning
x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size = val_size, random_state = no_seed)
scoring = 'accuracy'
models = []
models.append(('KNN', KNeighborsClassifier()))

results = []
names = []
"""
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=no_seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring= scoring)
    results.append(cv_results)
    names.append(name)
##    msg = name + ":" + str(cv_results.mean()) + "(" + str(cv_results.std()) + ")"
##    print(msg)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    

kfold = model_selection.KFold(n_splits=10, random_state=no_seed)
cv.results = model_selection.cross_val_score(models, x_train, y_train, cv=kfold, scoring='accuracy')
"""

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
pred = knn.predict(x_val)
print(accuracy_score(y_val, pred))
print(confusion_matrix(y_val,pred))
print(classification_report(y_val, pred))

