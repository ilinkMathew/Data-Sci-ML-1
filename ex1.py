from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score 
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
iris = datasets.load_iris()

# Exploring the usage the General Data set API's of iris dataset from scikit learn

# Uncomment the print statements to understand the datasets utilities

#Displays the data
#print iris.data

# Displays  the label
#print iris.target

# prints out the description of the Datasets
#print iris.DESCR

# prints out the  feature names
#print iris.feature_names

#prints out the name of the labels
print iris.target_names

# prints out the dimensions of the iris dataset 
#print iris.data.shape


# Splitting data in to training & Test data 


X_train , X_test , Y_train , Y_test = train_test_split(iris.data,iris.target,test_size=0.4 ,random_state=0)

print X_train.shape

print X_test.shape



####################################### initiating the Decision Tree classifier ############################

clf_Tree = tree.DecisionTreeClassifier()

# Train the Tree classifier
clf_Tree.fit(X_train,Y_train)

#calculating the Accuracy of Decision Tree for Iris Dataset 
accuracy_clfTree = accuracy_score(clf_Tree.predict(X_test),Y_test) 


########################################## Support Vector machine #########################################

## Default weights are assigned for this classifier
clf_linearSVC = svm.LinearSVC()
clf_linearSVC.fit(X_train,Y_train)


accuracy_clf_linearSVC = accuracy_score(clf_linearSVC.predict(X_test),Y_test) 

######################################### Bayes Algo #######################################################
## using Gaussian Distrubution ##

clf_GaussianNB = GaussianNB()
clf_GaussianNB.fit(X_train,Y_train)


accuracy_clf_GaussianNB = accuracy_score(clf_GaussianNB.predict(X_test),Y_test) 

	
################################################## KNN Algo ################################################
clf_KNN = KNeighborsClassifier(n_neighbors=5)
clf_KNN.fit(X_train,Y_train)

accuracy_clf_KNN = accuracy_score(clf_KNN.predict(X_test),Y_test)

#####################################################################
print iris.DESCR

print "Train Data (size,attributes):" , X_train.shape
print "Test Data (size,attributes):" , X_test.shape

table_header = ["Sl.No" ,"Classifier Name","Accuracy"]
table_values = [[1,"Decision Tree",accuracy_clfTree],[2,"LinearSVC",accuracy_clf_linearSVC],[3,"Naive Bayes",accuracy_clf_GaussianNB],[4,"KNN",accuracy_clf_KNN]]

print tabulate(table_values,table_header,tablefmt="grid")
