from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, features_train, labels_train):
        self.features_train = features_train 
        self.labels_train = labels_train
        

    def predict(self, features_test):
        predictions = []
        for item in features_test:
            # determine which other point is closest 
            labels = self.closest(item)
            predictions.append(labels)

        return predictions

    def closest(self, item):
        best_dist = euc(item, self.features_train[0])
        best_index = 0 
        for i in range(1,len(self.features_train)):
            dist = euc(item, self.features_train[i])
            if(dist < best_dist):
                best_dist = dist
                best_index = i

        return self.labels_train[best_index]


iris = datasets.load_iris()
features = iris.data
labels = iris.target 

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.5)
# print(features_train)
print(features_test)
# print(labels_train)
classifier = KNeighborsClassifier()
# classifier = ScrappyKNN()
f = classifier.fit(features_train, labels_train)
prediction = classifier.predict(features_test)
predicted_score = accuracy_score(labels_test, prediction)

print(predicted_score)