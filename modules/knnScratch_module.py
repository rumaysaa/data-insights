import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split

class KnnS:
    def __init__(self,n_neighbors):
        self.n_neighbors=n_neighbors

    def euclidean_distance(self,row1, row2):
        distance = 0.0
        for i in range(len(row1)):
            distance += (row1[i] - row2[i])**2
        return sqrt(distance)

    def get_neighbors(self,train, test, k):
        distances = []
        n=[]
        for test_row in test:
            dist = [self.euclidean_distance(test_row, train_row) for train_row in train]
            distances.append(dist)
        for i in range(len(distances)):
            n.append(np.argsort(distances)[i][:k])
        return n
    
    def fit(self,x,y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,random_state=42 )

    def predict(self,x_new):
        ng = self.get_neighbors(self.x_train,x_new,self.n_neighbors)
        li = []
        pred=[]
        for irow in ng:
            li.append(self.y_train[irow])
        for l in li:
            pred.append(np.argmax(np.bincount(l.astype(int))))
        return pred
    