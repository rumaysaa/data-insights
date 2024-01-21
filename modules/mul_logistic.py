import pickle
import numpy as np
from sklearn.model_selection import train_test_split
class Mlor:
    def __init__(self):
        with open('./pickle_files/mul_logistic_regression.pkl', 'rb') as file:
            self.model = pickle.load(file)
        
    def fit(self,x,y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y,test_size=0.2,random_state=42 )
        self.model.fit(self.x_train,self.y_train)

    def predict(self,x):
        ypredict = self.model.predict(x)
        return ypredict

    def predict_proba(self,x):
        probas = 1/(1+np.exp(-(np.dot(x, self.model.coef_.T)+self.model.intercept_)))
        return probas