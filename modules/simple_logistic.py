import numpy as np
from sklearn.model_selection import train_test_split

class Slor:

    def stoc_gradient_descent(self):
        cost=[]
        m_curr=b_curr=0
        n = len(self.x_train)
        for e in range(100):
            for index in range(len(self.x_train)):
                xs = self.x_train[index]
                ys = self.y_train[index]
                N = len(xs)
                f = ys - (m_curr*xs + b_curr)
                m_curr -= (0.01 * ((-2 * xs.dot(f).sum()) / N))
                b_curr -= (0.01 * ((-2 * f.sum() / N)))
                cost.append((1/n) * sum(val**2 for val in f))
        self.cost=cost
        return m_curr,b_curr

    def fit(self,X,Y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2,random_state=42 )
        x = self.x_train
        y = self.y_train
        self.x_mean = np.mean(x)
        self.y_mean = np.mean(y)
        self.m = ((x- self.x_mean)*(y- self.y_mean)).sum()/((x-self.x_mean)**2).sum()
        self.c = self.y_mean - (self.m* self.x_mean)

    def predict(self, new_x):
        predicted = 1/(1+(np.exp(-((self.m*new_x) + self.c))))
        return predicted#,np.where(predicted > 0.5, 1, 0)

    

    