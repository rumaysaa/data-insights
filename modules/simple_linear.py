import numpy as np
from sklearn.model_selection import train_test_split

class SLinearReg:

    def __init__(self,epochs,learning_rate,gd,batchsize):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.gd = gd
        self.batchsize=batchsize

    def gradient_descent(self):
        cost=[]
        m_curr=b_curr=0
        n = len(self.x_train)
        for i in range(self.epochs):
            y_predicted = (m_curr * self.x_train) + b_curr
            cost.append((1/n) * sum(val**2 for val in (self.y_train-y_predicted))[0])
            md = -(2/n)*sum(self.x_train*(self.y_train-y_predicted))
            bd = -(2/n)*sum(self.y_train-y_predicted)
            m_curr = m_curr - (self.learning_rate * md)
            b_curr = b_curr - (self.learning_rate * bd)
            self.cost = cost
            #print(f"m {m_curr}, b {b_curr}, cost {cost}, iteration {i}")
        return m_curr,b_curr
    
    def stoc_gradient_descent(self):
        cost=[]
        m_curr=b_curr=0
        n = len(self.x_train)
        for e in range(self.epochs):
            for index in range(len(self.x_train)):
                xs = self.x_train[index]
                ys = self.y_train[index]
                N = len(xs)
                f = ys - (m_curr*xs + b_curr)
                m_curr -= self.learning_rate * (-2 * xs.dot(f).sum() / N)
                b_curr -= self.learning_rate * (-2 * f.sum() / N)   
                cost.append((1/n) * sum(val**2 for val in f))
            self.cost=cost
            return m_curr,b_curr
        
    def minibatch_gradient_descent(self):
        m,b=0,0
        N = len(self.x_train)
        cost=[]
        for i in range(0,N,self.batchsize):
            x_batch = self.x_train[i:i+self.batchsize]
            y_batch = self.y_train[i:i+self.batchsize]
            f = y_batch - ((m*x_batch) + b)
            #print(x_batch,f)
            m -= self.learning_rate * (-2 * x_batch.T.dot(f).sum() / N)
            b -= self.learning_rate * (-2 * f.sum() / N)
            cost.append(((1/N) * sum(val**2 for val in f))[0])
        self.cost=cost
        return m,b

    def fit(self,x,y):
        self.x=x
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,random_state=42 )
        if(self.gd=='Batch'):
            self.m,self.c = self.gradient_descent()
        if(self.gd=='Stochastic'):
            self.m,self.c = self.stoc_gradient_descent()
        if(self.gd=='Mini Batch'):
            self.m,self.c = self.minibatch_gradient_descent()
    def predict(self, new_x):
        predicted = (self.m*new_x) + self.c
        return predicted
    

    

    