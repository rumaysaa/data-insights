from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class randomForest():
    def __init__(self):
        self.model = RandomForestClassifier()
    def fit(self,x,y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,random_state=42 )
        self.model.fit(self.x_train,self.y_train)
    def predict(self,xnew):
        pred = self.model.predict(xnew)
        return pred