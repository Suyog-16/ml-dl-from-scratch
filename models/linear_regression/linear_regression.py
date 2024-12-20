import numpy as np
class LinearRegression():
   def __init__(self,weights,bias,learning_rate=0.01, epochs=20):

      self.weights = None
      self.learning_rate = learning_rate
      self.epochs = epochs
      self.bias = 0
   
   def fit(self,X,y):
      #Initiliaze the weights and bias
      self.weights = np.zeros(X.shape[1])
      for i in range(self.epochs):
         y_pred = np.dot(X,self.weights) + self.bias
         error = y_pred - y

         # updating w and b
         dw = 1/len(X) * np.dot(X.T,error)

         db = 1/len(X) * np.sum(error)

         self.weights -= self.learning_rate * dw
         self.bias -= self.learning_rate * db
   def predict(self,X):
      return np.dot(X,self.weights) + self.bias