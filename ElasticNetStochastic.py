import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ElasticNet():
    #eta is the learning rate and lambda1 and labda2 are the l1 and l2 penalties, repsectively
    #iterations is the amount of interations that will be used for gradient descent
    def __init__(self,eta,lambda1,lambda2,iterations):
        self.eta = eta
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.iterations = iterations
       
       
    ##Just like in sklearn, this fits the given data to the model  
    #X is the deisgn matrix and Y is the response vector
    def fit(self,X,Y):
        #nexp is the number of data points/number of experiments
        self.nexp = X.shape[0]
        #npred is the number of predictors
        self.npred = X.shape[1]
       
        #initialize weights
        self.betas = np.zeros(self.npred)
        self.intercept = 0.0
       
        #store design matrix and response vector
        self.X = X
        self.Y = Y
       
        #Now, use gradient descent to find the weights
        for i in range(self.iterations):
            self.GradientDescent()
       
        return self
   
    #Similar but not the same as Mithcell page 93, since there is a different function to be minimized
    #The loss function to minimize is the mean of the
    #squared error plus the l1 and l2 penalties
    def GradientDescent(self):
        
        #Find predicted output Y_pred
        Y_pred = self.predict(self.X)
       
        #intercept is its own case
        self.intercept = self.intercept-(-2*self.eta*np.sum(self.Y-Y_pred)/self.nexp)
       
        for i in range(self.npred):
            #Because of absolute value in the l1 penalty, must have two cases for the non-intercept terms
            if self.betas[i]>0:
                self.betas[i] = self.betas[i]-((-2*self.eta*(np.dot(self.X[:,i],(self.Y-Y_pred))) +
                         self.lambda1+2*self.lambda2*self.betas[i])/self.nexp)
            else:
                self.betas[i] = self.betas[i]-((-2*self.eta*(np.dot(self.X[:,i],(self.Y-Y_pred))) -
                         self.lambda1+2*self.lambda2*self.betas[i])/self.nexp)
                     
    #Predict function just like from sklearn
    def predict(self,X):
        Y_pred = np.matmul(X,self.betas)+self.intercept
        return Y_pred
    
    #Get R2 Value
    def getr2(self,Y,Ypred):
        Yave = np.mean(Y)*np.ones(len(Y))
        TSS = sum((Y-Yave)**2)
        RSS = sum((Y-Ypred)**2)
        R2=1-RSS/TSS
        return R2
    
    #Get Adjusted R2 Value
    def getr2a(self,R2):
        R2a = 1-(1-R2)*(self.nexp-1)/(self.nexp-self.npred-1)
        return R2a
    
    #Get Mean Squared Error
    def getMSE(self,Y,Ypred):
        MSE = (1/len(Y))*sum(Y-Ypred)**2
        return MSE
        