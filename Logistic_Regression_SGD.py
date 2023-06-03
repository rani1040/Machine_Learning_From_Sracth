class Logistic_Regression_SGD:
    def __init__(self):
        self.theta = None
        self.coef_ = None
        self.intercept_ = None
        
    # Sigmoid Function
    def g(self,z):
        # Apply np.clip to limit the range of z to avoid overflow error 
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    
    def fit(self,X, y, num_epochs = 150 , learning_rate = 0.1):
        X = np.insert(X,0,1,axis=1)
        self.theta = np.random.randn(X.shape[1])
        
        
        for epoch in range(1,num_epochs+1):
            for i in range(len(X)): 
                random_index = np.random.randint(0,X.shape[0])
                
                xi = X[random_index]
                yi = y[random_index]
                z = np.dot(xi,self.theta)
                gradient =  np.dot(yi - self.g(z),xi)
                self.theta = self.theta + (learning_rate/ np.sqrt(epoch)) * gradient
        

        def predict(self, x_test):
                X = np.insert(0,1,axis=1)
                z = np.dot(X,self.theta)
                h = self.g(z)
                prediction = np.where(h>= 0.5, 1 , 0)
                return prediction
            
         
    
