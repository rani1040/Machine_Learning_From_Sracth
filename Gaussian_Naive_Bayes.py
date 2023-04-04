import pandas as pd
import numpy as np
from math import e,pi
from sklearn.model_selection import train_test_split
 
class GaussianNB:
    
    def __init__(self):
        self.summaries= {}
        
        
    def group_by_class(self,data,y_train,target):
        group_df={}
        for tar in target:
            group_df[tar] = x_train[y_train==tar]
        return group_df
    
    def cal_mean(self,numbers):
        # calculatiing mean 
        result = sum(numbers) / float(len(numbers))
        return result
    
    def standard_deviation(self,numbers):
        mean= self.cal_mean(numbers)
        square_diff = []
        for num in numbers:
            x = (num - mean)**2
            square_diff.append(x)
        square_diff_sum = sum(square_diff)
        n = float(len(numbers)-1)
        var = square_diff_sum / n
        return var ** .5
    
    def summarize(self,test_set):
        for feature in zip(*test_set):
            yield{
                'standard_deviation': self.standard_deviation(feature),
                'mean' : self.cal_mean(feature)
            }

    def prior_proab(self,group_df,target,data):
        return len(group_df[target]) / float(len(data))
    
    
    def normal_pdf(self,x,mean,std):
        exp_square = (x - mean)**2
        variance = std ** 2
        exp_pow = - exp_square / (2 * variance)
        exponent = e ** exp_pow
        deno =  ((2 * pi ) ** 0.5) * std
        normal_prob = exponent / deno
        return normal_prob
        
    def joint_probablities(self,rows):
        joint_probs = {}
        for target, features in self.summaries.items():
            total_features = len(features['summary']) 
            likelihood = 1 
            for index in range(total_features):
                feature = rows[index]
                mean = features['summary'][index]['mean']
                std = features['summary'][index]['standard_deviation']
                normal_prob = self.normal_pdf(feature, mean, std)
                likelihood *= normal_prob
            prior_prob = features['prior_prob']
            joint_probs[target] = prior_prob * likelihood
        return joint_probs
    
    def marginal_pdf(self,joint_prob):
        marginal_prob = sum(joint_prob.values())
        return marginal_prob
    
    def posterior_proability(self,test_row):
        posterior_prob = {}
        joint_probablities = self.joint_probablities(test_row)
        marginal_prob = self.marginal_pdf(joint_probablities)
        for target, joint_prob in joint_probablities.items():
            posterior_prob[target] = joint_prob / marginal_prob
        return posterior_prob
    
    def get_maximum_a_posterior(self,test_row):
        posterior_prob = self.posterior_proability(test_row)
        map_prob = max(posterior_prob, key=posterior_prob.get)
        return map_prob

    def predict(self,x_test):
        map_probs=[]
        for row in x_test:
            map_prob = self.get_maximum_a_posterior(row)
            map_probs.append(map_prob)
        return map_probs
    
    def accuracy(self,test_set,predicted):
        correct = 0
        for x,y in zip(test_set,predicted):
            if x==y:
                correct+=1
        return correct/float(len(test_set))
    
    def train(self,data,y_train):
        target = np.unique(y_train)
        group= self.group_by_class(data,y_train,target)
        for tar, features in group.items():
            self.summaries[tar] = {
                'prior_prob' : self.prior_proab(group,tar,data),
                'summary' : [i for i in self.summarize(features)]
            }
        return self.summaries
    

def main():
    x_train, x_test , y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=42) 
    nb = GaussianNB()
    nb.train(x_train, y_train)
    y_pred = nb.predict(x_test)
    print(nb.accuracy(y_test,y_pred))
    
    
if __name__ == '__main__':
    main()
