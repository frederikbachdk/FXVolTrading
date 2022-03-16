from math import log, sqrt, exp
from re import T
from scipy.stats import norm

class BS:
    def __init__(self,rf,rd,K,T,sigma,S):
        self.rf = rf
        self.rd = rd
        self.K = K
        self.T = T
        self.sigma = sigma
        self.S = S

    def d1(self):
        return (log(self.S/self.K) + (self.rd - self.rf + (self.sigma**2)/2)*self.T)/(self.sigma * sqrt(self.T))

    def d2(self):
        return self.d1()-self.sigma*sqrt(self.T)

    def call(self):
        return self.S*exp(-self.rf*self.T)*norm.cdf(self.d1())-self.K*exp(-self.rd*self.T)*norm.cdf(self.d2())
    
    def put(self):
        return self.K*exp(-self.rd*T)-self.S*exp(-self.rf*T)+self.call()