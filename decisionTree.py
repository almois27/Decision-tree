import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DecisionTree(object):
    def __init__(self, target, max_depth, min_sample_leaf,criterion):
        self.max_depth = max_depth
        self.min_sample_leaf=min_sample_leaf
        self.criterion=criterion
        self.nodes=[]
        self.depth=0
        self.target=target
        self.classif=False
        if self.criterion=='entropy' or self.criterion=='gini':
            self.classif=True
            
    
    def split_criterion(self,y):
        if self.criterion=='entropy':
            return self.entropy(y)
        elif self.criterion=='gini':
            return self.gini(y)
        elif self.criterion=='variance':
            return self.variance(y)
        elif self.criterion=='madmedian':
            return self.madmedian(y)
        
    def entropy(self,y):
        N=len(y)
        Ni=np.unique(y, return_counts=True)[1]
        Pi=Ni/N
        
        tmp=-1.0*Pi*np.log2(Pi)
        return tmp.sum()
    
    def gini(self,y):
        N=len(y)
        Ni=np.unique(y, return_counts=True)[1]
        Pi=Ni/N
        tmp=Pi*Pi
        return np.round(1-tmp.sum(),2)
    def variance(self,y):
        N=len(y)
        ymean=y.sum()/N
        tmp=(y-ymean)*(y-ymean)              
        return (tmp.sum())/N
    def madmedian(self,y):
        N=len(y)
        ymed=y.median()
        tmp=abs((y-ymed)) 
        return tmp.sum()/N
    
    def values(self,col,data):
        return data[col].unique()
    
       #x=data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
       # if x[col].dtype!='object':           
        #    #отсортировать и разделить
        #    tmp=data.sort_values(by=col)
         #   meanVal=dict()
         #   res=[]
         #   prevVal=tmp[self.target].iloc[0]
         #   s=[tmp[col].iloc[0]]
         #   for e in np.arange(tmp[col].size):
                
           #     if tmp[self.target].iloc[e]==prevVal:                    
           #         s.append(tmp[col].iloc[e])
          #      else:
            #        m=np.array(s).mean()
            #        res.append(m)
             #       meanVal[m]=prevVal
             #       s=[tmp[col].iloc[e]]
             #       prevVal=tmp[self.target].iloc[e]
          #  return res
        
         
    def findBestPred(self,col,data): 
        #Ищем предикат, который разбивает исходное множество
        #таким образом чтобы уменьшилось среднее значение энтропии
        y=data[self.target]
       
        max_IG=0
        best_pred=''
        xvalues=[]
       
        S0,S1,S2=1,1,1
        S0=self.split_criterion(y)
                
        
        
        xvalues=self.values(col,data)
        for v in xvalues: 
            left=data[data[col]<v]
            a1=len(left)/len(data)
            if len(left)<=1:
                S1=0
            else:
                S1=self.split_criterion(left[self.target])
            
            right=data[data[col]>=v]
            a2=len(right)/len(data)
            if len(right)<=1:
                S2=0
            else:
                S2=self.split_criterion(right[self.target])
                          
            IG=S0-a1*S1-a2*S2
            if IG>max_IG:                
                max_IG=IG
                best_pred=v
        
        
        return [best_pred,max_IG]
    
    def findBestSplit(self,data):
        #лучшее разбиение данных
        X=data.loc[:, data.columns != self.target]
        Y=data[self.target]        
        global_max_IG=0
        best_split=[]
        for col in X.columns:                      
            best_pred,loc_IG=self.findBestPred(col,data)
            if loc_IG>global_max_IG:
                global_max_IG = loc_IG
                best_split= [col,best_pred, global_max_IG]
                
        return best_split
    
    def build_tree(self, data,depth=0,side=" root"):
        if (len(data) <self.min_sample_leaf) or (depth >= self.max_depth):
            node={'depth':depth,"side": side, "position": "leaf","attr":"none","value":-1,"targets":data[self.target]}
            self.nodes.append(node)
            return  len(self.nodes)-1
        elif self.all_same(data):
            node={'depth':depth,"side": side, "position": "leaf","attr":"none","value":-1,"targets":data[self.target]}
            self.nodes.append(node)
            return len(self.nodes)-1
       
        else:
            col,pred,IG = self.findBestSplit(data)               
            left = data[data[col] < pred]
            right = data[data[col] >= pred]      
            li=self.build_tree(left, depth+1," left")
            ri=self.build_tree(right, depth+1," right")
            node={'depth':depth,"side": side, "position": "node","attr":col,"value":pred,"targets":data[self.target],"left":li,"right":ri}
            self.nodes.append(node)
            self.depth += 1 
            return len(self.nodes)-1     
        
    def fit(self, data):
            self.build_tree(data,depth=0,side=" root")
            return self
               
          
    def all_same(self, items):
        return items[self.target].nunique()==1
    
    def printTree(self):
        for n in self.nodes:
            for key,val in n.items():
                if key=="targets":
                    if self.classif:
                        print(key,":",val.value_counts())
                    else:
                         print(key,":",val.mean())
                else:
                    print(key,":", val)
            print()
            
    def predict(self,data):
        res=[]
        for index, row in data.iterrows():            
            res.append(self.get_predict(len(self.nodes)-1,row))
        return res
       
    def get_predict(self,node_indx, row):      
        
        node=self.nodes[node_indx]
        if node["position"]=="leaf":
            if  self.classif:            
                return node["targets"].value_counts().index[0], row[self.target]
            else:
                 return node["targets"].mean(), row[self.target]
        if row[node["attr"]]<node["value"]:            
            self.get_predict(node["left"],row)
        else:
            self.get_predict(node["right"],row)
            
   
                                               

            
if __name__ == '__main__':
    print("Classification:")
    data = pd.read_csv('iris.csv') 
    train, test = train_test_split(data, test_size=0.2)
    
    tree_class=DecisionTree('variety',1000, 1,'gini')
    tree_class.fit(train)
    tree_class.printTree()
    print()
    predict=tree_class.predict(test)
    print()
    print()
    
    
    #print("Regression:")
    #data=pd.read_csv('regression.csv')
    #train, test = train_test_split(data, test_size=0.2)
    #tree_reg=DecisionTree('Price',100, 1,'madmedian')
    #tree_reg.fit(train)
    #tree_reg.printTree()
    print()
    print("predict:")
   # predict=tree_reg.predict(test)
  