#coding=gbk
from hmmlearn import hmm
import numpy as np
import random

#HMM分类器
class Classifier:
    #类结构:
    #   models:dict  dict["动作名"]=一个HMM模型变量
    #   data:dict    dict["动作名"]=List[dataMatrix] (dataMatrix是[[]]矩阵)
    
    
    def __init__(self):  
        self.models = dict()
        self.data = dict()
        self.const_components = 5
        
    def __repr__(self):
        ans = ""
        for name,model in self.models.items():
            ans=ans+name+"\n"+str(model.startprob_)+'\n'+str(model.transmat_)+'\n'+str(model.means_)+'\n'+str(model.covars_)+'\n'
        return ans
            
    
    def addData(self,name,data):
        if self.data.get(name)==None:
            self.data[name]=[]
        self.data[name].append(data)
        
    def addDataFromADS(self,actionDataSet):
        for className,dataSet in actionDataSet.action.items():
            size=len(dataSet)
            #TrainDataNum=3
            #for t in range(TrainDataNum):
            #    i=random.randint(0,size-1)
            #    self.addData(className,dataSet[i].featureMat)
            for dataBlock in dataSet:
                if dataBlock.typeList[0]%1==0:
                    self.addData(className,dataBlock.featureMat)
             
    def train_HMM_Model(self,dataset,components):
        #输入参数1为list[]，其中每个元素是列宽相等的二维数组表示数据集合
        #输入参数2状态数
        #输出HMM模型
        model = hmm.GaussianHMM(n_components=components, covariance_type="diag")
        #model = hmm.GMMHMM(n_components=components, n_mix=3, covariance_type="diag")
        X = np.concatenate(dataset)
        lengths=[]
        for i in dataset:
            lengths.append(len(i))
        #print(X,lengths)
        model.fit(X,lengths)
        return model
        
    def fit(self):
        #根据data计算models
        for className,dataSet in self.data.items():
            if(len(dataSet)>1):
                self.models[className]=self.train_HMM_Model(dataSet,self.const_components)
    
    def score_samples(self,X):
        return [{"name":name,"score":model.score_samples(X)} for name,model in self.models.items()]
            
    def scores(self,X):
        return [{"name":name,"score":model.score(X)} for name,model in self.models.items()]
            
    def predict(self,X):
        score_=-1e100
        ans=""
        for name,model in self.models.items():
            s=model.score(X)
            if(s>score_):
                score_=s
                ans=name
        return ans
            
if __name__ =='__main__':
    #单元测试
    HMM=Classifier()
    HMM.addData("type1",[[0],[1],[0],[100],[101],[100]])
    HMM.addData("type1",[[0],[0],[1],[99],[102],[100]])
    HMM.addData("type2",[[0],[1],[0],[10],[11],[10]])
    HMM.addData("type2",[[0],[0],[1],[9],[12],[10]])
    HMM.addData("type3",[[0],[1],[2],[3],[4],[5]])
    HMM.addData("type3",[[-1],[2],[1],[4],[3],[5]])
    HMM.fit()
    print(HMM)
    print("test1:\n")
    print(HMM.score_samples([[0],[1],[0],[3],[99],[101]]))
    print("test2:\n")
    print(HMM.score_samples([[0],[1],[0],[9],[12],[8]]))
    print("test3:\n")
    print(HMM.score_samples([[2],[1],[3],[3],[4],[4]]))