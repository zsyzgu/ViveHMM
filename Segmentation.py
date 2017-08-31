#coding=gbk
import math
class Segmentation:  
    #��ṹ:
    #    n:int              �����
    #    data:[]            ���n֡�Ķ�������
    def __init__(self):  
        self.n=5
    def __del__(self):
        pass
    def __repr__(self):
        return str(dict({"actionName":self.actionName,"typeList":self.typeList,"data":len(self.data)}))
    def __str__(self):
        return str(dict({"actionName":self.actionName,"typeList":self.typeList,"data":self.data}))
    
    def FeatureMatToSpeedVector(self,data):
        #��feature����ת��Ϊ�����ۺ�һά�ٶ�����(����ǰ27ά��ͷ���ָ�9ά�ٶ�)
        result=[]
        for a in data:
            v=[]
            for i in range(int(len(a)/9)):
                v.append(math.sqrt(a[9*i+0]**2+a[9*i+1]**2+a[9*i+2]**2)+1e-10)
            #ans=3/(1/HeadV+1/LHandV+1/RHandV)
            ans=1e10
            for i in v:
                ans=min(ans,i)
            #ans=(HeadV+LHandV+RHandV)/3
            result.append(ans)
        return result
    
    def CleanFeatureMatrix(self,data):
        #��feature�����н�ȡ��������(����ǰ27ά��ͷ���ָ�9ά�����ٶ�)
        beginOffset=0.1
        endOffset=0.1
        beginTime=5 #��ʼ֡��
        endTime=40 #����֡��
        v=self.FeatureMatToSpeedVector(data)
        begin=0
        Cnt=0
        n=len(v)
        for t in range(n):
            if v[t]>beginOffset:
                Cnt+=1
            else:
                Cnt=0
            if Cnt>=beginTime:
                begin=t-beginTime+1
                break
        end=n
        Cnt=0
        for t in range(begin+1,n):
            if v[t]<endOffset:
                Cnt+=1
            else:
                Cnt=0
            if Cnt==endTime:
                end=t-endTime+1
        return data[begin:end]
                
    
    def push(self,data):
        pass
        
    def start(self):
        data=[]
        

if __name__ =='__main__':
    #����
    s = ActionDataSet()
    s.LoadDataFromFile("gyz_0_vec.txt")
    print(s)