#coding=gbk
import math
import Segmentation as sg
class ActionData:  
    #��ṹ:
    #    actionName:string  ������
    #    typeList:int[]     ��ǩ�б�(ÿ��λ���в�ͬ�ĺ���)
    #    data:[[]]          ��ά��֡����
    #        data[t]:       [0]:ʱ��s  ֮������������ʾһ����������
    #    featureMat         ��������
    def __init__(self, actionName_, typeList_, data_):  
        #print(actionName_)
        self.actionName = actionName_
        self.typeList = typeList_
        self.data = data_
        self.featureMat = self.ToFeatureMatrix()
        self.featureMat=sg.Segmentation().CleanFeatureMatrix(self.featureMat)
    def __del__(self):
        pass
    def __repr__(self):
        return str(dict({"actionName":self.actionName,"typeList":self.typeList,"data":len(self.data)}))
    def __str__(self):
        return str(dict({"actionName":self.actionName,"typeList":self.typeList,"data":self.data}))
            
    def MidFilter(self,data):
        #������ֵ�˲�
        FilterD=9 #����D֡�˲�
        ans_=data
        ans=[]
        ans_len=len(ans_)
        for i in range(FilterD-1,len(ans_)):
            featureVector=[]
            for t in range(len(ans_[i])):
                l=[]
                for j in range(FilterD):
                    if i-j>=0 and i-j<ans_len:
                        l.append(ans_[i-j][t])
                l=sorted(l)
                featureVector.append(l[int(len(l)/2)])
            ans.append(featureVector)
        return ans
        
    def AveFilter(self,data):
        #���о�ֵ�˲�
        FilterD=9 #����D֡�˲�
        ans_=data
        ans=[]
        ans_len=len(ans_)
        for i in range(FilterD-1,len(ans_)):
            featureVector=[]
            for t in range(len(ans_[i])):
                l=[]
                for j in range(FilterD):
                    if i-j>=0 and i-j<ans_len:
                        l.append(ans_[i-j][t])
                featureVector.append(sum(l)/len(l))
            ans.append(featureVector)
        return ans          
    def ToDeltaMatrix(self):
        ans=[]
        for t in range(1,len(self.data)):
            d_points=[]
            for i in range(len(self.data[t])):
                d_points.append(self.data[t][i]-self.data[t-1][i])
            ans.append(d_points)
        return ans
    def ToFeatureMatrix(self):
        CutPrefix=False #�Ƿ�õ�ǰ20֡����(��������)
        #���data��������
        ans=[]
        for t in range(1+CutPrefix*18,len(self.data)):
            delta_time = self.data[t][0]-self.data[t-1][0]
            d_points=[]   #�����������(position,Ŀǰ�İ汾��ŷ���Ƿ���10cm����������)
            for i in range(3,9):
                id_=i*3+1
                d_points.append((self.data[t][id_]-self.data[t-1][id_],self.data[t][id_+1]-self.data[t-1][id_+1],self.data[t][id_+2]-self.data[t-1][id_+2]))
            featureVector=[]
            #��������
            for i in range(len(d_points)):
                """len1=(d_points[i][0]**2+d_points[i][2]**2)**0.5
                featureVector.append(len1/delta_time)
                featureVector.append(d_points[i][1]/delta_time)"""
                featureVector.append(d_points[i][0]/delta_time)
                featureVector.append(d_points[i][1]/delta_time)
                featureVector.append(d_points[i][2]/delta_time)
            #��������
            """for i in range(len(d_points)):
                for j in range(i+1,len(d_points)):
                    len1=(d_points[i][0]**2+d_points[i][1]**2+d_points[i][2]**2)**0.5
                    len2=(d_points[j][0]**2+d_points[j][1]**2+d_points[j][2]**2)**0.5
                    dot=d_points[i][0]*d_points[j][0]+d_points[i][1]*d_points[j][1]+d_points[i][2]*d_points[j][2]
                    cross=d_points[i][0]*d_points[j][2]-d_points[i][2]*d_points[j][0]
                    featureVector.append(math.atan(len1/(len2+1e-10)))
                    featureVector.append(dot/(len1*len2+1e-10))
                    featureVector.append(cross/(len1*len2+1e-10))"""
            ans.append(featureVector)
            
        ans = self.MidFilter(ans)
        ans = self.AveFilter(ans)
        return ans
        
        
class ActionDataSet:
    #��ṹ:
    #   action:dict         action["������"]=list[ActionData()]
    
    def __init__(self):  
        self.action=dict()
      
    def __del__(self):
        pass
    
    def __repr__(self):
        return str(dict({"actionNum":len(self.action),"action":self.action}))
        
    def __str__(self):
        return str(dict({"actionNum":len(self.action),"action":self.action}))
            
    def addActionData(self,name,data):
        #whiteList=["inner_kick_left","inner_kick_right","long_kick_left","long_kick_right","toe_kick_left","toe_kick_right"]
        #if(name not in whiteList):
        #    return
        if self.action.get(name)==None:
            self.action[name]=[]
        self.action[name].append(data)
        
    def LoadDataFromFile(self,filename):
        ActionFilter = ['*'] #������'*'��ȫ���������ɣ�����ֻ�����б���Ķ���
        #ActionFilter = ['front_kick_left','front_kick_right','knee_lift_left','knee_lift_right','side_kick_left','side_kick_right'] #������'*'��ȫ���������ɣ�����ֻ�����б���Ķ���
        data=[[j for j in i.split(' ') if len(j)>0] for i in open(filename).read().split('\n') if len(i)>0]
        data.append(["end","0","0"])#��������һ���ָ��б�ʾ���ݽ�β
        lastDataName=""
        actionName=""
        typeList=[]
        dataMat=[]
        for line in data:
            dataName=line[0]+'_'+line[1]
            if lastDataName!=dataName or line[2]=="0":
                if lastDataName!="":
                    #����һ������
                    if('*' in ActionFilter or actionName in ActionFilter):
                        self.addActionData(actionName,ActionData(actionName,typeList,dataMat))
                lastDataName=dataName
                dataMat=[]
            actionName=line[0]
            typeList=[int(line[1])]
            matLine=line[2:]
            for i in range(len(matLine)):
                matLine[i]=float(matLine[i])
            dataMat.append(matLine)
            
if __name__ =='__main__':
    #����
    s = ActionDataSet()
    s.LoadDataFromFile("gyz_0_vec.txt")
    print(s)