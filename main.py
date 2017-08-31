#coding=gbk
import numpy as np
from hmmlearn import hmm

import PersonData_class as pd
import Classifier as cl
from Visualize import *

def LoadData(filename,usePickle=True):
    #构造ActionDataSet
    dumpfilename=filename+".dump"
    s = pd.ActionDataSet()
    if usePickle==False or os.path.exists(dumpfilename)==False:
        s.LoadDataFromFile(filename)
        pickle.dump(s,open(dumpfilename,'wb'))
    else:
        s=pickle.load(open(dumpfilename,'rb'))
    return s

def MainTest():
    s = pd.ActionDataSet()
    s=LoadData("3_vec.txt",True)
    #s.LoadDataFromFile("gyz_0_vec.txt")
    #s.LoadDataFromFile("qy_0_vec.txt")
    #s.LoadDataFromFile("sk_vec.txt")
    #s.LoadDataFromFile("lyq_vec.txt")
    #s.LoadDataFromFile("2_vec.txt")
    #s.LoadDataFromFile("3_vec.txt")
    HMM = cl.Classifier()
    HMM.addDataFromADS(s)
    HMM.fit()
    print(HMM)
    correct=0
    total=0
    for className,dataSet in s.action.items():
        for dataBlock in dataSet:
            if dataBlock.typeList[0]%3==0:
                total+=1
                if className==HMM.predict(dataBlock.featureMat):
                    correct+=1
                else:
                    print(className,HMM.predict(dataBlock.featureMat))
    print(correct,total,correct/total)
    
#测试动态序列不同比例前缀的准确率
def SplitTest(FileName):
    s = pd.ActionDataSet()
    s=LoadData(FileName)
    HMM = cl.Classifier()
    HMM.addDataFromADS(s)
    HMM.fit()
    print(HMM)
    total=0
    splitNum=10
    correct=[0 for i in range(splitNum+1)]
    out=open("report_"+FileName,"w");
    for className,dataSet in s.action.items():
        for dataBlock in dataSet:
            MatWeight=len(dataBlock.featureMat)
            total+=1
            for t in range(1,splitNum+1):
                Mat=dataBlock.featureMat[0:int(MatWeight*t/splitNum-0.01)+1]
                if className==HMM.predict(Mat):
                    correct[t]+=1
                else:
                    print(str(t/splitNum*100)+"%",className,HMM.predict(Mat))
                    out.write(str(t/splitNum*100)+"% "+className+" "+HMM.predict(Mat)+"\n")
    for t in range(1,splitNum+1):
        print(correct[t],total,correct[t]/total,str(t/splitNum*100)+"%")
    for t in range(1,splitNum+1):
        out.write("\n{0} {1} {2} {3}".format(correct[t],total,correct[t]/total,str(t/splitNum*100)+"%"))
    out.close()
    
def SyntheticTest(FileName):
    save_path='_'+FileName.split('.')[0]
    SplitTest(FileName)
    Visualization(FileName,save_path)
    WrongData_Visualization(FileName,save_path)
    ShowAVESpeed(FileName,save_path)
    RelevanceAnalysis(FileName,save_path)
    VisualizationSplitHotMap(FileName,save_path)
    

if __name__ =='__main__':
    #analysis_files = ['cyz.txt','gty.txt','gyz.txt','hwj.txt','lpn.txt','plh.txt','swn.txt','yyk.txt','yzp.txt','zmy.txt']
    analysis_files = ['mq.txt','pxy.txt','qy.txt','wrl.txt','yx.txt']
    for FileName in analysis_files:
        try:
            SyntheticTest(FileName)
        finally:
            pass
    #SplitTest("gyz_all.txt");
    #MainTest()
    #Visualization("gyz_all.txt")
    #VisualizationHMMPredict("gyz_all.txt")
    #ShowAVESpeed("gyz_all.txt")
    #WrongData_Visualization("3_vec.txt")
    #RelevanceAnalysis("gyz731_vec.txt")