#coding=gbk
import matplotlib as mpl
import matplotlib.pyplot as plt
import PersonData_class as pd
import Classifier as cl
import Segmentation as sg
import itertools
import pickle
import os
import math
import numpy as np
import sklearn.decomposition
from pylab import *
from main import *

def WrongData_Visualization(filename,save_path_id=""):
    usePickle = True #是否使用pickle缓存
    dumpfilename=filename+".dump"
    s = pd.ActionDataSet()
    if usePickle==False or os.path.exists(dumpfilename)==False:
        s.LoadDataFromFile(filename)
        pickle.dump(s,open(dumpfilename,'wb'))
    else:
        s=pickle.load(open(dumpfilename,'rb'))
    plt.figure(figsize=(19,10))
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95, wspace=0.25, hspace=0.25)
    featureNum=0
    actionMap=dict()#actionName到id的映射
    actionNameList=[]
    for actionName in s.action:
        actionNameList.append(actionName)
    actionNameList=sorted(actionNameList)
    for i in range(len(actionNameList)):
        actionMap[actionNameList[i]]=i+1
    for actionName in s.action:
        actionData=s.action[actionName]
        data=actionData[0].featureMat
        featureNum=len(data[0])
        break
    HMM = cl.Classifier()
    HMM.addDataFromADS(s)
    HMM.fit()
    for featureID in range(featureNum):
        print("feature:"+str(featureID))
        plt.clf()
        for actionName in s.action:
            actionData=s.action[actionName]
            print(actionName)
            plt.subplot(4,4,actionMap[actionName])
            plt.title(actionName,fontsize=10)
            colormap = itertools.cycle(["red","blue","green","gold","hotpink","aqua","brown","mediumorchid"])
            for action in actionData:
                data=action.featureMat
                x_data=[i[featureID] for i in data]
                color_="blue"
                if actionName!=HMM.predict(action.featureMat):
                    color_="red"
                plt.plot(x_data,color=color_,linewidth=0.8)
        save_path='features'+save_path_id
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        plt.savefig(save_path+"/feature_wrong_"+str(featureID)+".png")
        #plt.show()
        
def Visualization(filename,save_path_id=""):
    usePickle = True #是否使用pickle缓存
    dumpfilename=filename+".dump"
    s = pd.ActionDataSet()
    if usePickle==False or os.path.exists(dumpfilename)==False:
        s.LoadDataFromFile(filename)
        pickle.dump(s,open(dumpfilename,'wb'))
    else:
        s=pickle.load(open(dumpfilename,'rb'))
    plt.figure(figsize=(19,10))
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95, wspace=0.25, hspace=0.25)
    featureNum=0
    actionMap=dict()#actionName到id的映射
    actionNameList=[]
    for actionName in s.action:
        actionNameList.append(actionName)
    actionNameList=sorted(actionNameList)
    for i in range(len(actionNameList)):
        actionMap[actionNameList[i]]=i+1
    for actionName in s.action:
        actionData=s.action[actionName]
        data=actionData[0].featureMat
        featureNum=len(data[0])
        break
    for featureID in range(featureNum):
        print("feature:"+str(featureID))
        plt.clf()
        actionSize=(int)(math.sqrt(len(s.action))+0.9)
        for actionName in s.action:
            actionData=s.action[actionName]
            print(actionName)
            plt.subplot(actionSize,actionSize,actionMap[actionName])
            plt.title(actionName,fontsize=10)
            colormap = itertools.cycle(["red","blue","green","gold","hotpink","aqua","brown","mediumorchid"])
            for action in actionData:
                data=action.featureMat
                x_data=[i[featureID] for i in data]
                plt.plot(x_data,color=next(colormap),linewidth=0.8)
        save_path='features'+save_path_id
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        plt.savefig(save_path+"/feature_all_"+str(featureID)+".png")
        #plt.show()
        
def VisualizationHMMPredict(filename,save_path_id=""):
    usePickle = True #是否使用pickle缓存
    dumpfilename=filename+".dump"
    s = pd.ActionDataSet()
    if usePickle==False or os.path.exists(dumpfilename)==False:
        s.LoadDataFromFile(filename)
        pickle.dump(s,open(dumpfilename,'wb'))
    else:
        s=pickle.load(open(dumpfilename,'rb'))
    HMM = cl.Classifier()
    HMM.addDataFromADS(s)
    HMM.fit()
    plt.figure(figsize=(19,10))
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95, wspace=0.25, hspace=0.25)
    featureNum=0
    actionMap=dict()#actionName到id的映射
    actionNameList=[]
    for actionName in s.action:
        actionNameList.append(actionName)
    actionNameList=sorted(actionNameList)
    for i in range(len(actionNameList)):
        actionMap[actionNameList[i]]=i+1
    for actionName in s.action:
        actionData=s.action[actionName]
        data=actionData[0].featureMat
        featureNum=len(data[0])
        break
    for actionName in s.action:
        plt.clf()
        actionSize=(int)(math.sqrt(len(s.action))+0.9)
        actionData=s.action[actionName]
        print(actionName)
        for a in s.action:
            plt.subplot(actionSize,actionSize,actionMap[a])
            plt.title(a,fontsize=10)
        colormap = itertools.cycle(["red","blue","green","gold","hotpink","aqua","brown","mediumorchid"])
        for action in actionData:
            color_=next(colormap)
            data=action.featureMat
            score_data=dict()
            for t in range(1,len(data)):
                block=HMM.scores(data[:t])
                for d in block:
                    if(score_data.get(d["name"])==None):
                        score_data[d["name"]]=[]
                    score_data[d["name"]].append(d["score"])
            for name,score in score_data.items():
                plt.subplot(actionSize,actionSize,actionMap[name])
                #print(name,score)
                plt.plot(score,color=color_,linewidth=0.8)
        save_path='features'+save_path_id
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        plt.savefig(save_path+"/HMMScore_"+str(actionName)+".png")
    #plt.show()

#截取预测区间绘制预测准确率热力图
def VisualizationSplitHotMap(FileName,save_path_id=""):
    s = pd.ActionDataSet()
    s=LoadData(FileName)
    HMM = cl.Classifier()
    HMM.addDataFromADS(s)
    HMM.fit()
    print(HMM)
    total=0
    splitNum=10
    correct=np.array([[0 for j in range(splitNum)] for i in range(splitNum)])
    out=open("report_split_"+FileName,"w");
    for className,dataSet in s.action.items():
        for dataBlock in dataSet:
            MatWeight=len(dataBlock.featureMat)
            total+=1
            for start in range(0,splitNum):
                for end in range(start+1,splitNum+1):
                    Mat=dataBlock.featureMat[int(MatWeight*start/splitNum):int(MatWeight*end/splitNum-0.01)+1]
                    if className==HMM.predict(Mat):
                        correct[start][end-1]+=1
                    else:
                        print(start,end,className,HMM.predict(Mat))
                        out.write(str(start)+" "+str(end)+" "+className+" "+HMM.predict(Mat)+"\n")
    for t1 in range(0,splitNum):
        for t2 in range(t1,splitNum):
            out.write("\n{0} {1} {2} {3}-{4}".format(correct[t1][t2],total,correct[t1][t2]/total,str(t1/splitNum*100)+"%",str((t2+1)/splitNum*100)+"%"))
    out.close()
    plt.figure(figsize=(19,10))
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95, wspace=0.25, hspace=0.25)
    plt.clf()
    fig = plt.figure()
    correct[splitNum-1,0]=total
    ax = fig.add_subplot(111)  
    ax.imshow(correct,cmap=cm.hot)
    save_path='features'+save_path_id
    if os.path.exists(save_path)==False:
        os.mkdir(save_path)
    plt.savefig(save_path+"/SplitHotMap.png")
    

def ShowAVESpeed(filename,save_path_id=""):
    SG=sg.Segmentation()
    usePickle = True #是否使用pickle缓存
    dumpfilename=filename+".dump"
    s = pd.ActionDataSet()
    if usePickle==False or os.path.exists(dumpfilename)==False:
        s.LoadDataFromFile(filename)
        pickle.dump(s,open(dumpfilename,'wb'))
    else:
        s=pickle.load(open(dumpfilename,'rb'))
    plt.figure(figsize=(19,10))
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95, wspace=0.25, hspace=0.25)
    featureNum=0
    actionMap=dict()#actionName到id的映射
    actionNameList=[]
    for actionName in s.action:
        actionNameList.append(actionName)
    actionNameList=sorted(actionNameList)
    for i in range(len(actionNameList)):
        actionMap[actionNameList[i]]=i+1
    for actionName in s.action:
        actionData=s.action[actionName]
        data=actionData[0].featureMat
        featureNum=len(data[0])
        break
    plt.clf()
    actionSize=(int)(math.sqrt(len(s.action))+0.9)
    for actionName in s.action:
        actionData=s.action[actionName]
        print(actionName)
        plt.subplot(actionSize,actionSize,actionMap[actionName])
        plt.title(actionName,fontsize=10)
        colormap = itertools.cycle(["red","blue","green","gold","hotpink","aqua","brown","mediumorchid"])
        for action in actionData:
            data=action.featureMat
            x_data=SG.FeatureMatToSpeedVector(data)
            plt.plot(x_data,color=next(colormap),linewidth=0.8)
    save_path='features'+save_path_id
    if os.path.exists(save_path)==False:
        os.mkdir(save_path)
    plt.savefig(save_path+"/speed_"+str(filename)+".png")
    #plt.show()

def RelevanceAnalysis(filename,save_path_id=""):
    X_name=["leftfoot","rightfoot"]
    Y_name=["head","lefthand","righthand"]
    X_range=[(28,32),(37,41)]
    Y_range=[(1,4),(10,13),(19,22)]
    usePickle = True #是否使用pickle缓存
    dumpfilename=filename+".dump"
    s = pd.ActionDataSet()
    if usePickle==False or os.path.exists(dumpfilename)==False:
        s.LoadDataFromFile(filename)
        pickle.dump(s,open(dumpfilename,'wb'))
    else:
        s=pickle.load(open(dumpfilename,'rb'))
    plt.figure(figsize=(19,10))
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95, wspace=0.25, hspace=0.25)
    UseSpeed = True
    for actionName in s.action:
        actionData=s.action[actionName]
        print(actionName)
        plt.clf()
        colormap = itertools.cycle(["red","blue","green","gold","hotpink","aqua","brown","mediumorchid"])
        for action in actionData:
            color_=next(colormap)
            data=[]
            if UseSpeed:
                data=action.ToDeltaMatrix()
            else:
                data=action.data
            for xid in range(len(X_range)):
                for yid in range(len(Y_range)):
                    featureName=actionName+'_'+X_name[xid]+'_'+Y_name[yid]
                    totalId=xid*len(Y_range)+yid+1
                    print(featureName)
                    ax=plt.subplot(2,3,totalId)
                    #ax.set_xlim((-0.8,1.0))
                    #ax.set_ylim((-0.2,0.2))
                    plt.title(featureName,fontsize=10)
                    x_data=[]
                    y_data=[]
                    if UseSpeed:
                        x_data=[np.array(i[X_range[xid][0]:X_range[xid][1]])/i[0] for i in data]
                        y_data=[np.array(i[Y_range[yid][0]:Y_range[yid][1]])/i[0] for i in data]
                    else:
                        x_data=[i[X_range[xid][0]:X_range[xid][1]] for i in data]
                        y_data=[i[Y_range[yid][0]:Y_range[yid][1]] for i in data]
                    
                    pca=sklearn.decomposition.PCA(n_components=1, copy=True, whiten=False)
                    if UseSpeed:
                        x_data=np.array([math.sqrt(i[0]**2+i[1]**2+i[2]**2) for i in x_data])
                        y_data=np.array([math.sqrt(i[0]**2+i[1]**2+i[2]**2) for i in y_data])
                        x_data=action.AveFilter(action.MidFilter([[i] for i in x_data/sum(x_data)]))
                        y_data=action.AveFilter(action.MidFilter([[i] for i in y_data/sum(y_data)]))
                    else:
                        x_data=pca.fit_transform(np.array(x_data))
                        y_data=pca.fit_transform(np.array(y_data))
                        x_data=action.AveFilter(action.MidFilter([[i] for i in x_data]))
                        y_data=action.AveFilter(action.MidFilter([[i] for i in y_data]))
                    x_data=[i[0] for i in x_data]
                    y_data=[i[0] for i in y_data]
                    plt.plot(x_data,y_data,color=color_,linewidth=0.8)
        save_path='relevance'+save_path_id
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        plt.savefig(save_path+"/"+actionName+".png")
        #plt.show()