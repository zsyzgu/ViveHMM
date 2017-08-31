#coding=gbk

import socket               # 导入 socket 模块
import struct
import numpy as np
from hmmlearn import hmm

import PersonData_class as pd
import Classifier as cl

HMM=None #全局HMM模型
dataList=None #全局feature矩阵
cache=b''

def send(data_type=0,data=""):
    #向终端发送data_type类型的数据，data为字符串
    global Socket
    #data = bytes(data, encoding="utf8")
    data = bytes(data.encode('utf-8'))
    data = struct.pack('i',len(data))+struct.pack('i',data_type)+data
    Socket.send(data)

def parse():
    global HMM
    global dataList
    global cache
    cache_len=len(cache)
    if(cache_len<4):
        return False
    size=struct.unpack('i',cache[0:4])[0]
    if(cache_len<8+size):
        return False
    data_type=struct.unpack('i',cache[4:8])[0]
    data=cache[8:(8+size)]
    cache=cache[(8+size):]
    if(data_type==0):
        dataList=[]
    elif(data_type==1):
        result=struct.unpack(''.join(['f' for i in range(int(len(data)/4))]),data)
        #print("receive data len:"+str(len(list(result))))
        dataList.append(result)
    elif(data_type==2):
        #print(dataList)
        send(3,HMM.predict(dataList))
    else:
        print("parse Error! ",size,data_type,data)
    return True

def recv(data):
    global cache
    cache+=data
    while(parse()):
        pass
    
def start():
    #读取训练数据并训练HMM
    s = pd.ActionDataSet()
    #添加数据到训练集只需反复s.LoadDataFromFile('文件名')即可
    s.LoadDataFromFile('train.txt')
    global HMM
    HMM = cl.Classifier()
    HMM.addDataFromADS(s)
    HMM.fit()

if __name__ =='__main__':
    
    print("Begin Build HMM.")
    start()
    
    while (True):
        s = socket.socket()         # 创建 socket 对象
        host = ""                   # 获取本地主机名
        port = 29791                # 设置端口
        s.bind((host, port))        # 绑定端口
        s.listen(1)                 # 等待客户端连接

        print("End Build HMM. Listen(29791) now.")
        Socket, addr = s.accept()     # 建立客户端连接。
        print('Accept:', addr)
        while(True):
            data = Socket.recv(1024)
            if not data:
                break
            recv(data)
