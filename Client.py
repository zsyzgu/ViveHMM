#coding=gbk

import socket               # 导入 socket 模块
import time

a=[]
for t in range(5):
    s = socket.socket()         # 创建 socket 对象
    #host = socket.gethostname() # 获取本地主机名
    host = "127.0.0.1"
    port = 29791                # 设置端口好
    
    s.connect((host, port))
    for t in range(5):
        data=input()
        s.send(bytes(data, encoding="utf8"))
    a.append(s)
    #s.close()
time.sleep(10)