#coding=gbk

import socket               # ���� socket ģ��
import time

a=[]
for t in range(5):
    s = socket.socket()         # ���� socket ����
    #host = socket.gethostname() # ��ȡ����������
    host = "127.0.0.1"
    port = 29791                # ���ö˿ں�
    
    s.connect((host, port))
    for t in range(5):
        data=input()
        s.send(bytes(data, encoding="utf8"))
    a.append(s)
    #s.close()
time.sleep(10)