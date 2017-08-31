using UnityEngine;
using System.Collections;

// For UI Text
using UnityEngine.UI;

// For Encoding
using System.Text;
using System.Net;
using System.Net.Sockets;
using System;

using UnityEngine;
using System.Collections;

public static class HmmClient
{
    public static string Action = "";//缓存当前的动作，动态更新
    static bool isCalc = false;

    //请求HMM更新Action
    public static void getAction()
    {
        if (isCalc == true) return;
        Net.send(Net.NetDataType.query);
        isCalc = true;
    }

    //
    public static void hmmStart()
    {
        isCalc = false;
        Net.ReceiveData = delegate (Net.NetDataType type, string data) { Action = data; isCalc = false; };
        Net.send(Net.NetDataType.start);
    }

    public static void newFrame(float[] data)
    {
        Net.send(Net.NetDataType.newframe, data);
    }
}

public class Net : MonoBehaviour
{
    /*
    相关通信协议与约定
    传输格式:前四个字节描述包长度n，后四个字节描述包类型，后n个字节为data
    包类型(0-2为主动发出，3为被动接受)：
        0:告知HMM开始一个序列预测,data:无
        1:传入新的一帧数据(m维float),data:float[]
        2:请求HMM返回当前结果,data:无
        3:HMM返回请求,data:string
    */
    public enum NetDataType { start, newframe, query, get };
    public delegate void receiveEventHandle(NetDataType type, string data);
    public static receiveEventHandle ReceiveData;//收到HMM反馈

    public static Net instance_;
    public static string host = "127.0.0.1";
    public static int port = 29791;
    private static Socket clientSocket;

    private static void log(string str)
    {
        Debug.Log(str);
    }

    private void initSocket()
    {
        if (host != "" && port >= 0)
        {
            clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            clientSocket.ReceiveTimeout = 10;
            IPEndPoint ipe = new IPEndPoint(IPAddress.Parse(Net.host), Net.port);
            try
            {
                clientSocket.Connect(ipe);
            }
            catch (Exception e)
            {
                log(e.ToString());
            }
        }
        else Destroy(this);
        //send("Connection OK.");
    }

    // Send a string
    static private void send(byte[] buffer)
    {
        try
        {
            clientSocket.Send(buffer);
        }
        catch (Exception e)
        {
            log(e.ToString());
        }
    }
    static public void send(NetDataType type, string data)
    {
        byte[] buffer = Encoding.UTF8.GetBytes(data);
        byte[] send_buffer = new byte[buffer.Length + 8];
        byte[] ib;
        ib = BitConverter.GetBytes(buffer.Length);
        Buffer.BlockCopy(ib, 0, send_buffer, 0, 4);
        ib = BitConverter.GetBytes((int)type);
        Buffer.BlockCopy(ib, 0, send_buffer, 4, 4);
        Buffer.BlockCopy(buffer, 0, send_buffer, 8, buffer.Length);
        send(send_buffer);
    }
    static public void send(NetDataType type, byte[] buffer)
    {
        byte[] send_buffer = new byte[buffer.Length + 8];
        byte[] ib;
        ib = BitConverter.GetBytes(buffer.Length);
        Buffer.BlockCopy(ib, 0, send_buffer, 0, 4);
        ib = BitConverter.GetBytes((int)type);
        Buffer.BlockCopy(ib, 0, send_buffer, 4, 4);
        Buffer.BlockCopy(buffer, 0, send_buffer, 8, buffer.Length);
        send(send_buffer);
    }
    static public void send(NetDataType type)
    {
        byte[] send_buffer = new byte[8];
        byte[] ib;
        ib = BitConverter.GetBytes(0);
        Buffer.BlockCopy(ib, 0, send_buffer, 0, 4);
        ib = BitConverter.GetBytes((int)type);
        Buffer.BlockCopy(ib, 0, send_buffer, 4, 4);
        send(send_buffer);
    }
    static public void send(NetDataType type, int num)
    {
        byte[] send_buffer = new byte[12];
        byte[] ib;
        ib = BitConverter.GetBytes(4);
        Buffer.BlockCopy(ib, 0, send_buffer, 0, 4);
        ib = BitConverter.GetBytes((int)type);
        Buffer.BlockCopy(ib, 0, send_buffer, 4, 4);
        ib = BitConverter.GetBytes(num);
        Buffer.BlockCopy(ib, 0, send_buffer, 8, 4);
        send(send_buffer);
    }

    static public void send(NetDataType type, float[] data)
    {
        byte[] send_buffer = new byte[data.Length * 4];
        for (int i = 0; i < data.Length; i++)
        {
            float f = data[i];
            byte[] ib;
            ib = BitConverter.GetBytes(f);
            Buffer.BlockCopy(ib, 0, send_buffer, 4 * i, 4);
        }
        send(type, send_buffer);
    }
    // Recieve string from port
    byte[] receive_buffer;
    int restByte = 0;
    //成功解析出一个完整的包
    void get(NetDataType type, byte[] buffer)
    {
        //Debug.Log("get:" + type.ToString() + " buffer_length:" + buffer.Length);
        //Debug.Log(System.Text.Encoding.UTF8.GetString(buffer));
        if (type == NetDataType.get)
            ReceiveData(type, System.Text.Encoding.UTF8.GetString(buffer));
        else
        {
            string log = "";
            for (int i = 0; i < buffer.Length; i++) log += ((int)buffer[i]).ToString() + " ";
            Debug.Log("Error,Net can't parse type:" + type.ToString() + " data:" + log);
        }
    }
    //传输格式:前四个字节描述包长度n，后四个字节描述包类型，后n个字节为data
    bool parse()
    {
        if (receive_buffer == null) return false;
        //Debug.Log("ReceiveSize:" + receive_buffer.Length.ToString() + " resetByte=" + restByte.ToString());
        string log = ""; for (int i = 0; i < receive_buffer.Length; i++) log += ((int)receive_buffer[i]).ToString() + " ";
        //Debug.Log("ReceiveBuffer:" + log);
        if (restByte == 0 && receive_buffer.Length >= 4)
        {
            byte[] r = null;
            restByte = BitConverter.ToInt32(receive_buffer, 0) + 4;
            if (receive_buffer.Length > 4)
            {
                r = new byte[receive_buffer.Length - 4];
                Buffer.BlockCopy(receive_buffer, 4, r, 0, receive_buffer.Length - 4);
            }
            receive_buffer = r;
            //Debug.Log("Receive Data len:" + restByte.ToString());
            return true;
        }
        else if (restByte != 0 && receive_buffer.Length >= restByte)
        {
            byte[] r = null;
            if (restByte > 4)
            {
                byte[] data = new byte[restByte - 4];
                Buffer.BlockCopy(receive_buffer, 4, data, 0, restByte - 4);
                get((NetDataType)BitConverter.ToInt32(receive_buffer, 0), data);
            }
            if (receive_buffer.Length > restByte)
            {
                r = new byte[receive_buffer.Length - restByte];
                Buffer.BlockCopy(receive_buffer, restByte, r, 0, receive_buffer.Length - restByte);
            }
            receive_buffer = r;
            restByte = 0;
            return true;
        }
        return false;
    }
    public void receive()
    {
        if (clientSocket != null && clientSocket.Available > 0)
        {
            //try
            //{
                byte[] buffer = new byte[1024];
                //int length = clientSocket.Receive(buffer, min(buffer.Length, clientSocket.Available), 0);
                int length = clientSocket.Receive(buffer);
                if (length > 0)
                {
                    string log = "receive: len:" + length.ToString() + " ";
                    for (int i = 0; i < length; i++) log += buffer[i].ToString() + ' ';
                    //Debug.Log(log);
                    if (receive_buffer == null)
                    {
                        receive_buffer = new byte[length];
                        Buffer.BlockCopy(buffer, 0, receive_buffer, 0, length);
                    }
                    else
                    {
                        byte[] r = new byte[receive_buffer.Length + length];
                        Buffer.BlockCopy(receive_buffer, 0, r, 0, receive_buffer.Length);
                        Buffer.BlockCopy(buffer, 0, r, receive_buffer.Length, length);
                        receive_buffer = r;
                    }
                    while (parse()) ;
                }
            /*}
            catch (Exception e)
            {
                Debug.Log("Net.cs Error:"+e.ToString());
            }*/
        }
    }

    private int min(int length, int available)
    {
        if (length > available) return available;
        else return length;
    }

    // Start.
    void Start()
    {
        Debug.Log("Net init");
        if (instance_ == null)
            instance_ = this;
        else
        {
            Destroy(this);
            return;
        }
        initSocket();
        //DontDestroyOnLoad(this);
        HmmClient.hmmStart();
    }

    public void init()
    {
        ReceiveData = null;//收到对方指令
        receive_buffer = null;
        restByte = 0;
    }

    int cnt = 0;
    // Update is called once per frame
    void Update()
    {
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.newFrame(new float[18]);
        HmmClient.getAction();
        Debug.Log(HmmClient.Action);
        this.receive();
    }
}