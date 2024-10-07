"""
Pose6D Comm Client for Pose6D Server: 

Usage : 
    
	Launch Pose6D exe file.
    1. Select an object         : Menu > Project > Select Object Directory>
    2. Select the camera        : Menu > Camera > Select from List> and double click on the camera
    3. Connect to the camera    : Menu > Camera > Connect>
    4. Configure your robot     : Menu > Robot > Select Robot Model > and Double click on the robot type
	5. Configure TCPIP Connection : Menu > Robot> Configure Connection > type 127.0.0.1:8480 ans "Save"
    6. Configure the protocol   : Menu > Robot > Select Comm Protocol> and double click on <,mId, Name,Pose,Q,>
	7. Run Server by command    : Menu > Detect > Run TCP Server with Camera >
    
	In Python enviroment : 
    from ClientPose6D import ClientPose6D 
    p6d = ClientPose6D('127.0.0.1', 8480)
    p6d.connect()
    
    # call to get coordinates of the object from default robot position
    objectId, objectPose, objectQ = p6d.detect()
    
    # call to get coordinates of the object 1 from a different robot position
    robotPose = np.array([100,200,300,0,90,0])
    objectId, objectPose, objectQ = p6d.detect(robotPose = robotPose)  
    
    # call to get coordinates of the object 3 from a different robot position
    objectId, robotPose = 3, np.array([100,-200,345,0,90,0])
    objectId, objectPose, objectQ = p6d.detect(objectId, robotPose)   

    # to finish
    p6d.close()     
	
	
Request message string from Robot to Pose6D Server: 
    '<,MsgId,ObjId,Tx,Ty,Tz,Rx,Ry,Rz,0,>' 
	where: 
		MsgId    - 1, integer, message request id
		ObjId    - 1, integer, object id
		Tx,Ty,Tz - float numbers, robot position in mm
		Rx,Ry,Rz - float numbers. robot orientation in range -180:180
		0        - integer, not in use
		
Response message string from Pose6D Server to Robot: 
    '<,MsgId,ObjId,Tx,Ty,Tz,Rx,Ry,Rz,Q,>' 
	where: 
		MsgId    - 2, integer, message response id
		ObjId    - 1, integer, object id which is detected
		Tx,Ty,Tz - float numbers, object position in mm
		Rx,Ry,Rz - float numbers, object orientation in range -180:180
		Q        - float number, quality of detection in range 0:1. Q must be > 0.8
		


Created on Sun Sep  1 18:12:04 2019

@author: modified by Ben Zion Shaick (RobotAI),   

-----------------------------
 Ver    Date     Who    Descr
-----------------------------
1806   17.05.23  UD     Simple package to manage
1805   30.01.23  BZ     Simple string implementation
-----------------------------

"""

import socket
import time  
import numpy as np

#%% Message Decoder
class MsgPoseData:
    """
    Prepares message between RobotAI Server and Client for pose data request and response.
    """

    def __init__(self, objId = 0, objP = np.array([0,0,0,0,0,0]), objQ = 1):
        self.msgId          = 1
        
        # data fields
        self.objId          = objId
        self.objPose        = objP
        self.objQ           = objQ
        
    def setPose(self, objId = 0, objP = np.array([0,0,0,0,0,0]), objQ = 1):
        
        # data fields
        self.objId          = objId
        self.objPose        = objP
        self.objQ           = objQ        

    def encodeString(self):   
        "encode message into string "  
        
        data = '{0:d},{1:d},{2:f},{3:f},{4:f},{5:f},{6:f},{7:f},{8:f}'.format(self.msgId,self.objId,self.objPose[0],self.objPose[1],self.objPose[2],self.objPose[3],self.objPose[4],self.objPose[5],self.objQ)
        
        return data

    def decodeString(self, data):   
        "decode from string format"  
        
        if type(data) is not str:
            self.Print('ERROR : String Decode :  data is not a string')
            return
            
        msg                 = data.split(',')
        #print(msg)
        try:
            self.msgId      = int(msg[0])
            self.objId      = int(float(msg[1]))
            self.objPose    = np.array([float(x) for x in msg[2:8]])
            self.objQ       = float(msg[8])
        except Exception as e:
            self.msgId      = int(0)
            self.objId      = int(0)
            self.objPose    = np.array([0,0,0,0,0,0])
            self.objQ       = 0
            print(e)
          
        #print(self.objPose)    
        return 
    
    def getPose(self):        
        # data fields
        return self.objId, self.objPose, self.objQ



#%% Messages Encoders/Decoders

def callback_send(socket, data):
      try:
          #serialized = json.dumps(data, cls=NumpyEncoder)
          serialized = data # done in MsgObjectCount
          #print(data)
      except (TypeError, ValueError) as e:
          raise Exception('You can only send string-serializable data %s' %e)
      # send the length of the serialized data first
      head_serialized  = '<,' 
      tail_serialized  = ',>\r' 
      
      socket.sendall(bytes(head_serialized+serialized+tail_serialized, 'utf-8'))


def callback_recv(socket):
      # read the length of the data, letter by letter until we reach EOL
      connectionTerminated = False
      char = ' '
      head_stop_serialized  = '<'       
      tail_serialized  = '>' 
      length_msg_max   = 128
      deserialized     = None
      
      # wait for <
      while char != head_stop_serialized and not connectionTerminated:
          char = socket.recv(1).decode("utf-8")
          if not char : #== b'':
              connectionTerminated = True
              #print(char)
              
      if connectionTerminated:
          print('Client discsonnected')
          return deserialized

      # wait for end
      msg_str    = '' 
      char       = ''
      msg_len    = 0
      while char != tail_serialized and msg_len < length_msg_max and not connectionTerminated:
          msg_str += char
          msg_len += 1
          char        = socket.recv(1).decode("utf-8")
          if not char:
              connectionTerminated = True
          #print(char)
          
      if connectionTerminated:
          print('Connection terminated during receive:')
          print(deserialized)
          return deserialized
      
      # junk
      if msg_len > length_msg_max - 1:
          print('ERROR : garbage received')
          return deserialized
 
      # remove not relevant fields : , at the beginning and , at the end
      deserialized = msg_str[1:-1]
      #print('Dbg 1')
      #print(deserialized)
      return deserialized

#%% Message Decoder
class ClientPose6D:
    """
    Implements data request and response encode decode.
    """

    def __init__(self, ip = '127.0.0.1', port = 8480):
        self.IP          = ip
        self.PORT        = port
        self.sock        = None
        self.msgTx       = MsgPoseData()
        self.msgRx       = MsgPoseData()
        self.is_connected = False
        
        #self.connect()
        print(f"Pose6D Client created - IP : {ip}, Port : {port}")      
        
    def __del__(self):
        self.close()
        print("Pose6D closed")       

    def connect(self):
        # test mutiple connect disconnect
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.IP, self.PORT)
        
        while not self.is_connected:
            print('Attempting to connect...')
            try:
                self.sock.connect(server_address)
                self.is_connected = True
            except:
                print('connect failed')
                self.is_connected = False

            if not self.is_connected:
                time.sleep(5)

        print('Connection established')
            
    def detect(self, objectId = 1, robotPose = np.array([0,0,0,0,0,0])):
        
        if not self.is_connected:
            print('No connection to the server')
            self.connect()
        
        # sending request
        self.msgTx.setPose(objectId,robotPose,1)
        
        # Send the message
        try:
            dataTx = self.msgTx.encodeString()
            callback_send(self.sock, dataTx)
        except:
            print('Send problem')
            self.is_connected = False
            
        # Receive the message back
        try:
            dataRx  = callback_recv(self.sock)
            self.msgRx.decodeString(dataRx)
        except:
            print('Recv problem')   
            self.is_connected = False        
            
        objId, objPose, objQ = self.msgRx.getPose()
        print('Tx,Ty,Tz : %s' %str(objPose[0:3]))
        print('Rx,Ry,Rz : %s' %str(objPose[3:6]))        
        print('Quality  : %s' %str(objQ))
        
        return objId, objPose, objQ
           
            
    def close(self):
        print('Attempt to disconnect')
        try:
            self.sock.close()
            time.sleep(5)                
        except:
            print('disconnect failed')
        
       
#%% Tests           
class TestServerClientThreaded(): #unittest.TestCase
                     

    def TestClientConnect():
     
        # test mutiple connect disconnect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Connect the socket to the port where the server is listening
        server_address = ('127.0.0.1', 8480)
                
        # Create the data and load it into json
        for k in range(10):
            
             print('Attempt to connect')
             try:
                sock.connect(server_address)
                time.sleep(5)
             except:
                print('connect failed')
                
             print('Attempt to disconnect')
             try:
                sock.close()
                time.sleep(5)                
             except:
                print('disconnect failed')
           
        sock.close()
        print('TestClientConnect done')
         
          
    def TestClientSimple():
     
        #import time
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Connect the socket to the port where the server is listening
        server_address = ('192.168.0.5', 12345)
        try:
            sock.connect(server_address)
        except:
            print('Run server first')
            return
        
        # msg data encoder-decoder
        msgTx = '1,1,0,0,0.0,0,0,0' # msgId, objId, Tx,Ty,Tz,Rx,Ry,Rz,Q
        msgRx = ''
		
        msgObj = MsgPoseData()
        
        # Create the data and load it into json
        for k in range(100):
            
            data = msgTx
            
            # Send the message
            try:
                callback_send(sock, data)
            except:
                print('Send problem')
                break
            
            # Receive the message back
            try:
                msgRx  = callback_recv(sock)
                msgObj.decodeString(msgRx)
                print('Tx,Ty,Tz : %s' %str(msgObj.objPose[0:3]))
                print('Quality  : %s' %str(msgObj.objQ))
            except:
                print('Recv problem')
                break
            
            # decode

            print('Response from SERVER : %s' %msgRx)
            time.sleep(1)
            
        sock.close()
        print('TestClientSimple done')

    def TestClientPose6D():
     
        p6d = ClientPose6D('127.0.0.1', 5555)
        p6d.connect()
        
        # Create the data and load it into json
        for k in range(100):
            
            objectId, objectPose, objectQ = p6d.detect()
            time.sleep(1)
            
        p6d.close()
        print('TestClientPose6D done')

        
#%%
if __name__ == "__main__":
    """
    You can run TestClient in different kernels
    or you can run TestClientThreaded in the same Kernel/Console
    """
    
    from ClientPose6D import TestServerClientThreaded as tsc

    #tsc.TestClientConnect()
    tsc.TestClientPose6D()

