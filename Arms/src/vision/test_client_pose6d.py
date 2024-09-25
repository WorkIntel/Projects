"""
Comm Socket Test - Client for <,> Pose6D Server: 

Usage : 
    
	Launch Pose6D exe file.
	Configure TCPIP Connection : 127.0.0.1:8480
	Run Server by command : Menu > Detect > Run Camera with Server.
    
	In Python enviroment : 
    from Test_RobotClient import TestServerClientThreaded as tsc
    tsc.TestClientSimple()
	
	
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
1805   30.01.23 BZ     Simple string implementation
-----------------------------

"""

import socket
import time
    

#%% Messages

def callback_send(socket, data):
      try:
          #serialized = json.dumps(data, cls=NumpyEncoder)
          serialized = data # done in MsgObjectCount
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
        server_address = ('127.0.0.1', 8480)
        try:
            sock.connect(server_address)
        except:
            print('Run server first')
            return
        
        # msg data encoder-decoder
        msgTx = '1,1,100.1,-100.5,250,45.0,-90,180,0' # msgId, objId, Tx,Ty,Tz,Rx,Ry,Rz,Q
        msgRx = ''
        
        # Create the data and load it into json
        for k in range(10):
            
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
            except:
                print('Recv problem')
                break
            
            # decode

            print('Response from SERVER : %s' %msgRx)
            time.sleep(1)
            
        sock.close()
        print('TestClientSimple done')

        
        
#%%
if __name__ == "__main__":
    """
    You can run TestClient in different kernels
    or you can run TestClientThreaded in the same Kernel/Console
    """
    
    from Test_RobotClient import TestServerClientThreaded as tsc

    #tsc.TestClientConnect()
    tsc.TestClientSimple()

