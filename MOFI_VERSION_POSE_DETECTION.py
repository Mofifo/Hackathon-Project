import cv2
import mediapipe as mp
import numpy as np
import math
import time 
import serial
import serial.tools.list_ports

#for info in serial.tools.list_ports.comports():
 #   print(info)

mp_pose = mp.solutions.pose  
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

arduino_serial = serial.Serial()
arduino_serial.baudrate = 115200
arduino_serial.port = "/dev/cu.usbmodem141201"
#arduino_serial.open()
time.sleep(1)

duration = 1 # It chooses the most dominant pose in a 1 second duration

LEFT_HANDS_UP = '1'
LEFT_HANDS_DOWN = '2'
LEFT_HANDS_SIDEWAYS = '3'
RIGHT_HANDS_UP = '4'
RIGHT_HANDS_DOWN = '5'
RIGHT_HANDS_SIDEWAYS = '6'

def send_to_arduino(text):
    arduino_serial.write(bytes(text + '\n', 'utf-*'))
   # print("Sent to arduino: ", text)

def calculate_angle(a, b, c, landmarks):
    a_x, a_y = landmarks[a]
    b_x, b_y = landmarks[b]
    c_x, c_y = landmarks[c]
    
    # Calculate vectors
    v1 = np.array([a_x - b_x, a_y - b_y])
    v2 = np.array([c_x - b_x, c_y - b_y])
    
    # Calculate angle in degrees
    angle = math.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    #print(angle)#This will print definitely print the angle calculated
    return angle

   

def is_arm_up(armpit_angle, elbow_angle):
    #print(is_arm_up) # this should print the angle 
    return elbow_angle < 60

def is_arm_down(armpit_angle, elbow_angle):
   # print(is_arm_down)
    return (elbow_angle > 160 and elbow_angle < 181) and armpit_angle < 20

def is_arm_sideways(armpit_angle, elbow_angle):
    #print(is_arm_sideways)
    return (elbow_angle > 80 and elbow_angle < 181) and (armpit_angle > 39 and armpit_angle < 61)


cap = cv2.VideoCapture(0)

iterations = 1
timeout = time.time() + duration
current_left_pose = {
    "HANDS_DOWN": 0, 
    "HANDS_UP": 0, 
    "HANDS_SIDEWAYS": 0, 
    "CANT_RECOGNIZE": 0
    }
current_right_pose = {
    "HANDS_DOWN": 0, 
    "HANDS_UP": 0, 
    "HANDS_SIDEWAYS": 0, 
    "CANT_RECOGNIZE": 0
    }

while True:
    ret, img = cap.read()

    img = cv2.resize(img, (720, 600))
    
    results = pose.process(img)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Detected Pose', img)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        landmark_coords = [(lm.x,lm.y) for lm in landmarks]

        left_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        left_elbow = mp_pose.PoseLandmark.LEFT_ELBOW.value
        left_wrist = mp_pose.PoseLandmark.LEFT_WRIST.value
        left_hip = mp_pose.PoseLandmark.LEFT_HIP.value

        right_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        right_elbow = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        right_wrist = mp_pose.PoseLandmark.RIGHT_WRIST.value
        right_hip = mp_pose.PoseLandmark.RIGHT_HIP.value

        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist, landmark_coords)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist, landmark_coords)
        left_armpit_angle = calculate_angle(left_hip, left_shoulder, left_elbow, landmark_coords)
        right_armpit_angle = calculate_angle(right_hip, right_shoulder, right_elbow, landmark_coords)
         

        print(left_elbow_angle)
        print(right_elbow_angle)
        print(left_armpit_angle)
        print(right_armpit_angle)
        # For left arm
        #if is_arm_up(left_armpit_angle, left_elbow_angle):
         #   current_left_pose["HANDS_UP"] += 1
            
        #elif is_arm_down(left_armpit_angle, left_elbow_angle):
         #   current_left_pose["HANDS_DOWN"] += 1

        #elif is_arm_sideways(left_armpit_angle, left_elbow_angle):
         #   current_left_pose["HANDS_SIDEWAYS"] += 1

        #else:
         #   current_left_pose["CANT_RECOGNIZE"] += 1

        #For right arm
        #if is_arm_up(right_armpit_angle, right_elbow_angle):
         #   current_right_pose["HANDS_UP"] += 1
            
        #elif is_arm_down(right_armpit_angle, right_elbow_angle):
         #   current_right_pose["HANDS_DOWN"] += 1

        #elif is_arm_sideways(right_armpit_angle, right_elbow_angle):
         #   current_right_pose["HANDS_SIDEWAY=================================================================S"] += 1


        #else:
         #   current_right_pose["CANT_RECOGNIZE"] += 1

        #if(time.time() > `timeout):

            # Left pose
          #  dominant_left_poses_val = max(current_left_pose.values())
         #   dominant_left_poses = list(filter(lambda x: current_left_pose[x] == dominant_left_poses_val, current_left_pose))
            # print("DOMINANT LEFT POSE: " + dominant_left_poses[0])
            
            #match dominant_left_poses[0]: 
             #   case "HANDS_UP":
              #      send_to_arduino(LEFT_HANDS_UP)
               #     print("LEFT_HANDS_UP")
                #case "HANDS_DOWN":
                 #   send_to_arduino(LEFT_HANDS_DOWN)
                  #  print("LEFT_HANDS_DOWN")
                #case "HANDS_SIDEWAYS":
                 #   send_to_arduino(LEFT_HANDS_SIDEWAYS)
                  #  print("LEFT_HANDS_SIDEWAYS")
                #case _:
                 #   print("LEFT_HANDS_NOT_RECOGNIZED")

            # Right pose
            #dominant_right_poses_val = max(current_right_pose.values())
            #dominant_right_poses = list(filter(lambda x: current_right_pose[x] == dominant_right_poses_val, current_right_pose))
            # print("DOMINANT RIGHTPOSE: " + dominant_right_poses[0])
            
            #match dominant_right_poses[0]: 
             #   case "HANDS_UP":
              #      send_to_arduino(RIGHT_HANDS_UP)
               #     print("RIGHT_HANDS_UP")
                #case "HANDS_DOWN":
                  #  send_to_arduino(RIGHT_HANDS_DOWN)
                 #   print("RIGHT_HANDS_DOWN")
                #case "HANDS_SIDEWAYS":
                 #   send_to_arduino(RIGHT_HANDS_SIDEWAYS)
                  #  print("RIGHT_HANDS_SIDEWAYS")
                #case _:
                 #   print("RIGHT_HANDS_NOT_RECOGNIZED")

            #RESET
            #timeout = time.time() + duration
            #current_left_pose = {
            #"HANDS_DOWN": 0, 
            #"HANDS_UP": 0, 
            #"HANDS_SIDEWAYS": 0, 
            #"CANT_RECOGNIZE" : 0
            #}

            #current_right_pose = {
            #"HANDS_DOWN": 0, 
            #"HANDS_UP": 0, 
            #"HANDS_SIDEWAYS": 0, 
           # "CANT_RECOGNIZE" : 0
            #}

    cv2.waitKey()