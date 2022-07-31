#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:38:32 2022

@author: remy
"""

import serial
import time

#initialize serial communicatin with arduino board through usb (remember to update port)
arduino = serial.Serial(port = '/dev/cu.usbserial-14110', baudrate = 115200, timeout = 0.1)


while True:
    print("Enter a key to move the robot")

    var = str(input())
    #print("You entered: ", var)
    
    if(var == '0'):
        arduino.write(str.encode('0'))
        print("backwards")

   
     if(var == '1'):
        arduino.write(str.encode('1'))
        print('walking')

        
    if(var == '2'):
        arduino.write(str.encode('2'))
        print("stop")