#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:38:32 2022

@author: remy
"""

import serial
import time
import httpx

#initialize serial communicatin with arduino board through usb (remember to update port)
arduino = serial.Serial(port = '/dev/cu.usbserial-14110', baudrate = 115200, timeout = 0.1)


while True:
    time.sleep(5)
    r = httpx.get('https://slothy1.herokuapp.com/')
    command=r.json()
	
    if not command:
        print("Doing nothing")
    elif command[-1] == '1':
        arduino.write(str.encode('1'))
        print('Move forward')
    elif command[-1] == '0':
        arduino.write(str.encode('0'))
        print('Move backward')