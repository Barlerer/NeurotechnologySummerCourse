# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 15:12:27 2022

@author: Michaela
"""

import urllib3
http = urllib3.PoolManager()

while True:
    print("Enter a key to move the robot")

    var = str(input())
    #print("You entered: ", var)
    
    if(var=='0'):
        print('Move backward')
        r = http.request('GET', 'https://slothy1.herokuapp.com/postCommand?command=0')
    elif(var=='1'):
        print('Move forward')
        r = http.request('GET', 'https://slothy1.herokuapp.com/postCommand?command=1')