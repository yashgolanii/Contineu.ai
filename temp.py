#!/usr/bin/env python3
'''Measures sensor scanning speed'''
from rplidar import RPLidar
import time

PORT_NAME = '/dev/ttyUSB0'

def run():
    '''Main function'''
    lidar = RPLidar(PORT_NAME)
    health = lidar.get_health()
    print(health)
  

if __name__ == '__main__':
    run()