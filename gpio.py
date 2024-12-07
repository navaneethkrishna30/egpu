# sudo pip install Jetson.GPIO

import Jetson.GPIO as GPIO
import time

led_pin = 7

GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.HIGH)

while True:
    time.sleep(2)
    GPIO.output(led_pin, GPIO.HIGH)
    print("LED is ON")
    
    time.sleep(2)
    GPIO.output(led_pin, GPIO.LOW)
    print("LED is OFF")