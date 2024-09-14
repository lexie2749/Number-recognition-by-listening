import RPi.GPIO as GPIO
import time

# Pin definitions
DIR_PIN = 17  # Connect DIR pin of M542H to Raspberry Pi GPIO 17
PUL_PIN = 18  # Connect PUL pin of M542H to Raspberry Pi GPIO 18

# Number of steps to move in each direction
STEPS = 200

GPIO.setmode(GPIO.BCM)  # Use BCM GPIO numbering
GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(PUL_PIN, GPIO.OUT)

def pulse():
    GPIO.output(PUL_PIN, GPIO.HIGH)
    time.sleep(0.0005)  # Adjust pulse width for desired speed (in seconds)
    GPIO.output(PUL_PIN, GPIO.LOW)
    time.sleep(0.0005) 

try:
    while True:
        # Move forward
        GPIO.output(DIR_PIN, GPIO.HIGH)
        for _ in range(STEPS):
            pulse()

        time.sleep(1)  # Wait 1 second

        # Move backward
        GPIO.output(DIR_PIN, GPIO.LOW)
        for _ in range(STEPS):
            pulse()

        time.sleep(1)  # Wait 1 second

except KeyboardInterrupt:
    # Cleanup GPIO on Ctrl+C
    GPIO.cleanup()