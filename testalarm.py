from pygame import mixer
import time

mixer.init()
mixer.music.load("alarm.mpeg")
mixer.music.play()

print("Playing...")
time.sleep(5)