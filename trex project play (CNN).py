from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

#REKOR: 641

mon={"top":530,"left":700,"width":250,"height":100}
sct=mss()

width=125
height=50

model=model_from_json(open("model.json","r").read())
model.load_weights("trexproject_weights.h5")

#down=0, right=1, up=2
labels=["Down","Right","Up"]

framerate_time=time.time()
counter=0
i=0
delay=0.4
key_down_pressed=False
while True:
    img=sct.grab(mon)
    im=Image.frombytes("RGB",img.size,img.rgb)
    im2=np.array(im.convert("L").resize((width,height)))
    im2=im2/255
    x=np.array([im2])
    x=x.reshape(x.shape[0],width,height,1)
    r=model.predict(x)
    result=np.argmax(r) #tahmın edilmiş labelın indexini alır
    
    if result==0: #down
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed=True
        
    elif result==2: #up
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        time.sleep(delay)
    
        keyboard.press(keyboard.KEY_UP)
        
        if(i<1500):
            time.sleep(0.3)
        elif(1500<i and i<5000):
            time.sleep(0.2)
        else:
            time.sleep(0.17)

        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
        
    counter+=1

    if(time.time() - framerate_time) > 1:
        counter=0
        framerate_time=time.time()
        if(i<=1500):
            delay-=0.003
        else:
            delay-=0.005
        if(delay<0):
            delay=0
        print("***********************")
        print("down:{}\nright:{}\nup:{}".format(r[0][0],r[0][1],r[0][2]))
        i=i+1




















