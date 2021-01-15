# IoT Smart-Speaker
## Overview
This is the easiest way to control your speaker. 
Have you ever encounter the situation when you don't have spare hands turning off the music or feel annoying to control volume by keyboard? In my project, you can easily turn on and off the music simply by looking at the Picamera and blinking eyes. Our Python code will fetch and play studying music on youtube automatically. It means that you don't have to download the music onto your Pi. You can stop at anytime by blinking at the Picamera again. 
Also, I Utilized gesture sensor to detect hands movement. If you move your hand upward, the volume will goes up, vice versa.
This project can extend to anything you want to control with your eyes. For example, turn off the light, turn on the computer... and so on.

## Prerequisites

### Hardware Device

* Raspberry pi 4 *1
* Picamera *1
* Bluetooth Speaker *1

![](https://i.imgur.com/dk6Wgub.png)


### Software

* [raspbian os](https://www.raspberrypi.org/software/)

#### Python package
* [OpenCV for Raspberry pi 4](https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/)
	* As a small Reminder, be sure to follow each step on the tutorial website or something terrible will definitely happen beyond your imagination. I mean it! :fearful:
	* If you got berryconda installed on your pi previously, you can easily create virtual enviorment with it.
	* Be sure to install all of your pip package in virtual enviornment. If you use sudo apt-get instead, make sure to make the connection between the virtual enviornment  and the package outside with -ln command.
	* CMAKE takes lots of time, be patient.
* [OpenCV for Raspberry pi 3](https://nancyyluu.blogspot.com/2017/12/raspberry-pi-opencvcontrib.html?fbclid=IwAR0EQGX7_1VAalSN9g6dk1jNIGuW9GNlx-vQ34T20t1wMoWV4An9lHtFMhk)
	* The version is 3.2.0.
	* Put all the file in bak into bashrc, then run the CMAKE step again. This help us set the default Python enviornment to miniconda. 
* [Picamera](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/3)
	* Use raspistill -o xx.jpg to test if your picamera works fine.
* [imutils](https://pypi.org/project/imutils/)
* [dlib](https://pypi.org/project/dlib/)
* [youtube_dl](https://pypi.org/project/youtube_dl/)
* [pafy](https://pypi.org/project/pafy/)
* [python-vlc](https://pypi.org/project/python-vlc/)

## Implementation step by step

### Step1：Enviorment Configuration

1. Set up the raspberry pi as the picture show as well as carefully install all necessary package mentioned above. 
2. Connect the bluetooth speaker
	1. Go to the bluetooth interface and than check if the volume output has change to your bluetooth device
	
### Step2：

Train model either by [YOLO](https://teachablemachine.withgoogle.com/) or CV2 
* In my model, I utilized cv2 and dlib pretrained model to implement my following project

### Step3:
Define Eye Aspect Ratio
```python
def eye_aspect_ratio(eye):
	# 計算左右眼分別的眼高(歐式距離)
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# 計算雙眼水平距離
	C = dist.euclidean(eye[0], eye[3])

	# 計算EAR
	ear = (A + B) / (2.0 * C)

	# 回傳EAR
	return ear
```

### Step4:
Set up GUI
```python
window = tk.Tk()
window.title('eye_detector')
window.geometry('500x438')

window.configure(background='misty rose')

VECTOR_SIZE = 320

global photo
global label
global calculate_btn
global button
            
        
#header_label = tk.Label(window, text='eye_detecter')
#header_label.pack()
photo=tk.PhotoImage(file=r"C:\Users\KatieLDR\background6.gif")#tkinter僅支援gif壓縮檔
label=tk.Label(window,image=photo)  #圖片

calculate_btn = tk.Button(window, text='start', width=15, height=2,command=eye_detecter,background='#FFFFF0')
#calculate_btn.place(x=100,y=10,anchor='w')
calculate_btn.pack(side='left')

button = tk.Button(text = "quit", width=15, height=2, command = window.destroy,background='#FFFFF0')
button.pack(side='right')

label.pack()
window.mainloop()
```

### Step5: 
Set to import dlib model by command line. I've put the trained model in the respository.

```python
# 利用argument parse 讓我可以在command line 匯入人臉特徵的pretrained model
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())
```

### Step6:

Fetching all facial landmark
![](https://img-blog.csdn.net/20180111135627528?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaG9uZ2Jpbl94dQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

```
EYE_AR_THRESH = 0.3 # EAR 域值
EYE_AR_CONSEC_FRAMES = 3 # 當EAR小於域值時，接連多少frame照片一定有眨眼動作

# 對應dlib特徵點序號
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
RIGHT_EYE_MIDDLE = 40 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1
LEFT_EYE_MIDDLE = 46 - 1

#分別取出左右眼的facial landmark
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
```

### Step6:
Set video steaming with Picamer, the video quaily of VideoStream is a lot better than piRGBArray

```python
vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0) #開啟後等一秒
```
### Step7:

Setting up youtube streaming by Youtube API
```python
# 設定串流音樂撥放器
# url of youtube
url = 'https://www.youtube.com/watch?v=5MP6v-N_FnU&ab_channel=JustinBieberVEVO'
# create pafy object
video = pafy.new(url)
best = video.getbest()
#create vlc media player
media = vlc.MediaPlayer(best.url)
```

### Step 8:
Start identifying every frame in streaming video with following steps:
1. Get the frame and transfer it into gray scale by cv2 function
2. Save the factial landmark into NumPy array
3. Depicit the eye shape
4. Ditermine if 'blink' has taken place in this frame
	1. Check if the EAR is below or blink threshold, if so, increase the frame counter
	2. Check if sufficient number of frame below our predefined threshold, is so, increase the total blink counter
		1. If blink count is odd number, than play the music, otherwise, pause the music 
5. Show the result on window
6. Get the warning if too close to the camera

```python
# 遞迴循環video stream的每個frame
while True:
	
	# Steaming的過程查看buffer還有沒有剩下的frame
	if fileStream and not vs.more():
		break

	# 取得threaded video stream 的frame並轉換成灰階，才能進行特徵辨識
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0) #特徵辨識

	
	for rect in rects:
		# 將人臉特徵存到NumPy Array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# 分別計算兩眼的EAR後再算平均
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		#leftEARS = eye_aspect_ratios(leftEye)
		#rightEARS = eye_aspect_ratios(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		#dis = (leftEARS + rightEARS) / 2.0
		
		# 框出眼睛的形狀
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# 眨眼判斷
		if ear < EYE_AR_THRESH:
			COUNTER += 1
		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				close_counter += 1
				TOTAL += 1
				# 設定播放條件
				if TOTAL%2==1:
					print('play')
					media.play()#start playing
				else:
					print('pause')
					media.pause()#stop playing

			# 重設couter
			COUNTER = 0
		'''
		# 離鏡頭太近會警告
		if dis > 30:
			tkinter.messagebox.showwarning(title='Hi', message='有警告！')
		'''	
		# 將眨眼次數和EAR值放到畫面上
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)	
		#cv2.putText(img, "close:{0}".format(close_counter), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
```


### Step 9: 

Put the pretrained model and python code in the same directory and run the code by following command.
```linux
cd theFileLocation
python MusicPlayer.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

## Demo Video

*Because I detect blinking eyes and fetch the music on youtube simultaneously, so the first blink will take time to get the music* 

[Demo 1](https://youtu.be/trSXjLE6yNU )

[Demo 2](https://youtu.be/1acdpX91gp4)

## Remain unfinished

Because I made a false move, plugging the sensor into false GPIO pin, the gesture sensor burned out accidentally. As a result, I couldn't accomplish the gesture detect part of my project. I've tried to substitute ultrasonic senosr for gesture sensor but it can't work correctly either. Following is the step I've done for volume control so far.

### Python control Raspberry pi volume
1. First download necessary python package
```linux
pip install pyalsaaudio //for audio control
pip install RPi.GPIO 
```
2. Control with python code
```python
import RPi.GPIO as GPIO
import time
import alsaaudio

m = alsaaudio.Mixer('PCM')
vol = m.getvolume()[0]
is_mute = m.getmute()[0]

GPIO.setmode(GPIO.BCM)

def toggle_mute():
    global is_mute
    m.setmute(1 - is_mute)
    is_mute = m.getmute()[0]
    if is_mute:
        print('Muted')
    else:
        print('Un-muted')

vol_up_pin = 14
vol_dn_pin = 4
GPIO.setup(vol_up_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(vol_dn_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while True:
    vu = not GPIO.input(vol_up_pin)
    vd = not GPIO.input(vol_dn_pin)
    if vu and not vd:
        if is_mute:
            toggle_mute()
        if vol < 100:
            vol += 5
            m.setvolume(vol)
            vol = m.getvolume()[0]
        print 'Volume up Pressed', vol
    elif vd and not vu:
        if is_mute:
            toggle_mute()
        if vol > 0:
            vol -= 51. 
            m.setvolume(vol)
            vol = m.getvolume()[0]
        print 'Volume down Pressed', vol
    elif vd and vu:
        toggle_mute()
        time.sleep(0.5)

    time.sleep(0.2)
```
### Gesture sensor
1. Download necessary package
```linux
sudo apt-get install git
sudo git clone git://git.drogon.net/wiringPi
cd wiringPi
sudo ./build

pip install spidev
sudo apt-get install python-imaging
sudo apt-get install python-smbus
sudo apt-get install python-serial
```
2. Enable I2C on Raspberry pi interface
3. Wiring 
![I've grab the photo on the net cause I can't find raspberry pi simulator](https://i.imgur.com/ivNOGYq.png)
4. Control code
```python
class PAJ7620U2(object):
	def __init__(self,address=PAJ7620U2_I2C_ADDRESS):
		self._address = address
		self._bus = smbus.SMBus(1)
		time.sleep(0.5)
		if self._read_byte(0x00) == 0x20:
			print("\nGesture Sensor OK\n")
			for num in range(len(Init_Register_Array)):
				self._write_byte(Init_Register_Array[num][0],Init_Register_Array[num][1])
		else:
			print("\nGesture Sensor Error\n")
		self._write_byte(PAJ_BANK_SELECT, 0)
		for num in range(len(Init_Gesture_Array)):
				self._write_byte(Init_Gesture_Array[num][0],Init_Gesture_Array[num][1])
	def _read_byte(self,cmd):
		return self._bus.read_byte_data(self._address,cmd)
	
	def _read_u16(self,cmd):
		LSB = self._bus.read_byte_data(self._address,cmd)
		MSB = self._bus.read_byte_data(self._address,cmd+1)
		return (MSB	<< 8) + LSB
	def _write_byte(self,cmd,val):
		self._bus.write_byte_data(self._address,cmd,val)
	def check_gesture(self):
		Gesture_Data=self._read_u16(PAJ_INT_FLAG1)
		if Gesture_Data == PAJ_UP:
			print("Up\r\n")
		elif Gesture_Data == PAJ_DOWN:
			print("Down\r\n")
		elif Gesture_Data == PAJ_LEFT:
			print("Left\r\n")	
		elif Gesture_Data == PAJ_RIGHT:
			print("Right\r\n")	
		elif Gesture_Data == PAJ_FORWARD:
			print("Forward\r\n")	
		elif Gesture_Data == PAJ_BACKWARD:
			print("Backward\r\n")
		elif Gesture_Data == PAJ_CLOCKWISE:
			print("Clockwise\r\n")	
		elif Gesture_Data == PAJ_COUNT_CLOCKWISE:
			print("AntiClockwise\r\n")	
		elif Gesture_Data == PAJ_WAVE:
			print("Wave\r\n")	
		return Gesture_Data

if __name__ == '__main__':
	
	import time

	print("\nGesture Sensor Test Program ...\n")

	paj7620u2=PAJ7620U2()

	while True:
		time.sleep(0.05)
		paj7620u2.check_gesture()
```

## Further extention

* [Control volume by moving hand around gesture sensor](https://www.waveshare.com/wiki/PAJ7620U2_Gesture_Sensor)
* [Control play and pause by studying posture accurately, train the model by YOLO and matlab](https://www.ijrti.org/papers/IJRTI2006013.pdf)

## Reference

* [OpenCV face recognition](https://zhuanlan.zhihu.com/p/69127267)
* [Eye Aspect Ratio essay](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf?spm=5176.100239.blogcont336184.8.b7697a07zKT7r&file=05.pdf)
* [Eye landmark detection](https://medium.com/algoasylum/blink-detection-using-python-737a88893825)
* [Youtube Music Fetch](https://linuxconfig.org/how-to-play-audio-with-vlc-in-python#:~:text=player.play()-,Stopping%20And%20Pause,if%20the%20file%20is%20playing.&text=If%20the%20player%20is%20already,altogether%2C%20call%20the%20stop%20method.)
* [Volume Control](https://gist.github.com/peteristhegreat/3c94963d5b3a876b27accf86d0a7f7c0)
