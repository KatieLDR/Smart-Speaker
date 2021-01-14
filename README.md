# IoT Smart-Speaker
## Overview
By looking at the Picamera and blink eye，our Python code will fetch and play studying youtube music automatically. You can stop at any time by blinking at the Picamera again.
## Demo Video

[Demo 1](https://youtu.be/trSXjLE6yNU )

[Demo 2](https://youtu.be/1acdpX91gp4)

## Prerequisites

### Hardware Device

* Raspberry pi 4 *1
* Picamera *1
* Bluetooth Speaker *1

![](https://i.imgur.com/dk6Wgub.png)


### Software

* [raspbian os]()

#### Python package
* [OpenCV](https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/)
* [Picamera](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/3)
* [imutils](https://pypi.org/project/imutils/)
* [dlib](https://pypi.org/project/dlib/)
* [youtube_dl](https://pypi.org/project/youtube_dl/)
* [pafy](https://pypi.org/project/pafy/)
* [python-vlc](https://pypi.org/project/python-vlc/)

### Implementation step by step

#### Step1：Enviorment Configuration

1. Set up the raspberry pi as the picture show as well as install all necessary package mentioned above. Carefully download every package with pip in virtual enviornment because our code will be run in virtual enviornment.
2. Connect the bluetooth speaker
	1. Go to the bluetooth interface and than check if the volume output has change to your bluetooth device
	
#### Step2：

Train model either by [YOLO](https://teachablemachine.withgoogle.com/) or CV2 
* In my model, I utilized cv2 and dlib pretrained model to implement my following project

#### Step3:
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

#### Step4:
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

#### Step5: 
Set to import dlib model by command line. I've put the trained in the respository.

```python
# 利用argument parse 讓我可以在command line 匯入人臉特徵的pretrained model
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())
```

#### Step6:

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

#### Step6:
Set video steaming with Picamer, the video quaily of VideoStream is a lot better than piRGBArray

```python
vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0) #開啟後等一秒
```
#### Step7:

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

##### Step 8:
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


##### Step 9: 

Put the pretrained model and python code in the same directory and run the code by following command.
```linux
cd theFileLocation
python MusicPlayer.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

### Further extention

* [Control volume by moving hand around gesture sensor](https://www.waveshare.com/wiki/PAJ7620U2_Gesture_Sensor)
* [Control play and pause by studying posture accurately, train the model by YOLO and matlab](https://www.ijrti.org/papers/IJRTI2006013.pdf)

### Reference

* [Eye Aspect Ratio essay](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf?spm=5176.100239.blogcont336184.8.b7697a07zKT7r&file=05.pdf)
* [Eye blink detection](https://blog.csdn.net/hongbin_xu/article/details/79033116)
* [Youtube Music Fetch](https://linuxconfig.org/how-to-play-audio-with-vlc-in-python#:~:text=player.play()-,Stopping%20And%20Pause,if%20the%20file%20is%20playing.&text=If%20the%20player%20is%20already,altogether%2C%20call%20the%20stop%20method.)
