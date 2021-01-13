# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import vlc
import pafy
#import tkinter as tk  # 使用Tkinter前需要先匯入
#import tkinter.messagebox  # 要使用messagebox先要匯入模組

#計算EAR
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

'''
def eye_aspect_ratios(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return C
'''
	
''' 原本要用tkinter做GUI畫面，可是在PI上面安裝出錯
window = tk.Tk()
window.title('eye_detector')
window.geometry('500x438')

window.configure(background='misty rose')

VECTOR_SIZE = 320
'''
 
# 利用argument parse 讓我可以在command line 匯入人臉特徵的pretrained model
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())
 
EYE_AR_THRESH = 0.3 # EAR 域值
EYE_AR_CONSEC_FRAMES = 3 # 當EAR小於域值時，接連多少frame照片一定有眨眼動作

# 對應dlib特徵點序號
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
RIGHT_EYE_MIDDLE = 40 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1
LEFT_EYE_MIDDLE = 46 - 1

# 初始化frame counter 跟眨眼計數器
COUNTER = 0
TOTAL = 0
close_counter = 0

# 初始化dlib並打開dlib的face landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#分別取出左右眼的facial landmark
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 開始video streaming
print("[INFO] starting video stream thread...")
vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0) #開啟後等一秒
# 設定串流音樂撥放器
# url of youtube
url = 'https://www.youtube.com/watch?v=5MP6v-N_FnU&ab_channel=JustinBieberVEVO'
# create pafy object
video = pafy.new(url)
best = video.getbest()
#create vlc media player
media = vlc.MediaPlayer(best.url)

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
		
	# 顯示畫面
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# 按q 跳出
	if key == ord("q"):
		media.stop#stop vlc        
		break


cv2.destroyAllWindows()
vs.stop()

'''GUI視窗部分
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
'''