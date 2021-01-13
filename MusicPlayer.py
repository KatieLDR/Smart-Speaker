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
#import tkinter as tk  # �ϥ�Tkinter�e�ݭn���פJ
#import tkinter.messagebox  # �n�ϥ�messagebox���n�פJ�Ҳ�

#�p��EAR
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
	
''' �쥻�n��tkinter��GUI�e���A�i�O�bPI�W���w�˥X��
window = tk.Tk()
window.title('eye_detector')
window.geometry('500x438')

window.configure(background='misty rose')

VECTOR_SIZE = 320
'''
 
# �Q��argument parse ���ڥi�H�bcommand line �פJ�H�y�S�x��pretrained model
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())
 
EYE_AR_THRESH = 0.3 # EAR ���
EYE_AR_CONSEC_FRAMES = 3 # ��EAR�p���ȮɡA���s�h��frame�Ӥ��@�w���w���ʧ@

# ����dlib�S�x�I�Ǹ�
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
RIGHT_EYE_MIDDLE = 40 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1
LEFT_EYE_MIDDLE = 46 - 1

# ��l��frame counter ��w���p�ƾ�
COUNTER = 0
TOTAL = 0
close_counter = 0

# ��l��dlib�å��}dlib��face landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#���O���X���k����facial landmark
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# �}�lvideo streaming
print("[INFO] starting video stream thread...")
vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0) #�}�ҫᵥ�@��
# �]�w��y���ּ���
# url of youtube
url = 'https://www.youtube.com/watch?v=5MP6v-N_FnU&ab_channel=JustinBieberVEVO'
# create pafy object
video = pafy.new(url)
best = video.getbest()
#create vlc media player
media = vlc.MediaPlayer(best.url)

# ���j�`��video stream���C��frame
while True:
	
	# Steaming���L�{�d��buffer�٦��S���ѤU��frame
	if fileStream and not vs.more():
		break

	# ���othreaded video stream ��frame���ഫ���Ƕ��A�~��i��S�x����
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0) #�S�x����

	
	for rect in rects:
		# �N�H�y�S�x�s��NumPy Array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# ���O�p��Ⲵ��EAR��A�⥭��
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		#leftEARS = eye_aspect_ratios(leftEye)
		#rightEARS = eye_aspect_ratios(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		#dis = (leftEARS + rightEARS) / 2.0
		
		# �إX�������Ϊ�
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# �w���P�_
		if ear < EYE_AR_THRESH:
			COUNTER += 1
		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				close_counter += 1
				TOTAL += 1
				# �]�w�������
				if TOTAL%2==1:
					print('play')
					media.play()#start playing
				else:
					print('pause')
					media.pause()#stop playing

			# ���]couter
			COUNTER = 0
		'''
		# �����Y�Ӫ�|ĵ�i
		if dis > 30:
			tkinter.messagebox.showwarning(title='Hi', message='��ĵ�i�I')
		'''	
		# �N�w�����ƩMEAR�ȩ��e���W
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)	
		#cv2.putText(img, "close:{0}".format(close_counter), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
		
	# ��ܵe��
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# ��q ���X
	if key == ord("q"):
		media.stop#stop vlc        
		break


cv2.destroyAllWindows()
vs.stop()

'''GUI��������
global photo
global label
global calculate_btn
global button
            
        
#header_label = tk.Label(window, text='eye_detecter')
#header_label.pack()
photo=tk.PhotoImage(file=r"C:\Users\KatieLDR\background6.gif")#tkinter�Ȥ䴩gif���Y��
label=tk.Label(window,image=photo)  #�Ϥ�

calculate_btn = tk.Button(window, text='start', width=15, height=2,command=eye_detecter,background='#FFFFF0')
#calculate_btn.place(x=100,y=10,anchor='w')
calculate_btn.pack(side='left')

button = tk.Button(text = "quit", width=15, height=2, command = window.destroy,background='#FFFFF0')
button.pack(side='right')

label.pack()
window.mainloop()
'''