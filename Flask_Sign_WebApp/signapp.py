# -*- coding: utf-8 -*-
from flask import Flask, Response, render_template, request, redirect, url_for
import re
import cv2
import numpy as np
import mediapipe as mp
import mysql.connector
from tensorflow import keras
import pandas as pd
import face_recognition
from io import BytesIO
from detect_function import *

app = Flask(__name__)
cap = cv2.VideoCapture(0)

host = 'localhost'
user = 'root'
password = 'root'
database = 'users'
con = mysql.connector.connect(host=host,user=user, password=password, database=database)
cursor = con.cursor()

host = 'localhost'
user = 'root'
password = 'root'
database = 'face_recognition'
conn = mysql.connector.connect(host=host,user=user, password=password, database=database)
cur = conn.cursor()

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L',
                    'M','N','O','P','Q','R','S','T','U','V','W','X',
                    'Y','Z','0','1','2','3','4','5','6','7','8','9',
                    'Thank You','Namaste','Yes','No'])

model = keras.models.load_model(r'RealtimeSign.h5')
model.load_weights(r'RealtimeSign.h5')

cnn_model = keras.models.load_model(r'CNN_Model.h5')
cnn_model.load_weights(r'CNN_Model.h5')


labels = pd.read_csv("labels.csv").values

def frames():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            else:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)
        
                # Draw landmarks
                draw_styled_landmarks(image, results)
        
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
        
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
            
            
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                    
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 1: 
                        sentence = sentence[-1:]
            
                cv2.rectangle(image, (0, 410), (769, 505), (8, 8, 8, 0.75), -1)
                cv2.putText(image, ' '.join(sentence), (300,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
def face():
    cur.execute("select labels,images from face")
    rows = cur.fetchall()

    known_face_names=[]
    known_face_encodings =[]

    for r in rows:
        known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(BytesIO(r[1])))[0])
        known_face_names.append(r[0])
    while True:
        ret, frame = cap.read()

        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            global name
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/upload', methods =['GET', 'POST'])
def upload():
    msg = ''
    if request.method == 'POST':
        labels = request.form['labels']
        image = request.files['images']
        img = image.read()
        cur.execute('SELECT * FROM face WHERE labels =%s', (labels, ))
        face = cur.fetchone()
        if face:
            msg = 'Already Registered!'
        else:
            cur.execute('INSERT INTO face VALUES (%s,%s)', (labels, img,))
            conn.commit()
            msg = 'You are successfully registered !'
        return render_template('face_login.html', msg = msg)
    elif request.method == 'POST':
        msg = 'Register Yourself First!'
    return render_template('face_register.html', msg = msg)


@app.route('/login', methods =['GET', 'POST'])
def login():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
		username = request.form['username']
		password = request.form['password']
		cursor.execute('SELECT * FROM accounts WHERE username =%s  AND password=%s', (username, password, ))
		account = cursor.fetchone()
		if account:
			msg = 'Logged in successfully !'
			return render_template('face_login.html', msg = msg)
		else:
			msg = 'Incorrect username / password !'
	return render_template('login.html', msg = msg)

@app.route('/register', methods =['GET', 'POST'])
def register():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form :
		username = request.form['username']
		password = request.form['password']
		email = request.form['email']
		cursor.execute('SELECT * FROM accounts WHERE username =%s', (username, ))
		account = cursor.fetchone()
		if account:
			msg = 'Account already exists !'
		elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
			msg = 'Invalid email address !'
		elif not re.match(r'[A-Za-z0-9]+', username):
			msg = 'Username must contain only characters and numbers !'
		elif not username or not password or not email:
			msg = 'Please fill out the form !'
		else:
			cursor.execute('INSERT INTO accounts VALUES (NULL,%s,%s,%s)', (username, password, email, ))
			con.commit()
			msg = 'You have successfully registered !'
	elif request.method == 'POST':
		msg = 'Please fill out the form !'
	return render_template('register.html', msg = msg)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = request.files['images']
        img.save("img.jpg")
        
        image = cv2.imread("img.jpg")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, (90,90))
        
        image = np.reshape(image, (1,90,90,3))
        
        pred = cnn_model.predict(image)
        
        pred = np.argmax(pred)
 
        pred = labels[pred]              
        return render_template('predict.html', prediction = pred)
    return None
 
@app.route('/')
def start():
    return render_template('startpage.html')

@app.route('/logout')
def logout():
	return redirect(url_for('login'))

@app.route('/face_register')
def face_register():
    return render_template('face_register.html')

@app.route('/face_login')
def face_login():
    return render_template('face_login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/CNN_index')
def CNN_index():
    return render_template('CNN_index.html')

@app.route('/welcome')
def welcome():
    if name == "Unknown":
        return render_template('face_register.html')
    else:
        return render_template('welcome.html')

@app.route('/verify_face')
def verify_face():
    return Response(face(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
