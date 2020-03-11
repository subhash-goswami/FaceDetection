import cv2
from flask import Flask, render_template, request

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


@app.route('/face')
def face():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Read the input Frames from videos
    cap = cv2.VideoCapture('test3.mp4')

    while cap.isOpened():
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (260, 0, 0), 2)

        # Display output
        cv2.imshow('img', img)
        if cv2.waitKey(1    ) & 0xFF == ord('q'):
            break

    cap.release()


# # Read the input image
@app.route('/face_image')
def imgage_de():
    img = cv2.imread('2.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (260, 0, 0), 2)

    # Display output
    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    app.run(debug=True)
