import cv2
import tkinter as tk
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeach
from datetime import datetime
from PIL import ImageTk, Image
from tkinter import filedialog
import pandas as pd

engine = textSpeach.init()

path = 'student_images'
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])


def findEncoding(images):
    imgEncodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings


EncodeList = findEncoding(studentImg)


def MarkAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            timeStr = now.strftime('%H:%M')
            f.write(f'\n{name}, {timeStr}')
            statement = f'Welcome to class, {name}'
            engine.say(statement)
            engine.runAndWait()


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "FAMS - Face Recognition Based Attendance Management System")
        self.root.geometry('800x500')
        self.root.configure(background='black')

        self.title_label = tk.Label(root, text="Attendance System", font=(
            "Helvetica", 24), bg='black', fg='white')
        self.title_label.pack(pady=20)

        self.start_button = tk.Button(root, text="Start Attendance", command=self.start_attendance, font=(
            "Helvetica", 14), bg='#4caf50', fg='white')
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop Attendance", command=self.stop_attendance,
                                     state=tk.DISABLED, font=("Helvetica", 14), bg='#f44336', fg='white')
        self.stop_button.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image, font=(
            "Helvetica", 14), bg='#2196f3', fg='white')
        self.upload_button.pack(pady=10)

        self.vid = cv2.VideoCapture(0)
        self.canvas = tk.Canvas(root, width=640, height=480, bg='white')
        self.canvas.pack()

        self.attendance_started = False
        self.update()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", ".jpg;.jpeg;*.png")])
        if file_path:
            uploaded_image = cv2.imread(file_path)
            faces_in_uploaded_image = face_rec.face_locations(uploaded_image)
            encode_faces_in_uploaded_image = face_rec.face_encodings(
                uploaded_image, faces_in_uploaded_image)

            for encode_face, face_loc in zip(encode_faces_in_uploaded_image, faces_in_uploaded_image):
                matches = face_rec.compare_faces(EncodeList, encode_face)
                face_distances = face_rec.face_distance(
                    EncodeList, encode_face)
                match_index = np.argmin(face_distances)

                if matches[match_index]:
                    name = studentName[match_index]
                    MarkAttendance(name)

            if not os.path.exists("uploaded_images"):
                os.makedirs("uploaded_images")
            image_name = os.path.basename(file_path)
            cv2.imwrite(os.path.join("uploaded_images",
                        image_name), uploaded_image)

    def start_attendance(self):
        self.attendance_started = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def stop_attendance(self):
        self.attendance_started = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update(self):
        success, frame = self.vid.read()
        if self.attendance_started and success:
            facesInFrame = face_rec.face_locations(frame)
            encodeFacesInFrame = face_rec.face_encodings(frame, facesInFrame)

            for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame):
                matches = face_rec.compare_faces(EncodeList, encodeFace)
                facedis = face_rec.face_distance(EncodeList, encodeFace)
                matchIndex = np.argmin(facedis)

                if matches[matchIndex]:
                    name = studentName[matchIndex].upper()
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.rectangle(frame, (x1, y2-25), (x2, y2),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1+6, y2-6),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    MarkAttendance(name)

            photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(photo))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.photo = photo

        self.root.after(10, self.update)


if __name__ == "__main__":
    window = tk.Tk()
    app = AttendanceApp(window)
    window.mainloop()
