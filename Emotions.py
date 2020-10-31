from fileinput import filename
import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image


from Module.Testing import Testing

win = tk.Tk()
win.title('IMAGE CLASSIFICATION USING MACHINE LEARNING')
win.geometry("800x800")
status = ""
class gui:
    def camera(self):
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Camera -- Hit Space to capture")

        img_counter = 0

        while True:
            facedata = "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(facedata)
            img = cv2.imread("test.png")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #gray = cv2.resize(gray, (60, 60))
            #gray = cv2.resize(gray, (200, 200))
            minisize = (img.shape[1], img.shape[0])
            miniframe = cv2.resize(gray, minisize)
            #faces = cascade.detectMultiScale(miniframe)
            faces = cascade.detectMultiScale(
                img,
                scaleFactor=1.3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for f in faces:
                x, y, w, h = [v for v in f]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255))

                sub_face = img[y:y + h, x:x + w]
                cv2.imwrite("test.png", sub_face)
            ret, frame = cam.read()
            cv2.imshow("Camera -- Hit Space to capture", frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                messagebox.showinfo("Status", "Captured Successfully")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = "test.png"
                cv2.imwrite(img_name, frame)
                print("wrtiten")

        cam.release()

        cv2.destroyAllWindows()

        self.img = cv2.imread("test.png")
        self.gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.resized_image = cv2.resize(self.gray_image, (48, 48))
        self.image = np.array(self.resized_image, dtype=np.float32)
        self.image = self.image.reshape(-1, 48, 48, 1)/255

    def img(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                                   filetype=(("jpeg", "*.jpg"), ("All Files", "*.*")))
        facedata = "haarcascade_frontalface_default.xml"
        img = cv2.imread(self.filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(facedata)
        minisize = (img.shape[1], img.shape[0])
        miniframe = cv2.resize(img, minisize)
        #faces = cascade.detectMultiScale(miniframe)
        faces = cascade.detectMultiScale(
            img,
            scaleFactor=1.2,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for f in faces:
            x, y, w, h = [v for v in f]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255))

            sub_face = img[y:y + h, x:x + w]
            cv2.imwrite("test1.jpg", sub_face)
        messagebox.showinfo("Status", "File Selected Successfully")
        self.img = cv2.imread("test1.jpg")
        self.gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.resized_image = cv2.resize(self.gray_image, (48, 48))
        self.image = np.array(self.resized_image, dtype=np.float32)
        self.image = self.image.reshape(-1, 48, 48, 1) / 255
        print(self.image)

    def pred(self):
        self.cnn = Testing()
        self.predict = self.cnn.testModel(self.image)
        status = self.predict
        if status == 0:
            messagebox.showinfo("Result", "Anger")
        elif status == 1:
            messagebox.showinfo("Result", "disgust")

            #lbl = Label(topFrame, text="disgust", fg="purple", font=("Arial Bold", 10))
            # lbl.grid(row=0, column=0)
            #lbl.pack(side=BOTTOM)
        elif status == 2:
            messagebox.showinfo("Result", "Fear")

            #lbl = Label(topFrame, text="fear", fg="purple", font=("Arial Bold", 10))
            # lbl.grid(row=0, column=0)
            #lbl.pack(side=BOTTOM)
        elif status == 3:
           # lbl = Label(topFrame, text="happy", fg="purple", font=("Arial Bold", 10))
            # lbl.grid(row=0, column=0)
            #lbl.pack(side=BOTTOM)
            messagebox.showinfo("Result", "Happy")

        elif status == 4:
           # lbl = Label(topFrame, text="sad", fg="purple", font=("Arial Bold", 10))
            # lbl.grid(row=0, column=0)
            #lbl.pack(side=BOTTOM)
            messagebox.showinfo("Result", "Sad")

        elif status == 5:
            #lbl = Label(topFrame, text="Surprise", fg="purple", font=("Arial Bold", 10))
            # lbl.grid(row=0, column=0)
            #lbl.pack(side=BOTTOM)
            messagebox.showinfo("Result", "Surprise")

        elif status == 6:
            #lbl = Label(topFrame, text="Neutral", fg="purple", font=("Arial Bold", 10))
            # lbl.grid(row=0, column=0)
            #lbl.pack(side=BOTTOM)
            messagebox.showinfo("Result", "Neutral")


        #anger = 0, disgust = 1, fear = 2, happy = 3, sad = 4, surprise = 5, neutral = 6
        print(self.image)

obj = gui()
image = Image.open('ad.png')
photo_image = ImageTk.PhotoImage(image)
label = tk.Label(win, image=photo_image)
label.pack(fill=BOTH, expand=YES)
label.bind('<Configure>', photo_image)
topFrame = Frame(label)
topFrame.pack(side=TOP)
bottomFrame = Frame(label)
bottomFrame.pack(side=BOTTOM)
#label.bind('<Configure>', photo_image)

lbl = Label(topFrame, text="IMAGE CLASSIFICATION USING MACHINE LEARNING", font=("Arial Bold", 35))
#lbl.grid(row=0, column=0)
lbl.pack(side=TOP)

btn = Button(bottomFrame, text="Click me", fg="black", bg="#8fb4ef", command=obj.img, width=25, height=8, font=('Helvetica'))
#btn.grid(row=10, column=2)
btn.pack(side=LEFT)

btn_1 = Button(bottomFrame, text="Prediction", fg="black", bg="#8fb4ef", command=obj.pred, width=25, height=8, font=('Helvetica'))
btn_1.pack(side=RIGHT)

btn_2=Button(bottomFrame, text="Capture photo",fg="black", bg="#8fb4ef", command=obj.camera, width=25, height=8, font=('Helvetica'))
btn_2.pack()

win.mainloop()