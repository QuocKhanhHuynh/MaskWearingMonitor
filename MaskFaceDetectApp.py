from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from keras.preprocessing import image
from numpy import argmax

model = YOLO('best_float16.tflite')
model_lenet=keras.models.load_model("gender_detect_lenet.h5", compile=False)


def detectGender(img):
    categories = ['men', 'women']
    image = img
    image = cv2.resize(image, (64, 64)) 

    image = np.array(image, dtype="float") / 255.0            
    image=np.expand_dims(image, axis=0)

    pred=model_lenet.predict(image)
    Res=argmax(pred,axis=1)
        
    return categories[Res[0]]


def selectPic():
    global lbImage, image
    countMenMask = 0
    countMenNoMask = 0
    countWoMenMask = 0
    countWoMenNoMask = 0
    imageFileName = filedialog.askopenfilename(filetypes=(("jpg images","*.jpg"), ("png images","*.png")))
    frame = cv2.imread(imageFileName)
    (h, w, d) = frame.shape
    r = 800.0 / w
    dim = (800, int(h * r))
    frame = cv2.resize(frame, dim)
    results = model(frame)[0]
    i = 0
    for result in results.boxes.data.tolist():
        i = i +1
        x1, y1, x2, y2, score, class_id = result
        if score > 0.2:
            X1 = int(x1),
            X2 = int(x2),
            Y1 = int(y1),
            Y2 = int(y2)
            if (int(y1-(50/100.0*abs(y2-y1))) >= 0):
                Y1 = int(y1-(50/100.0*abs(y2-y1)))
            if (int(y2+(50/100.0*abs(y2-y1))) <= h):
                Y2 = int(y2+(50/100.0*abs(y2-y1)))
            if (int(x1-(50/100.0*abs(x2-x1)) >= 0)):
                X1 = int(x1-(50/100.0*abs(x2-x1)))
            if (int(x2+(50/100.0*abs(x2-x1))) <= w):
                X2 = int(x2+(50/100.0*abs(x2-x1)))
            if isinstance(X1, tuple):
                X1 = X1[0]
            if isinstance(Y1, tuple):
                Y1 = Y1[0]
            if isinstance(X2, tuple):
                X2 = X2[0]
            if isinstance(Y2, tuple):
                Y2 = Y2[0]
            cropped_image = frame[Y1:Y2, X1:X2]
            gender = detectGender(cropped_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            if int(class_id) == 0:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),  (192, 192, 255), 2)
                #cv2.imwrite('cut' + i + '.jpg', cropped_image)
                
                cv2.putText(frame, "no-mask" + " " + "{:.2f}".format(score * 100), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (192, 192, 255), 2, cv2.LINE_AA)
                if gender == 'men':
                    countMenNoMask += 1
                else:
                    countWoMenNoMask += 1 
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),  (0, 0, 255), 2)
                cv2.putText(frame, "mask" + " " + "{:.2f}".format(score * 100), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 255), 2, cv2.LINE_AA)
                if gender == 'men':
                    countMenMask += 1
                else:
                    countWoMenMask += 1
    cv2.imwrite('demoImage.jpg', frame)
    image = Image.open('demoImage.jpg')
    image = image.resize(dim)
    image = ImageTk.PhotoImage(image)
    lbImage.config(image=image)
    lbetMenMask.config(text=str(countMenMask))
    lbetMenNoMask.config(text=str(countMenNoMask))
    lbetWoMenMask.config(text=str(countWoMenMask))
    lbetWoMenNoMask.config(text=str(countWoMenNoMask))

def selectVideo():
    global lbImage, image, lbetMenMask, lbetMenNoMask, lbetWoMenMask, lbetWoMenNoMask
    videoFileName = filedialog.askopenfilename(filetypes=(("mp4 video","*.mp4"),))

    cap = cv2.VideoCapture(videoFileName)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    while True:
        countMenMask = 0
        countMenNoMask = 0
        countWoMenMask = 0
        countWoMenNoMask = 0
        ret, frame = cap.read()
        (h, w, d) = frame.shape
        r = 800.0 / w
        dim = (800, int(h * r))
        frame = cv2.resize(frame, dim)
        if not ret:
            print("Error: Unable to read frame.")
            break

        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > 0.3:
                #cropped_image = frame[int(y1-(1/100.0*h)):int(y2+(1/100.0*h)), int(x1-(1/100.0*w)):int(x2+(1/100.0*w))]
                X1 = int(x1),
                X2 = int(x2),
                Y1 = int(y1),
                Y2 = int(y2)
                if (int(y1-(50/100.0*abs(y2-y1))) >= 0):
                    Y1 = int(y1-(50/100.0*abs(y2-y1)))
                if (int(y2+(50/100.0*abs(y2-y1))) <= h):
                    Y2 = int(y2+(50/100.0*abs(y2-y1)))
                if (int(x1-(50/100.0*abs(x2-x1)) >= 0)):
                    X1 = int(x1-(50/100.0*abs(x2-x1)))
                if (int(x2+(50/100.0*abs(x2-x1))) <= w):
                    X2 = int(x2+(50/100.0*abs(x2-x1)))
                if isinstance(X1, tuple):
                    X1 = X1[0]
                if isinstance(Y1, tuple):
                    Y1 = Y1[0]
                if isinstance(X2, tuple):
                    X2 = X2[0]
                if isinstance(Y2, tuple):
                    Y2 = Y2[0]
                cropped_image = frame[Y1:Y2, X1:X2]
                gender = detectGender(cropped_image)

                if int(class_id) == 0:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),  (192, 192, 255), 2)
                    #cv2.imwrite('cut' + i + '.jpg', cropped_image)
                    
                    cv2.putText(frame, "no-mask" + " " + "{:.2f}".format(score * 100), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (192, 192, 255), 2, cv2.LINE_AA)
                    if gender == 'men':
                        countMenNoMask += 1
                    else:
                        countWoMenNoMask += 1 
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),  (0, 0, 255), 2)
                    cv2.putText(frame, "mask" + " " + "{:.2f}".format(score * 100), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 255), 2, cv2.LINE_AA)
                    if gender == 'men':
                        countMenMask += 1
                    else:
                        countWoMenMask += 1
        cv2.imwrite('demoVideo.jpg', frame)
        image = Image.open('demoVideo.jpg')
        image = image.resize(dim)
        image = ImageTk.PhotoImage(image)
        lbImage.config(image=image)
        lbImage.update_idletasks()
        lbetMenMask.config(text=str(countMenMask))
        lbetMenMask.update_idletasks()
        lbetMenNoMask.config(text=str(countMenNoMask))
        lbetMenNoMask.update_idletasks()
        lbetWoMenMask.config(text=str(countWoMenMask))
        lbetWoMenMask.update_idletasks()
        lbetWoMenNoMask.config(text=str(countWoMenNoMask))
        lbetWoMenNoMask.update_idletasks()
        

def selectCamera():
    global lbImage, image, lbMenMask, lbWoMenMask, lbMenNoMask, lbWoMenNoMask

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    while True:
        countMenMask = 0
        countMenNoMask = 0
        countWoMenMask = 0
        countWoMenNoMask = 0
        ret, frame = cap.read()
        (h, w, d) = frame.shape
        r = 800.0 / w
        dim = (800, int(h * r))
        frame = cv2.resize(frame, dim)
        
        if not ret:
            print("Error: Unable to read frame.")
            break

        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > 0.3:
                X1 = int(x1),
                X2 = int(x2),
                Y1 = int(y1),
                Y2 = int(y2)
                if (int(y1-(50/100.0*abs(y2-y1))) >= 0):
                    Y1 = int(y1-(50/100.0*abs(y2-y1)))
                if (int(y2+(50/100.0*abs(y2-y1))) <= h):
                    Y2 = int(y2+(50/100.0*abs(y2-y1)))
                if (int(x1-(50/100.0*abs(x2-x1)) >= 0)):
                    X1 = int(x1-(50/100.0*abs(x2-x1)))
                if (int(x2+(50/100.0*abs(x2-x1))) <= w):
                    X2 = int(x2+(50/100.0*abs(x2-x1)))
                if isinstance(X1, tuple):
                    X1 = X1[0]
                if isinstance(Y1, tuple):
                    Y1 = Y1[0]
                if isinstance(X2, tuple):
                    X2 = X2[0]
                if isinstance(Y2, tuple):
                    Y2 = Y2[0]
                cropped_image = frame[Y1:Y2, X1:X2]
                gender = detectGender(cropped_image)
                if int(class_id) == 0:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),  (192, 192, 255), 2)
                    
                    cv2.putText(frame, "no-mask" + " " + "{:.2f}".format(score * 100), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (192, 192, 255), 2, cv2.LINE_AA)
                    if gender == 'men':
                        countMenNoMask += 1
                    else:
                        countWoMenNoMask += 1 
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),  (0, 0, 255), 2)
                    cv2.putText(frame, "mask" + " " + "{:.2f}".format(score * 100), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 255), 2, cv2.LINE_AA)
                    if gender == 'men':
                        countMenMask += 1
                    else:
                        countWoMenMask += 1
        cv2.imwrite('demoCamera.jpg', frame)
        image = Image.open('demoCamera.jpg')
        image = image.resize(dim)
        image = ImageTk.PhotoImage(image)
        lbImage.config(image=image)
        lbImage.update_idletasks()
        lbetMenMask.config(text=str(countMenMask))
        lbetMenMask.update_idletasks()
        lbetMenNoMask.config(text=str(countMenNoMask))
        lbetMenNoMask.update_idletasks()
        lbetWoMenMask.config(text=str(countWoMenMask))
        lbetWoMenMask.update_idletasks()
        lbetWoMenNoMask.config(text=str(countWoMenNoMask))
        lbetWoMenNoMask.update_idletasks()

window = Tk()
window.configure(bg="#ADD8E6")

    
#window.geometry('1200x600')
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Đặt kích thước và vị trí của cửa sổ
window.geometry(f"{screen_width}x{screen_height}+0+0")
window.title("Monitor mask wearers")
window.overrideredirect(True)

btnExist = Button(window, text="Exit", command=quit, width=9, height=1).place(x=1190, y= 10)
lbTitle = Label(window, text="MONITOR MASK WEARERS", bg="#ADD8E6", fg="red", font=("Helvetica", 20, "bold"))
lbTitle.place(x=450, y=40)
lbImage = Label(window, padx=400, pady=266, bg="grey")
lbImage.place(x=60, y=120)
btnImage = Button(window, text="Image", width=9, height=1, command=selectPic).place(x=340, y= 680)
btnVideo = Button(window, text="Video", width=9, command=selectVideo).place(x=420, y=680)
btnCamera = Button(window, text="Camera", width=9, command=selectCamera).place(x=500, y=680)

group_box = LabelFrame(window, text="Information", bg="#ADD8E6", padx=5, pady=5, font=("Helvetica", 15, "bold"))
group_box.place(x=870, y=110, width=350, height=560)
lbMen = Label(window, text="Men", bg="#ADD8E6", fg="black", font=("Helvetica", 15, "bold"))
lbMen.place(x=900, y=146)
lbMenMask = Label(window, text="Mask", bg="#ADD8E6", fg="black")
lbMenMask.place(x=910, y=176)
lbetMenMask = Label(window,width=10, bg='white')
lbetMenMask.place(x=950, y=176)
lbMenNoMask = Label(window, text="No Mask", bg="#ADD8E6", fg="black")
lbMenNoMask.place(x=1050, y=176)
lbetMenNoMask = Label(window,width=10, bg='white')
lbetMenNoMask.place(x=1110, y=176)
lbWoMen = Label(window, text="WoMen", bg="#ADD8E6", fg="black", font=("Helvetica", 15, "bold"))
lbWoMen.place(x=900, y=216)
lbWoMenMask = Label(window, text="Mask", bg="#ADD8E6", fg="black")
lbWoMenMask.place(x=910, y=246)
lbetWoMenMask = Label(window,width=10, bg='white')
lbetWoMenMask.place(x=950, y=246)
lbWoMenNoMask = Label(window, text="No Mask", bg="#ADD8E6", fg="black")
lbWoMenNoMask.place(x=1050, y=246)
lbetWoMenNoMask = Label(window,width=10, bg='white')
lbetWoMenNoMask.place(x=1110, y=246)

window.mainloop()