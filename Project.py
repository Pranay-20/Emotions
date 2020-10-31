import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("Camera -- Hit Space to capture")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("Camera -- Hit Space to capture", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "test.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

cam.release()

cv2.destroyAllWindows()

facedata = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)
img = cv2.imread("test.png")

minisize = (img.shape[1], img.shape[0])
miniframe = cv2.resize(img, minisize)

faces = cascade.detectMultiScale(miniframe)

for f in faces:
    x, y, w, h = [v for v in f]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))

    sub_face = img[y:y + h, x:x + w]
    #face_file_name = "Faces/" + str(y) + ".jpg"
    cv2.imwrite("test.png", sub_face)

cv2.imshow("test.png", img)

