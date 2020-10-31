import cv2

def facechop(image):
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    img = cv2.imread(image)

    minisize = (img.shape[1], img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [v for v in f]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))

        sub_face = img[y:y + h, x:x + w]
        face_file_name = "Faces/" + str(y) + ".jpg"
        cv2.imwrite(face_file_name, sub_face)

    cv2.imshow(image, img)

    return


if __name__ == '__main__':
    facechop("C:/Users/dELL/Pictures/Camera Roll/WIN_20190212_16_28_57_Pro.jpg")

    while (True):
        key = cv2.waitKey(20)
        if key in [27, ord('Q'), ord('q')]:
            break
