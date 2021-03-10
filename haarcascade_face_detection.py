import cv2 as cv

face_cascade = cv.CascadeClassifier("opencv\haarcascade_frontalface_default.xml")

#read the input image
img = cv.imread("team.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#the faces variable will be the vector of rectangle where each rectangle contains the detected object(face)
faces = face_cascade.detectMultiScale(gray, 1.1,)

#then iterate these faces values in a for loop
for (x,y,w,h) in faces: # x,y,w,h are the four parameters of a rectangle(face variable)
    cv.rectangle(img,(x,y), (x+w, y+h), (255,0,0), 3)

cv.imshow("img",img)
cv.waitKey(0)
cv.destroyAllWindows()    