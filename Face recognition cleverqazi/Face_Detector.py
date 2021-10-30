import cv2

trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img=cv2.imread('C:\\Users\\Dhruv Mittal\\Desktop\\dhruv_formal_pic.jpg')

grayscale_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscale_img)


print(face_coordinates)


cv2.imshow('Cleverqazi Face Detector',grayscale_img)

cv2.waitKey()
print("Code Completed")