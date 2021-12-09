import  cv2

# Lodad some pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Chose and image
img = cv2.imread('face1.jpeg')
# Campute video from webcamera
webcam = cv2.VideoCapture(0)

# Must convert to grayscale 
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces coordinates
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around the face
for (x, y, w, h) in face_coordinates:
  cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)

print(face_coordinates)

#
cv2.imshow('Image', img)

#
cv2.waitKey()
 
print ("Code Completed")
