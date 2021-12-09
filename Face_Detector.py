import  cv2
import random

# Load some pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Chose and image
img = cv2.imread('face1.jpeg')
# Campute video from webcamera
webcam = cv2.VideoCapture(0)
# interate forever over frames
while True:
  # Read the current frame
  successful_frame_read, frame = webcam.read()
  # must covert to grayscaled
  grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces coordinates
  face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
  # Draw rectangles around the face whit colors
  for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x,y), (x+w, y+h),(random.randrange(256),random.randrange(256),random.randrange(256)),2)
  print(face_coordinates)

  #
  cv2.imshow('Image', frame)
  cv2.waitKey(1)

print ("Code Completed")
