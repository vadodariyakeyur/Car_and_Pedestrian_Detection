import cv2

#input video
video = cv2.VideoCapture('Pedestrians Compilation.mp4')

#Pre-Trained Car Classifiers
car_classifier_file = 'car.xml'
pedestrian_classifier_file = 'pedestrian.xml'

while True:
    #Read Current Frame
    read_successfull, frame = video.read()
    
    #if video ends break out of loop
    if read_successfull:
        #create frame to grayscale
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    #create classifiers
    car_tracker = cv2.CascadeClassifier(car_classifier_file)
    pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)

    #detect cars of any scale
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    
    #drawing bounding boxes on the color image
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow("Detecting Cars and Pedestrians",frame)
    
    
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break


video.release()