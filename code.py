import numpy
import os
import datetime  
import cv2
import face_recognition
path = "Photos"
images = []
student_names = []
mylist = os.listdir(path)
def markAttendance(name):
    f=open("attendance.csv",'a')
    time=datetime.datetime.now()
    snap_time=time.strftime("%d/%m/%Y  %H:%M:%S")
    f.write(name+","+snap_time+"\n")
    f.close()
for img in mylist:
    cimg = cv2.imread(f'{path}/{img}')
    images.append(cimg)
    student_names.append(os.path.splitext(img)[0])
def find_encodings(images):
    encode_list = []
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode1 = face_recognition.face_encodings(img)[0]
        encode_list.append(encode1)
    return encode_list
encode_list=find_encodings(images)
pic = cv2.imread("praneeth.webp")
pics = cv2.resize(pic, (0, 0), None, 0.25, 0.25)
#pics = cv2.cvtColor(pic , cv2.cvtColor_BGR2RGB)
#find encodings of a pic
faces_pic = face_recognition.face_locations(pics)
encode_pic = face_recognition.face_encodings(pics, faces_pic)
for encode_face, face_loc in zip(encode_pic, faces_pic):
    matches = face_recognition.compare_faces(encode_list,encode_face)
    face_dis = face_recognition.face_distance(encode_list,encode_face)
    match_index = numpy.argmin(face_dis)
    if matches[match_index]:
        name = student_names[match_index]
        y1,x2,y2,x1=face_loc
        cv2.rectangle(pic,(x1*4, y1*4),(x2*4,y2*4),(0,0,0),2)
        cv2.putText(pic, name, (x1+6 , y2-6) , cv2.QT_FONT_BLACK, 1, (0,0,0), 2) 
        markAttendance(name)
cv2.imshow("ATTENDANCE", pic)
cv2.waitKey(0)
    

