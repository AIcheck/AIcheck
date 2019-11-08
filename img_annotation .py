# -*- coding: utf-8 -*-
 
import cv2
import sys
import numpy as np 
import os
 
def search(dir):
    files = os.listdir(dir)
    return files
dest_anno_dir = 'data/face/test_img/annotations/'   
name=search('data/face/test_img/images')
# 사용자 입력
for i in name:
    imagePath = './data/face/test_img/images/'+str(i)
    # cascPath = sys.argv[2]
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # from datetime import datetime
    # start_time = datetime.now()

    # 계산 반복 횟수 (한번만 처리하려면 아래를 1로 하거나 for문을 제거하세요)
    #iteration_count = 100
    # for cnt in range():

        # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,     # 이미지에서 얼굴 크기가 서로 다른 것을 보상해주는 값
            minNeighbors=5,    # 얼굴 사이의 최소 간격(픽셀)입니다
            minSize=(30, 30),   # 얼굴의 최소 크기입니다
        )

        # 검출된 얼굴 주변에 사각형 그리기
    rectangle=[]
    for (x, y, w, h) in faces:

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        t = float(x), float(y), float(x+w), float(y+h)
        rectangle.append(list(t))
    # end_time = datetime.now()
    

    #     name = '_'.join(i.split('/'))
    #     src = os.path.join(img_dir, '{}.jpg'.format(i))
    #     dest = os.path.join(dest_img_dir, '{}.jpg'.format(name))
    #     copyfile(src, dest)

    #     annos = list_dict[i]


        #annotations = OrderedDict()
    annotations = dict()
    annotations['face']=rectangle
    if len(str(i)) == 10:
        file_name=str(i)[:6]
    elif len(str(i)) == 11:
        file_name=str(i)[:7]
    with open(os.path.join(dest_anno_dir, '{}.anno'.format(file_name)), 'w', encoding="utf-8") as m:
        json.dump(annotations, m, ensure_ascii=False, indent="\t")
