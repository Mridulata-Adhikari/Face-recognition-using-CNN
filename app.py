import binascii
import os
import cv2
from flask import Flask, jsonify, request, render_template, Response, redirect, url_for, send_from_directory
from source.face_recognition import recognize_faces

from source.utils import draw_rectangles, read_image, prepare_image
from datetime import datetime
from time import gmtime, strftime, localtime
import requests
import os
import cv2
from flask import Flask, jsonify, request, render_template, Response
from source.face_recognition import recognize_faces
from source.utils import draw_rectangles, read_image, prepare_image
from source.model_training import create_mlp_model

import pandas as pd
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
from imutils import paths
import pickle
import time
import cv2
import os
import csv
from collections import defaultdict



app = Flask(__name__)

app.config.from_object('config')
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app = Flask("Flask Image Gallery")
app.config['IMAGE_EXTS'] = [".png", ".jpg", ".jpeg", ".gif", ".tiff"]


def encode(x):
    return binascii.hexlify(x.encode('utf-8')).decode()

def decode(x):
    return binascii.unhexlify(x.encode('utf-8')).decode()

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/recognize', methods=['POST'])
def detect():
    file = request.files['image']

    # Read image
    image = read_image(file)

    # Recognize faces
    classifier_model_path = "models" + os.sep + "finalrecognizer.pickle"
    label_encoder_path = "models" + os.sep + "finalle.pickle"
    faces = recognize_faces(image, classifier_model_path, label_encoder_path,
                            detection_api_url=app.config["DETECTION_API_URL"])

    return jsonify(recognitions=faces)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']

    # Read image
    image = read_image(file)

    # Recognize faces
    classifier_model_path = "models" + os.sep + "finalrecognizer.pickle"
    label_encoder_path = "models" + os.sep + "finalle.pickle"
    faces = recognize_faces(image, classifier_model_path, label_encoder_path,
                            detection_api_url="http://127.0.0.1:3000/")

    # Draw detection rects
    draw_rectangles(image, faces)

    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('stillphoto.html', face_recognized=len(faces) > 0, num_faces=len(faces), image_to_show=to_send,
                           init=True)



@app.route('/static')
def static_page():
    return render_template('stillphoto.html')



@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

video = cv2.VideoCapture(0)

def gen(video):

    cleaner = pd.read_csv('attendance-system.csv')
    cleaner.to_csv('attendance-system.csv', index=False)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detector", default="face_detection_model",
                    help="path to OpenCV's deep learning face detector")
    ap.add_argument("-m", "--embedding-model", default="models/openface_nn4.small2.v1.t7",
                    help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-r", "--recognizer", default="models/finalrecognizer.pickle",
                    help="path to model trained to recognize faces")
    ap.add_argument("-l", "--le", default="models/finalle.pickle",
                    help="path to label encoder")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(args["recognizer"], "rb").read())
    le = pickle.loads(open(args["le"], "rb").read())

    # initialize the video stream, then allow the camera sensor to warm up

    # start the FPS throughput estimator
    fps = FPS().start()
    faces_list = []
    proba_list = []
    proba = 0
    count = 0
    now = datetime.now()
    dictionaryin = {}
    dictionaryout = {}

    unknown_counter = 0

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        success, image = video.read()

        frame = image
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        dt_string = now.strftime("%d/%m/%Y")
        hr_string = strftime("%H:%M:%S", localtime())

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                img_counter = 0

                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

                # print(le.classes_)

                if proba >= 0.70:
                    faces_list.append(name)
                    proba_list.append(proba)
                    count = count + 1

                if name == "Mridulata":
                    if proba >= 0.70:
                        cv2.putText(frame, "WELCOME MRIDULATA!!!", (40, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                if name == "Smrity":
                    if proba >= 0.90:
                        cv2.putText(frame, "WELCOME SMRITY!!!", (40, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                if name == "Saloni":
                    if proba >= 0.90:
                        cv2.putText(frame, "WELCOME SALONI!!!", (40, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                if name == "Sujata":
                    if proba >= 0.90:
                        cv2.putText(frame, "WELCOME SUJATA!!!", (40, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                if name == "Unknown":
                    if proba >= 0.90:
                        unknown_dir = "images/unknown"
                        test = datetime
                        date_string = time.strftime("%Y-%m-%d-%H:%M")

                        unknowns_name = unknown_dir + os.sep + date_string + ".jpg"
                        cv2.imwrite(unknowns_name, frame)
                        unknown_counter += 1

        if count == 20:

            d = defaultdict(list)
            for key, value in zip(faces_list, proba_list):
                d[key].append(value)
            occurence = dict(d)
            thisset = set(occurence)
            for x in thisset:
                occurance_individual = len(occurence[x])
                occurence[x] = sum(item for item in occurence[x])

            a = sum(occurence.values())

            for x in thisset:
                occurence[x] = occurence[x] / a

            attendance = {word for word, prob in occurence.items() if prob >= 0.3}
            # students = max(occurence, key=occurence.get)
            students = list(attendance)

            headers = ['Date', 'Name', 'Time Sign In', 'Time Sign Out']

            def write_csv(data):

                with open('attendance-system.csv', 'a') as outfile:
                    outfile.truncate()
                    file_is_empty = os.stat('attendance-system.csv').st_size == 0
                    writer = csv.writer(outfile, lineterminator='\n', )
                    if file_is_empty:
                        writer.writerow(headers)

                    writer.writerow(data)

            # time.sleep(1)
            current_hour = datetime.now().second
            fps.stop()
            waktu = fps.elapsed()

            if waktu >= 0 and waktu <= 15:
                print('Attendance system Open for sign in')
                for a in students:
                    write_csv([dt_string, a, hr_string, ''])

                records = pd.read_csv('attendance-system.csv')  # Records dictionaryin for notification
                deduped = records.drop_duplicates(['Name'], keep='first')
                deduped = deduped.drop(columns=['Time Sign Out'])
                dictionaryin = deduped.set_index('Name').T.to_dict('list')

            elif waktu >= 30 and waktu <= 45:

                for a in students:
                    write_csv([dt_string, a, '', hr_string])
                print('Attendance system Open for sign out')

                records = pd.read_csv('attendance-system.csv')  # Records dictionaryout for notification
                signed_out = records.loc[records['Time Sign In'].notna()]
                deduped_out = signed_out.drop_duplicates(['Name'], keep='first')
                deduped_out = deduped_out.drop(columns=['Time Sign In'])
                dictionaryout = deduped_out.set_index('Name').T.to_dict('list')
            else:
                print('Attendance system close until Next Course')

            print(dt_string, hr_string, students)

            faces_list.clear()
            proba_list.clear()
            count = 0



        # update the FPS counter
        fps.update()

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


    fps.stop()



    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))




@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


cv2.destroyAllWindows()



@app.route('/images')
def home():
    root_dir = "/home/mridulata/face-recognition-app-tutorial/images/unknown"
    
    image_paths = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in app.config['IMAGE_EXTS']):
                image_paths.append(encode(os.path.join(root,file)))
    return render_template('images.html', paths=image_paths)


@app.route('/cdn/<path:filepath>')
def download_file(filepath):
    dir,filename = os.path.split(decode(filepath))
    return send_from_directory(dir, filename, as_attachment=False)


@app.route('/view')
def view():
    filename = 'attendance-system.csv'
    data = pd.read_csv(filename, header=0)
    myData = list(data.values)
    return render_template('view.html', myData=myData)


@app.route('/showdata')
def showdata():

    if request.method == 'POST':
        results = []

        filename = 'attendance-system.csv'
        user_csv = pd.read_csv(filename, header=0)
        reader = csv.DictReader(user_csv)

        for row in reader:
            results.append(dict(row))

        fieldnames = [key for key in results[0].keys()]

        return render_template('showdata.html', results=results, fieldnames=fieldnames, len=len)


if __name__=="__main__":
    
    app.run(host='0.0.0.0',  port=5000, threaded=True)


