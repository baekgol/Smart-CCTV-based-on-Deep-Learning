import time
import pickle
import GradCAM as gradcam
import cv2
import face_recognition
import numpy as np
import pygame as pg
import tensorflow as tf
from PIL import Image
from datetime import datetime
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from keras import backend as k
from keras.models import load_model
from sdk.api.message import Message
from sdk.exceptions import CoolsmsException


# 탄력적인 GPU Memory 할당
def growthGPUAlloc(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = False

socketio = SocketIO(app)


# flask 캐시 제거
@app.after_request
def noCache(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge, chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.route('/')
def index():
    print("index")
    return render_template('index.html')


# 자바스크립트 변경 시 Shift-F5로 갱신
# 서버 실행 후 자동 연결
@socketio.on('connect', namespace='/index')
def connect():
    print("Connect")


@socketio.on('disconnect', namespace='/index')
def disconnect():
    print('Disconnect')


# 팝업창 렌더링
@app.route('/popup_alert_1/<cameraNum>/<state>/<isGradcam>')
def popup_alert_1(cameraNum, state, isGradcam):
    cameraNum2 = int(cameraNum) + 1
    cameraNum2 = str(cameraNum2)
    return render_template('popup_alert.html', cameraNum1=cameraNum, cameraNum2=cameraNum2, state=state, isGradcam=isGradcam)


@app.route('/popup_alert_2/<cameraNum>/<state>/<isGradcam>')
def popup_alert_2(cameraNum, state, isGradcam):
    cameraNum2 = int(cameraNum) + 1
    cameraNum2 = str(cameraNum2)
    return render_template('popup_alert.html', cameraNum1=cameraNum, cameraNum2=cameraNum2, state=state, isGradcam=isGradcam)


# CCTV 실시간 영상 송출
@app.route('/video_feed1')
def video_feed1():
    return Response(predict(2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
def video_feed2():
    return Response(predict(3),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 상황 발생 이미지 실시간 저장
@app.route('/state_img1')
def state_img1():
    return Response(stateImgStream(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img2')
def state_img2():
    return Response(stateImgStream(2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img3')
def state_img3():
    return Response(stateImgStream(3),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img4')
def state_img4():
    return Response(stateImgStream(4),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img5')
def state_img5():
    return Response(stateImgStream(5),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img6')
def state_img6():
    return Response(stateImgStream(6),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img7')
def state_img7():
    return Response(stateImgStream(7),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img8')
def state_img8():
    return Response(stateImgStream(8),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img9')
def state_img9():
    return Response(stateImgStream(9),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img10')
def state_img10():
    return Response(stateImgStream(10),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img11')
def state_img11():
    return Response(stateImgStream(11),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img12')
def state_img12():
    return Response(stateImgStream(12),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img13')
def state_img13():
    return Response(stateImgStream(13),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img14')
def state_img14():
    return Response(stateImgStream(14),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img15')
def state_img15():
    return Response(stateImgStream(15),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img16')
def state_img16():
    return Response(stateImgStream(16),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img17')
def state_img17():
    return Response(stateImgStream(17),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img18')
def state_img18():
    return Response(stateImgStream(18),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img19')
def state_img19():
    return Response(stateImgStream(19),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img20')
def state_img20():
    return Response(stateImgStream(20),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img21')
def state_img21():
    return Response(stateImgStream(21),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/state_img22')
def state_img22():
    return Response(stateImgStream(22),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def stateImgStream(imgNum):
    yield (b'--frame\r\n'
       b'Content-Type: image/jpeg\r\n\r\n' + open('static/cam_img/state_img/state_img_%d.jpg' %(imgNum), 'rb').read() + b'\r\n')


# 상황발생유무 전송
def stateResponse(stateValue, state):
    global state_img_count
    with app.app_context():
        if state == 'Fire':
            emit('fireResponse', {'fire_state': stateValue}, broadcast=True, namespace='/index')
        if state == 'TrafficAccident':
            emit('trafficResponse', {'traffic_accident_state': stateValue}, broadcast=True, namespace='/index')
        if state == 'Criminal':
            emit('criminalResponse', {'criminal_state': stateValue}, broadcast=True, namespace='/index')


# 팝업 정보 전송 및 띄우기
def popup(cameraNum, state, isGradcam = False):
    with app.app_context():
        emit('stateWindow', {'cameraNum': cameraNum-1, 'state': state, 'isGradcam': isGradcam}, broadcast=True, namespace='/index')


# 유관기관 문자메세지 상황 전파
def notify(cameraNum, state):
    # set api key, api secret
    api_key = "NCS6X1TDX9Q2YGOO"
    api_secret = "WIRJ52PDVVXLC1N8UWCGVL4BC1KMEQKQ"

    # 4 params(to, from, type, text) are mandatory. must be filled
    params = dict()
    params['to'] = '01012345678'  # Recipients Number '01000000000,01000000001'
    params['from'] = '01012345678'  # Sender number

    # Message
    if state == 'Fire':
        params['type'] = 'sms'  # Message type ( sms, lms, mms, ata )
        params['text'] = 'CCTV %d에서 화재 발생!\n실시간 상황 링크: http://192.168.85.184:8090/?action=stream_%d' % (cameraNum, cameraNum - 2)
    elif state == 'TrafficAccident':
        params['type'] = 'sms'  # Message type ( sms, lms, mms, ata )
        params['text'] = 'CCTV %d에서 교통사고 발생!\n실시간 상황 링크: http://192.168.85.184:8090/?action=stream_%d' % (cameraNum, cameraNum - 2)
    else:
        params['type'] = 'mms'  # Message type ( sms, lms, mms, ata )
        params['text'] = 'CCTV %d에서 범죄자 식별!\n실시간 상황 링크: http://192.168.85.184:8090/?action=stream_%d' % (cameraNum, cameraNum - 2)
        params['image'] = 'static/cam_img/cam%d_criminal.jpg' % (cameraNum - 1)

    cool = Message(api_key, api_secret)
    response = cool.balance()
    print("cash : %s" % response['cash'])

    try:
        response = cool.send(params)
        print("Success Count : %s" % response['success_count'])
        print("Error Count : %s" % response['error_count'])
        print("Group ID : %s" % response['group_id'])

        if "error_list" in response:
            print("Error List : %s" % response['error_list'])

    except CoolsmsException as e:
        print("Error Code : %s" % e.code)
        print("Error Message : %s" % e.msg)


# 경고음
def alarm(state):
    pg.mixer.init()
    if state == 'Fire':
        fire_alarm = pg.mixer.Sound("sound/siren.wav")
        fire_alarm.play()
    if state == 'TrafficAccident':
        traffic_accident_alarm = pg.mixer.Sound("sound/siren.wav")
        traffic_accident_alarm.play()
    else:
        criminal_alarm = pg.mixer.Sound("sound/siren.wav")
        criminal_alarm.play()


# 깜빡임
BLINK_FRAME_INTERVAL = 3
def blink(isFire, isTrafficAccident, isCriminal, isBlink, fireCount, trafficAccident_Contact_Count, trafficAccident_Flip_Count, criminalCount, redCount, originalCount, blinkCount, frame):
    if isBlink is False:    # 20번의 이미지에 대해 경고(빨강)이미지 출력
        frame = cv2.rectangle(frame, (0, 0), (639, 479), color=(0, 0, 0))
        frame = cv2.addWeighted(frame, 0.5, redImg, 0.5, 0.0)
        redCount += 1
        if redCount >= BLINK_FRAME_INTERVAL:
            isBlink = True
            redCount = 0
    elif isBlink is True:   # 20번의 이미지에 대해 경고이미지 출력을 끔
        frame = cv2.rectangle(frame, (0, 0), (639, 479), color=(0, 0, 0))
        originalCount += 1
        if originalCount >= BLINK_FRAME_INTERVAL:
            isBlink = False
            originalCount = 0
            blinkCount += 1
            if blinkCount >= 10:
                if isFire:
                    isFire = False
                    blinkCount = 0
                    fireCount = 0
                if isTrafficAccident:
                    isTrafficAccident = False
                    blinkCount = 0
                    trafficAccident_Contact_Count = 0
                    trafficAccident_Flip_Count = 0
                if isCriminal:
                    isCriminal = False
                    blinkCount = 0
                    criminalCount = 0

    return isFire, isTrafficAccident, isCriminal, isBlink, fireCount, trafficAccident_Contact_Count, trafficAccident_Flip_Count, criminalCount, redCount, originalCount, blinkCount, frame


# 얼굴 감지
def face_detect(data, frame, isFaceDetect):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='cnn')

    # 포착한 사람 수만큼 2차원의 리스트에 저장
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize the list of names for each face detected
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # 각각의 얼굴들을 input image와 비교하기
        # 이 함수는 true/false리스트를 반환(각각의 데이터셋의 이미지들에 대해)
        # 218개의 boolean value 리스트를 반환
        # compare_faces함수는 euclidean거리를 계산해서 임베딩한 값과 모든 데이터들을 비교하는 함수이다.
        # encoding : 포착한 사람 중 한명
        # data['encodings'] : 전체 이미지에 대한 128개의 점들을 모아놓은 2차원 리스트
        # matches : 전체 이미지에 대한 Target이 True인지 False인지 반환
        matches = face_recognition.compare_faces(data['encodings'], encoding)

        name = "Unknown"

        # check to see if we have found a match
        # if True in matches:
        if matches.count(True) >= 150:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b] # True인 인덱스만 리스트에 저장
            counts = {}

            # 각각 사진들이랑 임베딩 값들이랑 비교해가지고 가장 일치하는 수가 많은
            # 폴더랑 그 사진 얼굴이랑 아마 matching되는 것 같음
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)

            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    for name in names:
        if name != 'Unknown':
            isFaceDetect = True
            break

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        if name == 'Unknown':
            continue
        # 초록 rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # 사람이름 쓰기
        y = top - 15 if top - 15 > 15 else top + 15

        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    return frame, isFaceDetect


# 프레임 저장
def saveImage(cameraNum, isGradcam, frame, frame_grad):
    frame = cv2.resize(frame, dsize=(640, 480))
    imgToString = cv2.imencode('.jpg', frame)[1].tostring()

    if cameraNum == 2:
        if isGradcam:
            frame_grad = cv2.resize(frame_grad, dsize=(640, 480))
            cv2.imwrite('static/cam_img/cam1_grad.jpg', frame_grad)
    else:
        if isGradcam:
            frame_grad = cv2.resize(frame_grad, dsize=(640, 480))
            cv2.imwrite('static/cam_img/cam2_grad.jpg', frame_grad)

    return imgToString, frame_grad


# Grad-Cam 시각화
def gradVisualization(frame, frame_grad):
    # for i in range(frame_grad.shape[0]):
    #     for j in range(frame_grad.shape[1]):
    #         if frame_grad[i, j]
    #         print(frame_grad[i][j][:])
    #         print("teststes")
    #         print(frame_grad[i][j])
    #         print("test")
    #         print(frame_grad[i, j])

    # for i in range(3):
    #     for j in frame_grad[i]:
    #         if j[0] >= 255:
    #             cv2.line(frame, (frame_grad[i]))
    frame = cv2.resize(frame, (img_width, img_height))
    frame_grad = cv2.addWeighted(frame, 0.5, frame_grad, 0.5, 0.0)

    return frame_grad


# 상황 판별
def whatState(cameraNum, isFire, isTrafficAccident, isCriminal, isFaceDetect, isGradcam, fireCount, trafficAccident_Contact_Count, trafficAccident_Flip_Count, criminalCount, fire_state, traffic_accident_state, criminal_state, popup_fire, popup_traffic, startTime, endTime, pred_fire, pred_traffic, frame, frameRGB, frame_grad):
    global state_img_count
    if pred_fire[0] * 100 <= 10:
        popup_fire = False
    if pred_traffic[0] * 100 <= 10 and pred_traffic[1] * 100 <= 10:
        popup_traffic = False

    # 화염
    if pred_fire[0] * 100 > 90 and isFire is False:
        fireCount += 1
        if fireCount >= 20:
            fire_state = True
            isGradcam = True
            isFire, isGradcam, fire_state, popup_fire, startTime, endTime, frame_grad = fireState(cameraNum, isFire, isGradcam, fire_state, popup_fire, startTime, endTime, frame, frameRGB)

    # 차량접촉
    if (pred_traffic[0] * 100 > 90 and isTrafficAccident is False):
        trafficAccident_Contact_Count += 1
        if trafficAccident_Contact_Count >= 20:
            traffic_accident_state = True
            isTrafficAccident, traffic_accident_state, popup_traffic, startTime, endTime = trafficAccidentState(
                cameraNum, isTrafficAccident, traffic_accident_state, popup_traffic, startTime, endTime, frame)

    # 차량전복
    if (pred_traffic[1] * 100 > 90 and isTrafficAccident is False):
        trafficAccident_Flip_Count += 1
        if trafficAccident_Flip_Count >= 20:
            traffic_accident_state = True
            isTrafficAccident, traffic_accident_state, popup_traffic, startTime, endTime = trafficAccidentState(cameraNum, isTrafficAccident, traffic_accident_state, popup_traffic, startTime, endTime, frame)
    
    # 범죄자
    if (isFaceDetect is True) and (isCriminal is False):
        criminal_state = True
        isCriminal, criminal_state, startTime, endTime = criminalState(cameraNum, isCriminal, criminal_state, startTime, endTime, frame)
        isFaceDetect = False
    elif (isFaceDetect is True) and (isCriminal is True):
        isFaceDetect = False

    if state_img_count >= 22:
        state_img_count = 0

    return isFire, isTrafficAccident, isCriminal, isFaceDetect, isGradcam, fireCount, trafficAccident_Contact_Count, trafficAccident_Flip_Count, criminalCount, fire_state, traffic_accident_state, criminal_state, popup_fire, popup_traffic, startTime, endTime, frame_grad


# 화재 상황
def fireState(cameraNum, isFire, isGradcam, fire_state, popup_fire, startTime, endTime, frame, frameRGB):
    global state_img, state_img_count
    stateResponse(fire_state, 'Fire')
    if cameraNum == 2:
        frame_grad = gradcam.main(cam1_fire_model, frame, frameRGB, graph1, 'Fire')
    else:
        frame_grad = gradcam.main(cam2_fire_model, frame, frameRGB, graph2, 'Fire')
    # frame_grad = gradVisualization(frame, frame_grad)
    _, frame_grad = saveImage(cameraNum, isGradcam, frame, frame_grad)
    if state_img == False:
        state_img_count += 1
        cv2.imwrite('static/cam_img/state_img/state_img_%d.jpg' % (state_img_count), frame_grad)
        startTime = time.time()
        with app.app_context():
            emit('stateImgCount', {'state_img_count': state_img_count, 'cameraNum': cameraNum, 'state': 'Fire', 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}, broadcast=True, namespace='/index')
        state_img = True
    if popup_fire == False:
        popup(cameraNum, 'Fire', isGradcam)
        popup_fire = True
    if fire_state:
        isFire = True
        fire_state = False
        isGradcam = False
    alarm('Fire')
    #notify(cameraNum, 'Fire')
    return isFire, isGradcam, fire_state, popup_fire, startTime, endTime, frame_grad


# 교통사고 상황
def trafficAccidentState(cameraNum, isTrafficAccident, traffic_accident_state, popup_traffic, startTime, endTime, frame):
    global state_img, state_img_count
    stateResponse(traffic_accident_state, 'TrafficAccident')
    if cameraNum == 2:
        cv2.imwrite('static/cam_img/cam1_traffic.jpg', frame)
    else:
        cv2.imwrite('static/cam_img/cam2_traffic.jpg', frame)
    if state_img == False:
        state_img_count += 1
        cv2.imwrite('static/cam_img/state_img/state_img_%d.jpg' % (state_img_count), frame)
        startTime = time.time()
        with app.app_context():
            emit('stateImgCount', {'state_img_count': state_img_count, 'cameraNum': cameraNum, 'state': 'TrafficAccident', 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}, broadcast=True, namespace='/index')
        state_img = True
    if popup_traffic == False:
        popup(cameraNum, 'TrafficAccident')
        popup_traffic = True
    if traffic_accident_state:
        isTrafficAccident = True
        traffic_accident_state = False
    alarm('TrafficAccident')
    #notify(cameraNum, 'TrafficAccident')
    return isTrafficAccident, traffic_accident_state, popup_traffic, startTime, endTime


# 범죄자 상황
def criminalState(cameraNum, isCriminal, criminal_state, startTime, endTime, frame):
    global state_img, state_img_count
    stateResponse(criminal_state, 'Criminal')
    if cameraNum == 2:
        cv2.imwrite('static/cam_img/cam1_criminal.jpg', frame)
    else:
        cv2.imwrite('static/cam_img/cam2_criminal.jpg', frame)
    if state_img == False:
        state_img_count += 1
        cv2.imwrite('static/cam_img/state_img/state_img_%d.jpg' % (state_img_count), frame)
        startTime = time.time()
        with app.app_context():
            emit('stateImgCount', {'state_img_count': state_img_count, 'cameraNum': cameraNum, 'state': 'Criminal', 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}, broadcast=True, namespace='/index')
        state_img = True
    popup(cameraNum, 'Criminal')
    if criminal_state:
        isCriminal = True
        criminal_state = False
    alarm('Criminal')
    #notify(cameraNum, 'Criminal')
    return isCriminal, criminal_state, startTime, endTime


# 예측
def predict(cameraNum):
    isFire = False
    isTrafficAccident = False
    isCriminal = False
    isFaceDetect = False
    isBlink = False
    isGradcam = False

    redCount = 0
    originalCount = 0
    blinkCount = 0
    fireCount = 0
    trafficAccident_Contact_Count = 0
    trafficAccident_Flip_Count = 0
    criminalCount = 0

    fire_state = False
    traffic_accident_state = False
    criminal_state = False

    popup_fire = False
    popup_traffic = False

    startTime = 0
    endTime = 0

    global state_img

    if cameraNum == 2:
        cam_ip = 'http://192.168.1.48:8090/?action=stream_0'
        print('Camera1 On')
    else:
        cam_ip = 'http://192.168.1.48:8090/?action=stream_1'
        print('Camera2 On')

    cap = cv2.VideoCapture(cam_ip)

    frame_index = 0 # 프레임 인덱스
    skip_frame = 0  # 스킵해야하는 프레임 개수
    diff = 0        # 딜레이

    while (cap.isOpened()):
        ret, frame = cap.read()

        # 30 FPS
        if frame_index < 30:
            isSkip = False

            # 정해진 프레임 개수가 되면 영상 출력
            if frame_index % 4 == 0:
                st_time = time.time()

                if ret:
                    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_grad = frame

                    img = Image.fromarray(frameRGB)
                    img = img.resize((224, 224), Image.NEAREST)
                    x = np.asarray(img, dtype=k.floatx())
                    x *= (1. / 255)

                    if cameraNum == 2:
                        with graph1.as_default():
                            pred_fire = cam1_fire_model.predict(np.expand_dims(x, axis=0))[0]
                        with graph2.as_default():
                            pred_traffic = cam1_traffic_model.predict(np.expand_dims(x, axis=0))[0]
                        print(
                            'CCTV 2 - 화염: %.3f%%, 연기: %.3f%%, 일반(화염): %.3f%%     ||     차량접촉: %.3f%%, 차량전복: %.3f%%, 일반(교통사고): %.3f%%' % (
                            pred_fire[0] * 100, pred_fire[2] * 100, pred_fire[1] * 100, pred_traffic[0] * 100, pred_traffic[1] * 100,
                            pred_traffic[2] * 100))
                    else:
                        with graph3.as_default():
                            pred_fire = cam2_fire_model.predict(np.expand_dims(x, axis=0))[0]
                        with graph4.as_default():
                            pred_traffic = cam2_traffic_model.predict(np.expand_dims(x, axis=0))[0]
                        print(
                            'CCTV 3 - 화염: %.3f%%, 연기: %.3f%%, 일반(화염): %.3f%%     ||     차량접촉: %.3f%%, 차량전복: %.3f%%, 일반(교통사고): %.3f%%' % (
                            pred_fire[0] * 100, pred_fire[2] * 100, pred_fire[1] * 100, pred_traffic[0] * 100, pred_traffic[1] * 100,
                            pred_traffic[2] * 100))

                    frame, isFaceDetect = face_detect(face_model, frame, isFaceDetect)

                    isFire, isTrafficAccident, isCriminal, isFaceDetect, isGradcam, fireCount, trafficAccident_Contact_Count, trafficAccident_Flip_Count, criminalCount, fire_state, traffic_accident_state, criminal_state, popup_fire, popup_traffic, startTime, endTime, frame_grad = whatState(
                        cameraNum, isFire, isTrafficAccident, isCriminal, isFaceDetect, isGradcam, fireCount, trafficAccident_Contact_Count, trafficAccident_Flip_Count,
                        criminalCount, fire_state, traffic_accident_state, criminal_state, popup_fire, popup_traffic, startTime, endTime, pred_fire, pred_traffic, frame, frameRGB, frame_grad)

                    imgToString, _ = saveImage(cameraNum, isGradcam, frame, frame_grad)

                    if isFire or isTrafficAccident or isCriminal:
                        isFire, isTrafficAccident, isCriminal, isBlink, fireCount, trafficAccident_Contact_Count, trafficAccident_Flip_Count, criminalCount, redCount, originalCount, blinkCount, frame = blink(
                            isFire, isTrafficAccident, isCriminal, isBlink, fireCount, trafficAccident_Contact_Count, trafficAccident_Flip_Count, criminalCount,
                            redCount, originalCount, blinkCount, frame)
                        imgToString, _ = saveImage(cameraNum, isGradcam, frame, frame_grad)

                    if cameraNum == 2:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + imgToString + b'\r\n')
                    else:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + imgToString + b'\r\n')
                else:
                    break

                with app.app_context():
                    emit('state_img_modal_refresh', broadcast=True, namespace='/index')
                    emit('state_img_data_refresh', broadcast=True, namespace='/index')
                ed_time = time.time() - st_time
                diff += ed_time
                endTime = time.time() - startTime
                if endTime >= 10:
                    state_img = False

        # 30 프레임이 지나갔을 경우
        # 초당 30 프레임과의 딜레이를 동기화
        else:
            # 처음일 경우
            if not isSkip:
                diff = diff - 1
                skip_frame = diff * 30  # 이 값만큼 프레임 스킵
                isSkip = True
                # print("skip_frame : {}".format(skip_frame))

            # print("diff : {}".format(diff))
            # print("skip_frame : {}".format(skip_frame))

            skip_frame -= 1

            # 정해진 프레임을 전부 스킵했을 경우 초기화
            if skip_frame <= 0:
                diff = 0
                frame_index = -1

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    print("end")


if __name__ == "__main__":
    k.set_session(growthGPUAlloc())

    cam1_fire_model = load_model('model/09_19(b8_e200).h5')
    cam2_fire_model = load_model('model/09_19(b8_e200).h5')
    cam1_traffic_model = load_model('model/09_18(b8_e200_steps_per_epoch1000).h5')
    cam2_traffic_model = load_model('model/09_18(b8_e200_steps_per_epoch1000).h5')
    graph1 = tf.get_default_graph()
    graph2 = tf.get_default_graph()
    graph3 = tf.get_default_graph()
    graph4 = tf.get_default_graph()
    face_model = pickle.loads(open('model/model_face.pickle', 'rb').read(), encoding='latin1')

    img_width = 224
    img_height = 224
    redImg = np.zeros((480, 640, 3), np.uint8)
    redImg[:] = (0, 0, 255)
    state_img = False
    state_img_count = 0

    socketio.run(app)

    # release memory
    k.clear_session()
