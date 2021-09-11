import cv2 as cv
import time
import argparse

def getFaceBox(net, frame, confidenceThreshold = 0.7):
    frameStreamed = frame.copy()
    frameHeight = frameStreamed.shape[0]
    frameWidth = frameStreamed.shape[1]
    blob = cv.dnn.blobFromImage(frameStreamed, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    boundingBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidenceThreshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            boundingBoxes.append([x1, y1, x2, y2])
            cv.rectangle(frameStreamed, (x1, y1), (x2, y2), (253, 246, 203), int(round(frameHeight/700)), 8)
    return (frameStreamed, boundingBoxes)
parser = argparse.ArgumentParser(description = 'Driver script')
parser.add_argument("-i")
args = parser.parse_args()
faceDetectionProto = "FaceDetectionProto.pbtxt"
faceDetectionModel = "FaceDetectionModel.pb"
ageSegDetectionProto = "AgeSegDetectionProto.prototxt"
ageSegDetectionModel = "AgeSegDetectionModel.caffemodel"
genderDetectionProto = "GenderDetectionProto.prototxt"
genderDetectionModel = "GenderDetectionModel.caffemodel"
modelMeanValues = (70.84946212, 96.937281029, 109.9236400323)
ageList = ['(0-4)', '(6-12)', '(15-19)', '(21-28)', '(31-40)', '(44-56)', '(58-68)', '(70-100)']
genderList = ['Male', 'Female']
faceConvNet = cv.dnn.readNet(faceDetectionModel, faceDetectionProto)
ageConvNet = cv.dnn.readNetFromCaffe(ageSegDetectionProto, ageSegDetectionModel)
genderConvNet = cv.dnn.readNetFromCaffe(genderDetectionProto, genderDetectionModel)
capture = cv.VideoCapture(args.i if args.i else 0)
padding = 20
while cv.waitKey(1) < 0:
    t = time.time()
    isFrame, capturedFrame = capture.read()
    if not isFrame:
        cv.waitKey()
        break
    frameFace, boundingBoxes = getFaceBox(faceConvNet, capturedFrame)
    if not boundingBoxes:
        continue
    for boundingBox in boundingBoxes:
        detectedFace = capturedFrame[max(0, boundingBox[1] - padding) : min(boundingBox[3] + padding, capturedFrame.shape[0] - 1), max(0, boundingBox[0] - padding) : min(boundingBox[2] + padding, capturedFrame.shape[1] - 1)]
        blob = cv.dnn.blobFromImage(detectedFace, 1.0, (227, 227), modelMeanValues, swapRB = False)
        genderConvNet.setInput(blob)
        genderPredictions = genderConvNet.forward()
        detectedGender = genderList[genderPredictions[0].argmax()]
        print("Gender : {}, Confidence = {:.3f}".format(detectedGender, genderPredictions[0].max()))
        ageConvNet.setInput(blob)
        agePredictions = ageConvNet.forward()
        detectedAgeSeg = ageList[agePredictions[0].argmax()]       
        print("Age : {}, Confidence = {:.3f}".format(detectedAgeSeg, agePredictions[0].max()))
        label = "{}, {}".format(detectedGender, detectedAgeSeg)
        if (detectedGender == "Female"):
            cv.putText(frameFace, label, (boundingBox[0] - 5, boundingBox[1] - 10), cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 204, 255), 0, cv.LINE_AA)
        else:
            cv.putText(frameFace, label, (boundingBox[0] - 5, boundingBox[1] - 10), cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 204), 0, cv.LINE_AA)
        cv.imshow("Age&GenderDetection", frameFace)
        name = args.i       
    #print("Time : {:.3f}".format(time.time() - t))

