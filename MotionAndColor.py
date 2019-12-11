import imutils
import cv2
import numpy as np

# from scipy import stats
# NUM_FRAMES = 5

# cap = cv2.VideoCapture("/home/ayman/Desktop/video.mp4")
cap = cv2.VideoCapture(0)
frame = cap.read()[1]
# frame = im.resize(frame, width=500)
# median_frame = np.zeros(cap.read()[1].shape[:2], np.uint8)
current = np.zeros(frame.shape, np.uint8)
current_1 = np.zeros(frame.shape, np.uint8)
current_2 = np.zeros(frame.shape, np.uint8)
moving_mask = np.zeros(frame.shape[:2], np.bool_)
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def get_moving_mask(channel, th1, th2):
    moving_mask = np.array(abs(current[:, :, channel] - current_1[:, :, channel]) > th1) & (
            abs(current[:, :, channel] - current_2[:, :, channel]) > th2)
    moving_mask_result = moving_mask.astype(np.uint8)
    moving_mask_result = cv2.erode(moving_mask_result, np.ones((3, 3)), iterations=2)
    moving_mask_result = cv2.dilate(moving_mask_result, np.ones((3, 3)), iterations=30)
    moving_mask_result = cv2.erode(moving_mask_result, np.ones((3, 3)), iterations=30)
    return moving_mask_result.astype(np.bool_)


def remove_face(gray, image):
    faces_rects = haar_cascade_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)


# i = 0
# frames = np.array(NUM_FRAMES * [median_frame], np.uint8)
while True:
    ret, original_frame = cap.read()
    if not ret:
        break
    # frame = im.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 5)
    # gray = cv2.erode(gray, (20, 20), iterations=20)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # frames[i] = gray
    # i = (i + 1) % NUM_FRAMES
    # median_frame = np.median(frames, axis=0).astype(np.uint8)
    frame = original_frame.copy()
    remove_face(gray, frame)
    yCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    current = yCrCb = cv2.GaussianBlur(yCrCb, (5, 5), 0)
    result = np.zeros(yCrCb.shape[:2], np.uint8)
    color_mask = (54 <= yCrCb[:, :, 0]) & (yCrCb[:, :, 0] <= 163) \
                 & (131 <= yCrCb[:, :, 1]) & (yCrCb[:, :, 1] <= 157) \
                 & (110 <= yCrCb[:, :, 2]) & (yCrCb[:, :, 2] <= 135)

    # result = abs(gray - median_frame)

    # moving_mask_result = moving_mask_check.astype(np.uint8)
    # moving_mask_result = cv2.erode(moving_mask_result, np.ones((3, 3)), iterations=5)
    # moving_mask_result = cv2.dilate(moving_mask_result, np.ones((3, 3)), iterations=5)
    # if moving_mask_result.sum() > 15000:
    #     moving_mask = moving_mask_check
    moving_mask = get_moving_mask(0, 5, 10) & get_moving_mask(1, 5, 10) & get_moving_mask(2, 5, 10)
    mask = moving_mask & color_mask
    result[mask] = 255
    # result = abs(gray - median_frame)
    result = cv2.erode(result, np.ones((3, 3)), iterations=2)
    result = cv2.dilate(result, np.ones((3, 3)), iterations=2)
    # result = cv2.dilate(result, np.ones((3, 3)), iterations=10)
    # result = cv2.erode(result, np.ones((3, 3)), iterations=10)
    contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    x_max = y_max = w_max = h_max = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > w_max * h_max:
            w_max = w
            h_max = h
            x_max = x
            y_max = y
    cv2.rectangle(original_frame, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 0), 2)
    cv2.drawContours(original_frame, contours, -1, (255, 0, 0))
    cv2.imshow("image", original_frame)
    # cv2.imshow("Face", frame)
    # cv2.imshow("gray", gray)
    # cv2.imshow("mean", median_frame)
    current_2 = current_1
    current_1 = yCrCb
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()
