import imutils
import cv2
import numpy as np

# from scipy import stats
NUM_FRAMES = 5

# cap = cv2.VideoCapture("/home/ayman/Desktop/video.mp4")
cap = cv2.VideoCapture(0)
frame = cap.read()[1]
# frame = im.resize(frame, width=500)
median_frame = np.zeros(cap.read()[1].shape[:2], np.uint8)
current = np.zeros(frame.shape, np.uint8)
current_1 = np.zeros(frame.shape, np.uint8)
current_2 = np.zeros(frame.shape, np.uint8)
moving_mask = np.zeros(frame.shape[:2], np.bool_)
frames = NUM_FRAMES * [np.zeros(frame.shape[:2], dtype=np.uint8)]
lst_contour = None
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


def draw_max_contour(binary_image):
    _, contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(binary_image.shape)
    if len(contours) != 0:
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour)
        hull = check_contour(hull)
        cv2.drawContours(result, [hull], 0, (1, 0, 0), cv2.FILLED)
    elif lst_contour is not None:
        cv2.drawContours(result, [lst_contour], 0, (255, 255, 255), cv2.FILLED)
    return result


def get_color_mask(frame):
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    # frame = cv2.flip(frame, 1)
    frame = cv2.GaussianBlur(frame, (15, 15), cv2.BORDER_DEFAULT)
    imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
    # frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, (11, 11))
    return skinRegion.astype(np.bool_)


def check_contour(contour, min_area=15000, r=1):
    global lst_contour
    x, y, w, h = cv2.boundingRect(contour)
    ratio = 1.0 * w / h
    if (lst_contour is not None) and ((cv2.contourArea(contour) < min_area) or (ratio > r)):
        contour = lst_contour
    lst_contour = contour
    return contour


# def draw_max_contour(binary_image):
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     result = np.zeros(binary_image.shape)
#     if len(contours) != 0:
#         max_contour = max(contours, key=cv2.contourArea)
#         max_contour = check_contour(max_contour)
#         cv2.drawContours(result, [max_contour], 0, (255, 255, 255), cv2.FILLED)
#     elif lst_contour is not None:
#         cv2.drawContours(result, [lst_contour], 0, (255, 255, 255), cv2.FILLED)
#     return result


i = 0
# frames = np.array(NUM_FRAMES * [median_frame], np.uint8)
while True:
    ret, original_frame = cap.read()
    if not ret:
        break
    o_frame = original_frame.copy()
    # frame = im.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 5)
    # gray = cv2.erode(gray, (20, 20), iterations=20)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    frames[i] = gray
    i = (i + 1) % NUM_FRAMES
    result = cv2.absdiff(gray, median_frame)
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    frame = original_frame.copy()
    # remove_face(gray, frame)
    # yCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # current = yCrCb = cv2.GaussianBlur(yCrCb, (5, 5), 0)
    # result = np.zeros(yCrCb.shape[:2], np.uint8)
    # color_mask = (54 <= yCrCb[:, :, 0]) & (yCrCb[:, :, 0] <= 163) \
    #              & (131 <= yCrCb[:, :, 1]) & (yCrCb[:, :, 1] <= 157) \
    #              & (110 <= yCrCb[:, :, 2]) & (yCrCb[:, :, 2] <= 135)

    # result = abs(gray - median_frame)

    # moving_mask_result = moving_mask_check.astype(np.uint8)
    # moving_mask_result = cv2.erode(moving_mask_result, np.ones((3, 3)), iterations=5)
    # moving_mask_result = cv2.dilate(moving_mask_result, np.ones((3, 3)), iterations=5)
    # if moving_mask_result.sum() > 15000:
    #     moving_mask = moving_mask_check
    # moving_mask = get_moving_mask(0, 5, 10) & get_moving_mask(1, 5, 10) & get_moving_mask(2, 5, 10)
    # mask = moving_mask & color_mask
    # result[mask] = 255
    # result = abs(gray - median_frame)
    # result = cv2.erode(result, np.ones((3, 3)), iterations=2)
    # result = cv2.dilate(result, np.ones((3, 3)), iterations=2)
    # result = cv2.dilate(result, np.ones((3, 3)), iterations=10)
    # result = cv2.erode(result, np.ones((3, 3)), iterations=10)
    # contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    # x_max = y_max = w_max = h_max = 0
    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     if w * h > w_max * h_max:
    #         w_max = w
    #         h_max = h
    #         x_max = x
    #         y_max = y
    # cv2.rectangle(original_frame, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 0), 2)
    # cv2.drawContours(original_frame, contours, -1, (255, 0, 0))
    result[result < 5] = 0
    result[result >= 5] = 255

    result = cv2.erode(result, np.ones((3, 3), dtype='uint8'), iterations=2)
    result = cv2.dilate(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 8)), iterations=5)
    result = cv2.erode(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), iterations=5)

    # result = cv2.erode(result, np.ones((3, 3)), iterations=5)
    # result = cv2.dilate(result, np.ones((3, 3)), iterations=50)
    # result = cv2.erode(result, np.ones((3, 3)), iterations=45)
    result = draw_max_contour(result)
    result = result.astype(np.bool_)
    # result = np.zeros(result.shape, dtype=np.bool_)
    # result[y:y + h, x:x + w] = 1
    #
    img = np.zeros(o_frame.shape, dtype=np.uint8)
    # img[result] = gray[result]
    skin_mask = get_color_mask(o_frame)
    mask = result & skin_mask
    # mask = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=5)
    mask = mask.astype(np.bool_)
    img[mask] = o_frame[mask]
    # cv2.imshow("image", hello.astype(np.uint8) * 255)
    # cv2.imshow("Face", frame)
    # cv2.imshow("gray", gray)
    cv2.imshow("img", img)
    cv2.imshow("gray", gray)
    cv2.imshow("median", median_frame)
    # current_2 = current_1
    # current_1 = yCrCb
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()
