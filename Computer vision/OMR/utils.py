import cv2
import numpy as np


def rectContour(contour):
    reccon = []
    for i in contour:
        area = cv2.contourArea(i)
        # print(area)
        if area > 800:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                reccon.append(i)
    reccon = sorted(reccon, key=cv2.contourArea, reverse=True)

    return reccon


def get_corner_points(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx


def reorder(my_points):
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), np.int32)
    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, 1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]

    return my_points_new


def split_box(img):
    rows = np.vsplit(img, 5)
    boxs = []
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxs.append(box)

    return boxs


def showAnswers(img, index, grading, ans, questions, choices):
    secw = int(img.shape[1] / questions)
    secH = int(img.shape[0] / choices)

    for x in range(0, questions):
        my_ans = index[x]
        cx = (my_ans * secw) + secw // 2
        cy = (x * secH) + secH // 2

        if grading[x] == 1:
            my_color = (0, 255, 0)
        elif grading[x] == -1:
            my_color = (255, 0, 0)
            correct_ans = ans[x]
            cv2.circle(img, ((correct_ans * secw) + secw // 2, (x * secH) + secH // 2), 30, (255, 0, 0), cv2.FILLED)

        else:
            my_color = (0, 0, 255)
            correct_ans = ans[x]
            cv2.circle(img, ((correct_ans * secw) + secw // 2, (x * secH) + secH // 2), 30, (0, 255, 0), cv2.FILLED)

        cv2.circle(img, (cx, cy), 40, my_color, cv2.FILLED)

    return img


def find_max_index(arr):

    if np.max(arr) > 4000:

        max_indices = np.where(arr == np.max(arr))[0]
        return max_indices[0]
    else:
        return -1
