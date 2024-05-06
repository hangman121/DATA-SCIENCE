import cv2
import numpy as np
###############################
import utils

path = "markedomr.jpg"
width = 700
height = 700
questions = 5
choices = 5
ans = [1, 1, 0, 0, 4]

###############################

# preprocessing

image = cv2.imread(path)
image_resize = cv2.resize(image, (width, height))
imgContours = image_resize.copy()
img_final = image_resize.copy()
image_big = image_resize.copy()
image_grade = image_resize.copy()
image_gr = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
image_blur = cv2.GaussianBlur(image_gr, (5, 5), 1)
image_edges = cv2.Canny(image_blur, 10, 50)

# finding all the contours

contours, hierarchy = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2)

# finding the area if the contours are rectangles

rect_con = utils.rectContour(contours)
biggestContour = utils.get_corner_points(rect_con[0])
grade_contour = utils.get_corner_points(rect_con[1])

# drawing the contours

if biggestContour.size != 0 and grade_contour.size != 0:
    cv2.drawContours(image_big, biggestContour, -1, (255, 0, 0), 10)
    cv2.drawContours(image_grade, grade_contour, -1, (0, 0, 255), 10)

    biggestContour = utils.reorder(biggestContour)
    grade_contour = utils.reorder(grade_contour)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    image_colored = cv2.warpPerspective(image_resize, matrix, (width, height))

    ptg1 = np.float32(grade_contour)
    ptg2 = np.float32([[0, 0], [325, 0], [0, 125], [325, 125]])
    matrixg = cv2.getPerspectiveTransform(ptg1, ptg2)
    image_colored2 = cv2.warpPerspective(image_resize, matrixg, (325, 125))

    # apply threshold

    image_warp_gray = cv2.cvtColor(image_colored, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.threshold(image_warp_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = utils.split_box(img_thresh)

    my_pixel_val = np.zeros((questions, choices))
    countC = 0
    countR = 0

    # getting the non_zero pixel values of each box
    for img in boxes:
        total_pixels = cv2.countNonZero(img)
        my_pixel_val[countR][countC] = total_pixels
        countC += 1
        if countC == choices:
            countR += 1
            countC = 0
    print(my_pixel_val)

    # finding the index value of the markings
    my_index = []
    for x in range(0, questions):
        arr = my_pixel_val[x]
        my_index_val = utils.find_max_index(arr)
        my_index.append(my_index_val)
    print(my_index)

    # grading

    grading = []

    for x in range(0, questions):
        if my_index[x] == ans[x]:
            grading.append(1)
        elif my_index[x] == -1:
            grading.append(-1)
        else:
            grading.append(0)

    # print(grading)
    answers = [0 if x == -1 else x for x in grading]
    score = (sum(answers) / questions) * 100
    print(score)

    #     displaying answer
    img_re = image_colored.copy()
    img_re = utils.showAnswers(img_re, my_index, grading, ans, questions, choices)
    img_raw = np.zeros_like(image_colored)
    img_raw = utils.showAnswers(img_raw, my_index, grading, ans, questions, choices)
    inv_matrix = cv2.getPerspectiveTransform(pt2, pt1)
    img_inv = cv2.warpPerspective(img_raw, inv_matrix, (width, height))

    img_raw_grade = np.zeros_like(image_colored2)
    cv2.putText(img_raw_grade, str(int(score)) + "%", (100, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
    inv_matrixg = cv2.getPerspectiveTransform(ptg2, ptg1)
    img_inv_grade = cv2.warpPerspective(img_raw_grade, inv_matrixg, (width, height))

    img_final = cv2.addWeighted(img_final, 1, img_inv, 1, 0)
    img_final = cv2.addWeighted(img_final, 1, img_inv_grade, 1, 0)

# images = [image_colored, image_colored2]
# stacked = np.hstack(images)

cv2.imshow("image", img_final)

cv2.waitKey(0)
