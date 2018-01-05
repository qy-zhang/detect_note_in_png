import cv2
import numpy as np
import time
import NMS

start_time = time.clock()
# load picture
score_png = cv2.imread("score2.png", cv2.IMREAD_UNCHANGED)   # score
score = np.array(score_png[:, :, -1], float)
score = np.divide(score, 255)   # normalize
score = np.round(score)
quarter_png = cv2.imread("quarter.png", cv2.IMREAD_UNCHANGED)   # quarter
quarter = np.array(quarter_png[:, :, -1], float)
quarter = np.divide(quarter, 255)   # normalize
quarter = np.round(quarter)
half_png = cv2.imread("half.png", cv2.IMREAD_UNCHANGED)     # half
half = np.array(half_png[:, :, -1], float)
half = np.divide(half, 255)     # normalize
half = np.round(half)
whole_png = cv2.imread("whole.png", cv2.IMREAD_UNCHANGED)   # whole
whole = np.array(whole_png[:, :, -1], float)
whole = np.divide(whole, 255)   # normalize
whole = np.round(whole)

# define overlap
overlap_quarter = np.zeros(score.shape, float)
overlap_half = np.zeros(score.shape, float)
overlap_whole = np.zeros(score.shape, float)

# fill score array
r, c = score.shape
append_row = np.zeros((50, score.shape[1]))
score = np.vstack((score, append_row))
append_column = np.zeros((score.shape[0], 50))
score = np.hstack((score, append_column))

# convolution
for r_score in range(r):
    for c_score in range(c):
        # sum_quarter = 0
        # for r_quarter in range(quarter.shape[0]):
        #     for c_quarter in range(quarter.shape[1]):
        #         sum_quarter = sum_quarter + quarter[r_quarter][c_quarter]*score[r_score+r_quarter][c_score+c_quarter]
        sum_quarter = np.sum(quarter * score[r_score:(r_score+quarter.shape[0]), c_score:(c_score+quarter.shape[1])])
        sum_half = np.sum(half * score[r_score:(r_score+half.shape[0]), c_score:(c_score+half.shape[1])])
        sum_whole = np.sum(whole * score[r_score:(r_score+whole.shape[0]), c_score:(c_score+whole.shape[1])])
        overlap_quarter[r_score][c_score] = sum_quarter
        overlap_half[r_score][c_score] = sum_half
        overlap_whole[r_score][c_score] = sum_whole
    # print('process line %s' % r_score)
end_time = time.clock()
print("Running time : %s s" % (end_time - start_time))

# filter result
quarter_thresh = 433    # max_overlap_quarter = 440
quarter_result = np.where(overlap_quarter > quarter_thresh)
half_thresh = 288   # max_overlap_half = 295
half_result = np.where(overlap_half > half_thresh)

# save result
np.savetxt("quarter_result.txt", quarter_result, "%d")
np.savetxt("half_result.txt", half_result, "%d")
# show detect result in picture
for i in range(len(quarter_result[0])):
    cv2.rectangle(score_png, (quarter_result[1][i], quarter_result[0][i]),
                  (quarter_result[1][i]+quarter.shape[1], quarter_result[0][i]+quarter.shape[0]), (0, 0, 255, 255), 1)
    cv2.imwrite("score2_new.png", score_png)
for i in range(len(half_result[0])):
    cv2.rectangle(score_png, (half_result[1][i], half_result[0][i]),
                  (half_result[1][i]+half.shape[1], half_result[0][i]+half.shape[0]), (0, 255, 0, 255), 1)
    cv2.imwrite("score2_new.png", score_png)
