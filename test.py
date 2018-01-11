import cv2
import numpy as np
import time
import NMS


score_path = "score2.png"
start_time = time.clock()
# load score picture
score_png = cv2.imread(score_path, cv2.IMREAD_UNCHANGED)   # score
score = np.array(score_png[:, :, -1], float)
score = np.divide(score, 255)   # normalize
score = np.round(score)
# load template picture
quarter_png = cv2.imread("quarter.png", cv2.IMREAD_UNCHANGED)   # quarter
quarter = np.array(quarter_png[:, :, -1], float)
quarter = np.divide(quarter, 255)   # normalize
quarter = np.round(quarter)

# define overlap
overlap_quarter = np.zeros(score.shape, float)

# fill score array
r, c = score.shape
append_row = np.zeros((150, score.shape[1]))
score = np.vstack((score, append_row))
append_column = np.zeros((score.shape[0], 60))
score = np.hstack((score, append_column))

# convolution
for r_score in range(r):
    for c_score in range(c):
        # sum_quarter = 0
        # for r_quarter in range(quarter.shape[0]):
        #     for c_quarter in range(quarter.shape[1]):
        #         sum_quarter=sum_quarter+quarter[r_quarter][c_quarter]*score[r_score+r_quarter][c_score+c_quarter]
        sum_quarter = np.sum(quarter * score[r_score:(r_score+quarter.shape[0]), c_score:(c_score+quarter.shape[1])])
        overlap_quarter[r_score][c_score] = sum_quarter
        if sum_quarter < 440 * 0.1:
            c_score = c_score + quarter.shape[1]

end_time = time.clock()
print("Running time : %s s" % (end_time - start_time))

