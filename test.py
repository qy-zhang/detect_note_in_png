# !/usr/bin/python
# -*- coding: UTF-8 -*-

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
half_png = cv2.imread("half.png", cv2.IMREAD_UNCHANGED)  # half
half = np.array(half_png[:, :, -1], float)
half = np.divide(half, 255)  # normalize
half = np.round(half)
whole_png = cv2.imread("whole.png", cv2.IMREAD_UNCHANGED)  # whole
whole = np.array(whole_png[:, :, -1], float)
whole = np.divide(whole, 255)  # normalize
whole = np.round(whole)
g_clef_png = cv2.imread("g_clef.png", cv2.IMREAD_UNCHANGED)  # half
g_clef = np.array(g_clef_png[:, :, -1], float)
g_clef = np.divide(g_clef, 255)  # normalize
g_clef = np.round(g_clef)
f_clef_png = cv2.imread("f_clef.png", cv2.IMREAD_UNCHANGED)  # half
f_clef = np.array(f_clef_png[:, :, -1], float)
f_clef = np.divide(f_clef, 255)  # normalize
f_clef = np.round(f_clef)

# define overlap
overlap_quarter = np.zeros(score.shape, float)
overlap_half = np.zeros(score.shape, float)
overlap_whole = np.zeros(score.shape, float)
overlap_g_clef = np.zeros(score.shape, float)
overlap_f_clef = np.zeros(score.shape, float)

# fill score array
r, c = score.shape
append_row = np.zeros((150, score.shape[1]))
score = np.vstack((score, append_row))
append_column = np.zeros((score.shape[0], 60))
score = np.hstack((score, append_column))

# convolution
r_score, c_score = 0, 0
while r_score < r:
    while c_score < c:
        sum_quarter = np.sum(
            quarter * score[r_score:(r_score + quarter.shape[0]), c_score:(c_score + quarter.shape[1])])
        overlap_quarter[r_score][c_score] = sum_quarter
        sum_half = np.sum(half * score[r_score:(r_score + half.shape[0]), c_score:(c_score + half.shape[1])])
        overlap_half[r_score][c_score] = sum_half
        sum_whole = np.sum(whole * score[r_score:(r_score + whole.shape[0]), c_score:(c_score + whole.shape[1])])
        overlap_whole[r_score][c_score] = sum_whole

        sum_g_clef = np.sum(
            g_clef * score[r_score:(r_score + g_clef.shape[0]), c_score:(c_score + g_clef.shape[1])])
        overlap_g_clef[r_score][c_score] = sum_g_clef
        sum_f_clef = np.sum(
            f_clef * score[r_score:(r_score + f_clef.shape[0]), c_score:(c_score + f_clef.shape[1])])
        overlap_f_clef[r_score][c_score] = sum_f_clef
        c_score = c_score + 1
        if sum_quarter < 50 and sum_half < 50 and sum_whole < 50:
            c_score = c_score + quarter.shape[1]

    r_score = r_score + 1
    c_score = 0


#
# for r_score in range(r):
#     for c_score in range(c):
#         # sum_quarter = 0
#         # for r_quarter in range(quarter.shape[0]):
#         #     for c_quarter in range(quarter.shape[1]):
#         #         sum_quarter=sum_quarter+quarter[r_quarter][c_quarter]*score[r_score+r_quarter][c_score+c_quarter]
#         sum_quarter = np.sum(
#             quarter * score[r_score:(r_score + quarter.shape[0]), c_score:(c_score + quarter.shape[1])])
#         overlap_quarter[r_score][c_score] = sum_quarter
#         sum_half = np.sum(half * score[r_score:(r_score + half.shape[0]), c_score:(c_score + half.shape[1])])
#         overlap_half[r_score][c_score] = sum_half
#         sum_whole = np.sum(whole * score[r_score:(r_score + whole.shape[0]), c_score:(c_score + whole.shape[1])])
#         overlap_whole[r_score][c_score] = sum_whole
#         if sum_quarter < 50 and sum_half < 50 and sum_whole < 50:
#             c_score = c_score + quarter.shape[1]
#         # sum_g_clef = np.sum(
#         #     g_clef * score[r_score:(r_score + g_clef.shape[0]), c_score:(c_score + g_clef.shape[1])])
#         # overlap_g_clef[r_score][c_score] = sum_g_clef
#         # sum_f_clef = np.sum(
#         #     f_clef * score[r_score:(r_score + f_clef.shape[0]), c_score:(c_score + f_clef.shape[1])])
#         # overlap_f_clef[r_score][c_score] = sum_f_clef
#
# for r_score in range(r):
#     for c_score in range(c):
#         sum_g_clef = np.sum(
#             g_clef * score[r_score:(r_score + g_clef.shape[0]), c_score:(c_score + g_clef.shape[1])])
#         overlap_g_clef[r_score][c_score] = sum_g_clef
#         sum_f_clef = np.sum(
#             f_clef * score[r_score:(r_score + f_clef.shape[0]), c_score:(c_score + f_clef.shape[1])])
#         overlap_f_clef[r_score][c_score] = sum_f_clef


end_time = time.clock()
print("Running time : %s s" % (end_time - start_time))

