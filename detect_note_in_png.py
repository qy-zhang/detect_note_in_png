# !/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import time
import NMS


score_path = "SkateRock.png"
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
half_png = cv2.imread("half.png", cv2.IMREAD_UNCHANGED)     # half
half = np.array(half_png[:, :, -1], float)
half = np.divide(half, 255)     # normalize
half = np.round(half)
whole_png = cv2.imread("whole.png", cv2.IMREAD_UNCHANGED)   # whole
whole = np.array(whole_png[:, :, -1], float)
whole = np.divide(whole, 255)   # normalize
whole = np.round(whole)
g_clef_png = cv2.imread("g_clef.png", cv2.IMREAD_UNCHANGED)     # half
g_clef = np.array(g_clef_png[:, :, -1], float)
g_clef = np.divide(g_clef, 255)     # normalize
g_clef = np.round(g_clef)
f_clef_png = cv2.imread("f_clef.png", cv2.IMREAD_UNCHANGED)     # half
f_clef = np.array(f_clef_png[:, :, -1], float)
f_clef = np.divide(f_clef, 255)     # normalize
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
        sum_quarter = np.sum(quarter*score[r_score:(r_score + quarter.shape[0]), c_score:(c_score + quarter.shape[1])])
        overlap_quarter[r_score][c_score] = sum_quarter
        sum_half = np.sum(half * score[r_score:(r_score+half.shape[0]), c_score:(c_score+half.shape[1])])
        overlap_half[r_score][c_score] = sum_half
        sum_whole = np.sum(whole * score[r_score:(r_score+whole.shape[0]), c_score:(c_score+whole.shape[1])])
        overlap_whole[r_score][c_score] = sum_whole
        if sum_quarter < 20 and sum_half < 20 and sum_whole < 20:
            c_score = c_score + quarter.shape[1]
        c_score = c_score + 1
    c_score = 0
    r_score = r_score + 1

r_score, c_score = 0, 0
find_clef = 0
while r_score < r:
    while c_score < c:
        sum_g_clef = np.sum(g_clef * score[r_score:(r_score+g_clef.shape[0]), c_score:(c_score+g_clef.shape[1])])
        overlap_g_clef[r_score][c_score] = sum_g_clef
        sum_f_clef = np.sum(f_clef * score[r_score:(r_score + f_clef.shape[0]), c_score:(c_score+f_clef.shape[1])])
        overlap_f_clef[r_score][c_score] = sum_f_clef
        if sum_g_clef < 50 and sum_f_clef < 50:
            c_score = c_score + g_clef.shape[1]
        if sum_g_clef >= 2010 or sum_f_clef >= 910:
            find_clef = 1
            break
        c_score = c_score + 1
    c_score = 0
    r_score = r_score + 1
    if find_clef == 1:
        r_score = r_score + 200
        find_clef = 0

# for r_score in range(r):
#     for c_score in range(c):
#         # sum_quarter = 0
#         # for r_quarter in range(quarter.shape[0]):
#         #     for c_quarter in range(quarter.shape[1]):
#         #         sum_quarter=sum_quarter+quarter[r_quarter][c_quarter]*score[r_score+r_quarter][c_score+c_quarter]
#         sum_quarter = np.sum(quarter * score[r_score:(r_score+quarter.shape[0]), c_score:(c_score+quarter.shape[1])])
#         sum_half = np.sum(half * score[r_score:(r_score+half.shape[0]), c_score:(c_score+half.shape[1])])
#         sum_whole = np.sum(whole * score[r_score:(r_score+whole.shape[0]), c_score:(c_score+whole.shape[1])])
#         sum_g_clef = np.sum(g_clef * score[r_score:(r_score+g_clef.shape[0]), c_score:(c_score+g_clef.shape[1])])
#         sum_f_clef = np.sum(f_clef * score[r_score:(r_score + f_clef.shape[0]), c_score:(c_score+f_clef.shape[1])])
#         overlap_quarter[r_score][c_score] = sum_quarter
#         overlap_half[r_score][c_score] = sum_half
#         overlap_whole[r_score][c_score] = sum_whole
#         overlap_g_clef[r_score][c_score] = sum_g_clef
#         overlap_f_clef[r_score][c_score] = sum_f_clef

end_time = time.clock()
print("Running time : %s s" % (end_time - start_time))

# filter result
# add thresh
quarter_thresh = 430    # max_overlap_quarter = 440
quarter_result = np.where(overlap_quarter > quarter_thresh)
# save only one result at a certain area
box = np.zeros((len(quarter_result[0]), 5), int)
for i in range(len(quarter_result[0])):
    box[i][0] = quarter_result[0][i]
    box[i][1] = quarter_result[1][i]
    box[i][2] = quarter_result[0][i] + quarter.shape[0]
    box[i][3] = quarter_result[1][i] + quarter.shape[1]
    box[i][4] = overlap_quarter[quarter_result[0][i]][quarter_result[1][i]]
quarter_result_nms = NMS.non_max_suppression_fast(box, 0.6)
# add thresh
half_thresh = 275   # max_overlap_half = 295
half_result = np.where(overlap_half > half_thresh)
# save only one result at a certain area
box = np.zeros((len(half_result[0]), 5), int)
for i in range(len(half_result[0])):
    box[i][0] = half_result[0][i]
    box[i][1] = half_result[1][i]
    box[i][2] = half_result[0][i] + half.shape[0]
    box[i][3] = half_result[1][i] + half.shape[1]
    box[i][4] = overlap_half[half_result[0][i]][half_result[1][i]]
half_result_nms = NMS.non_max_suppression_fast(box, 0.6)
# add thresh
whole_thresh = 478    # max_overlap_whole = 485
whole_result = np.where(overlap_whole > whole_thresh)
# save only one result at a certain area
box = np.zeros((len(whole_result[0]), 5), int)
for i in range(len(whole_result[0])):
    box[i][0] = whole_result[0][i]
    box[i][1] = whole_result[1][i]
    box[i][2] = whole_result[0][i] + whole.shape[0]
    box[i][3] = whole_result[1][i] + whole.shape[1]
    box[i][4] = overlap_whole[whole_result[0][i]][whole_result[1][i]]
whole_result_nms = NMS.non_max_suppression_fast(box, 0.6)
# add thresh
g_clef_thresh = 2000   # max_overlap_g_clef = 2283
g_clef_result = np.where(overlap_g_clef > g_clef_thresh)
# save only one result at a certain area
box = np.zeros((len(g_clef_result[0]), 5), int)
for i in range(len(g_clef_result[0])):
    box[i][0] = g_clef_result[0][i]
    box[i][1] = g_clef_result[1][i]
    box[i][2] = g_clef_result[0][i] + g_clef.shape[0]
    box[i][3] = g_clef_result[1][i] + g_clef.shape[1]
    box[i][4] = overlap_g_clef[g_clef_result[0][i]][g_clef_result[1][i]]
g_clef_result_nms = NMS.non_max_suppression_fast(box, 0.6)
# add thresh
f_clef_thresh = 900   # max_overlap_f_clef = 1007
f_clef_result = np.where(overlap_f_clef > f_clef_thresh)
# save only one result at a certain area
box = np.zeros((len(f_clef_result[0]), 5), int)
for i in range(len(f_clef_result[0])):
    box[i][0] = f_clef_result[0][i]
    box[i][1] = f_clef_result[1][i]
    box[i][2] = f_clef_result[0][i] + f_clef.shape[0]
    box[i][3] = f_clef_result[1][i] + f_clef.shape[1]
    box[i][4] = overlap_f_clef[f_clef_result[0][i]][f_clef_result[1][i]]
f_clef_result_nms = NMS.non_max_suppression_fast(box, 0.6)

# show detect result in picture
new_name = score_path[:-4] + "_new.png"
for i in range(len(quarter_result_nms)):
    cv2.rectangle(score_png, (quarter_result_nms[i][1], quarter_result_nms[i][0]),
                  (quarter_result_nms[i][3], quarter_result_nms[i][2]), (0, 0, 255, 255), 1)
    cv2.imwrite(new_name, score_png)
for i in range(len(half_result_nms)):
    cv2.rectangle(score_png, (half_result_nms[i][1], half_result_nms[i][0]),
                  (half_result_nms[i][3], half_result_nms[i][2]), (0, 255, 0, 255), 1)
    cv2.imwrite(new_name, score_png)
for i in range(len(whole_result_nms)):
    cv2.rectangle(score_png, (whole_result_nms[i][1], whole_result_nms[i][0]),
                  (whole_result_nms[i][3], whole_result_nms[i][2]), (255, 0, 0, 255), 1)
    cv2.imwrite(new_name, score_png)
for i in range(len(g_clef_result_nms)):
    cv2.rectangle(score_png, (g_clef_result_nms[i][1], g_clef_result_nms[i][0]),
                  (g_clef_result_nms[i][3], g_clef_result_nms[i][2]), (0, 255, 0, 255), 1)
    cv2.imwrite(new_name, score_png)
for i in range(len(f_clef_result_nms)):
    cv2.rectangle(score_png, (f_clef_result_nms[i][1], f_clef_result_nms[i][0]),
                  (f_clef_result_nms[i][3], f_clef_result_nms[i][2]), (0, 255, 0, 255), 1)
    cv2.imwrite(new_name, score_png)

# save result
np.savetxt("quarter_result.txt", quarter_result_nms, "%d")
np.savetxt("half_result.txt", half_result_nms, "%d")
np.savetxt("whole_result.txt", whole_result_nms, "%d")
np.savetxt("g_clef_result.txt", g_clef_result_nms, "%d")
np.savetxt("f_clef_result.txt", f_clef_result_nms, "%d")

note_all = np.zeros((0, 5), int)
if len(quarter_result_nms) != 0:
    note_all = np.vstack((note_all, quarter_result_nms))
if len(half_result_nms) != 0:
    note_all = np.vstack((note_all, half_result_nms))
if len(whole_result_nms) != 0:
    note_all = np.vstack((note_all, whole_result_nms))
note_all_arg = np.argsort(note_all[:, 0] + note_all[:, 1])  # sort by x + y
note_all = note_all[note_all_arg]
high_range = np.zeros((len(g_clef_result_nms), 2), int)
low_range = np.zeros((len(f_clef_result_nms), 2), int)
for i in range(len(g_clef_result_nms)):
    high_range[i][0] = g_clef_result_nms[i][0] - 32
    high_range[i][1] = g_clef_result_nms[i][0] + 186
    low_range[i][0] = f_clef_result_nms[i][0] - 62
    low_range[i][1] = f_clef_result_nms[i][0] + 156

high_list = [[] for i in range(len(high_range))]
low_list = [[] for i in range(len(low_range))]
for i in range(len(note_all)):
    for j in range(len(high_range)):
        if high_range[j][0] < note_all[i][0] < high_range[j][1]:
            high_list[j].append(note_all[i])
            break
        elif low_range[j][0] < note_all[i][0] < low_range[j][1]:
            low_list[j].append(note_all[i])
            break
# save file
high_file = open("high_list.txt", "w")
for i in range(len(high_list)):
    for j in range(len(high_list[i])):
        for k in range(len(high_list[i][j])):
            high_file.write(str(high_list[i][j][k]))
            if k != len(high_list[i][j]) - 1:
                high_file.write(",")
        if j != len(high_list[i]) - 1:
            high_file.write("\n")
    high_file.write("\n")
high_file.close()
low_file = open("low_list.txt", "w")
for i in range(len(low_list)):
    for j in range(len(low_list[i])):
        for k in range(len(low_list[i][j])):
            low_file.write(str(low_list[i][j][k]))
            if k != len(low_list[i][j]) - 1:
                low_file.write(",")
        if j != len(low_list[i]) - 1:
            low_file.write("\n")
    low_file.write("\n")
low_file.close()

end_time = time.clock()
print("Running time : %s s" % (end_time - start_time))

