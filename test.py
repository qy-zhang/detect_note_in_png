import numpy as np
import NMS

# a = np.array([[0.1, 0.6], [0.2, 0.5]])
# b = np.where(a == 0.5)
# print(b)
# position = [[66, 67, 67, 78, 78, 78, 78], [195, 194, 195, 17, 18, 126, 127]]
# box = np.zeros((len(position[0]), 4), int)
# for i in range(len(position[0])):
#     box[i][0] = position[0][i]
#     box[i][1] = position[1][i]
#     box[i][2] = position[0][i] + 30
#     box[i][3] = position[1][i] + 30
# res = NMS.non_max_suppression_fast(box, 0.6)
# print res
# res_arg = np.argsort(res[:, 0] + res[:, 1])
# res = res[res_arg]
# print res

whole = np.loadtxt("whole_result.txt")
half = np.loadtxt("half_result.txt")
quarter = np.loadtxt("quarter_result.txt")
score = np.vstack((quarter, half, whole))
score_arg = np.argsort(score[:, 0] + score[:, 1])  # sort by x + y
score = score[score_arg]
np.savetxt("score_result.txt", score, "%d")


