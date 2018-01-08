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
# np.savetxt("score_result.txt", score, "%d")


high = [500, 1104, 1706, 2310, 2914]
low = [716, 1320, 1924, 2528, 3132]
high_range = np.zeros((len(high), 2), int)
low_range = np.zeros((len(low), 2), int)
for i in range(len(high)):
    high_range[i][0] = high[i] - 67
    high_range[i][1] = high[i] + 84 + 67
    low_range[i][0] = low[i] - 67
    low_range[i][1] = low[i] + 84 + 67

high_list = [[] for i in range(len(high_range))]
low_list = [[] for i in range(len(low_range))]
for i in range(len(score)):
    for j in range(len(high_range)):
        if high_range[j][0] < score[i][0] < high_range[j][1]:
            high_list[j].append(score[i])
            break
        elif low_range[j][0] < score[i][0] < low_range[j][1]:
            low_list[j].append(score[i])
            break

# save file
high_file = open("high_list.txt", "w")
for i in range(len(high_list)):
    for j in range(len(high_list[i])):
        high_file.write(str(high_list[i][j]))
        high_file.write("\n")
    high_file.write("\n")
high_file.close()
low_file = open("low_list.txt", "w")
for i in range(len(low_list)):
    for j in range(len(low_list[i])):
        low_file.write(str(low_list[i][j]))
        low_file.write("\n")
    low_file.write("\n")
low_file.close()
