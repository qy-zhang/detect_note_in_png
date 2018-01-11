import numpy as np
import NMS

a = np.array([[0.1, 0.6], [0.2, 0.5]])
b = np.where(a == 0.5)
print(b)
position = [[66, 67, 67, 78, 78, 78, 78], [195, 194, 195, 17, 18, 126, 127]]
box = np.zeros((len(position[0]), 4), int)
for i in range(len(position[0])):
    box[i][0] = position[0][i]
    box[i][1] = position[1][i]
    box[i][2] = position[0][i] + 30
    box[i][3] = position[1][i] + 30
res = NMS.non_max_suppression_fast(box, 0.6)
print res
res_arg = np.argsort(res[:, 0] + res[:, 1])
res = res[res_arg]
print res
