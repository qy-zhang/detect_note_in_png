# -*- coding:utf-8 -*-

import librosa
import numpy as np
import sys


def frame(audio_path):
    fs = 44100
    frame_size = 4096
    hop_size = 512
    x, fs_original = librosa.load(audio_path, fs)
    y = np.zeros(frame_size / 2)
    x = np.hstack([y, x, y])
    x = x * 2
    n_frame = np.floor((len(x)-frame_size)/hop_size)+1
    n_frame = int(n_frame)
    x_frame = np.zeros([n_frame,frame_size])
    cur_pos = 0
    for index in xrange(n_frame):
        x_frame[index, :] = x[cur_pos:cur_pos+frame_size]
        cur_pos = cur_pos+hop_size
    file_path = audio_path[0:-4] + 'frame.txt'
    np.savetxt(file_path, x_frame)


if __name__ == '__main__':
    # frame(sys.argv[1])
    frame('play.mp3')
