# !/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import time
import shutil
import detect_note


def traverse_file(path):
    list_dir1 = os.listdir(path)
    list_dir2 = []
    for i in list_dir1:
        if os.path.isdir(path + '/' + i):
            list_dir2.append(path + '/' + i)
    for i in list_dir2:
        list_file = os.listdir(i)
        for j in list_file:
            file_name, ext = os.path.splitext(j)
            if ext == '.png':
                png_path = i + '/' + j
                detect_note.detect_note_in_png(png_path, './')
                high_name = file_name + '_high_list.txt'
                low_name = file_name + '_low_list.txt'
                g_clef_name = file_name + '_g_clef_list.txt'
                shutil.move('./high_list.txt', i + '/' + high_name)
                shutil.move('./low_list.txt', i + '/' + low_name)
                shutil.move('./g_clef_list.txt', i + '/' + g_clef_name)


start_time = time.clock()
traverse_file('./Test')
end_time = time.clock()
print("Running time : %s s" % (end_time - start_time))
