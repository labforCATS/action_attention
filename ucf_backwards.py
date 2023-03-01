import os
import shutil

for i in range(1,48):
    os.makedirs("/research/cwloka/data/action_attn/ucf_backwards/" + str(i))
    for j in range(64):
        current_file = "/research/cwloka/data/action_attn/output_i3d/inputs/input" + str(i) + "pathway0frame" + str(j) + ".jpg"
        new_file = "/research/cwloka/data/action_attn/ucf_backwards/" + str(i) + "/" + str(i) + "_" + str(63-j) + ".jpg"
        shutil.copyfile(current_file, new_file)
