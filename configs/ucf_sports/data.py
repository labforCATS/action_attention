import csv
import shutil


def first_20(filename):
    """
    Pulls out the first 20 videos of a csv file and creates a new one
    Also copies the corresponding frames and videos into a new file
    """
    count = 0
    with open(filename, "r") as file:
        csv_reader = csv.reader(file, delimiter=" ")
        new_csv = []
        video_num = []
        current_num = 0
        for row in csv_reader:
            if row[1] != current_num:
                count += 1
                current_num = row[1]
                video_num += [row[0]]
            if count == 3:
                break
            new_csv += [row[:-1] + ['""']]
    absolute = "/media/cats/32b7c353-4595-42d8-81aa-d029f1556567/something_something/20bn-something-something-v2/"
    absolute_new = "/media/cats/32b7c353-4595-42d8-81aa-d029f1556567/something_something/small_set/"
    csv_place = "./../../val.csv"
    # for folder in video_num[1:]:
    #     shutil.copytree(absolute + folder, absolute_new + folder)
    f = open(csv_place, "w")
    writer = csv.writer(f, delimiter=" ")
    for row in new_csv:
        row[3] = absolute_new + row[3]
        writer.writerow(row)
        print(row)
    f.close
    return
