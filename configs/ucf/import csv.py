import csv

def first_20(filename):
    """
    Pulls out the first 20 videos of a csv file and creates a new one
    Also copies the corresponding frames and videos into a new file
    """

    csv_reader = csv.reader(filename)
    new_csv = []
    for row in csv_reader:
        print(row)
        break