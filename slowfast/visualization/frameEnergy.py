import cv2
import os
import matplotlib.pyplot as plt

name = "/research/cwloka/data/action_attn/output/heatmaps/heatmap42pathway0frame1.jpg"

def openFrame(name):
    frame = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    count = 0
    height, width = frame.shape
    for i in range(height):
        for j in range(width):
            count += (frame[i][j] / 255)
    return count/(height*width) # percent activated

videoname = 'heatmap42pathway0'
folder = '/research/cwloka/data/action_attn/output/heatmaps/'
def loopFrames(heatmapname=videoname, folder=folder):
    file_path = folder + heatmapname + 'frame0.jpg'
    count = 0
    energy = []

    while os.path.exists(file_path):
        energy += [openFrame(file_path)]
        count += 1
        file_path = folder + heatmapname + 'frame' + str(count) + '.jpg'
    
    fig = plt.figure()
    plt.plot(range(len(energy)), energy)
    fig.savefig("/research/cwloka/data/action_attn/output/energyGraphs/"+heatmapname+'plot.png')
    plt.close(fig)
    plt.close('all')

num_videos = 47

def loopvideos(folder=folder):
    for i in range(19, num_videos+1):
        heatmapName = 'heatmap' + str(i) + 'pathway0'
        loopFrames(heatmapName, folder)
        # heatmapName = 'heatmap' + str(i) + 'pathway1'
        # loopFrames(heatmapName, folder)
        print(i)




