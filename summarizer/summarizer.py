import cv2
import os
from numpy import unique

FOLDER="summarized_videos\\"

def create_video_from_frames(frames,prefix,bitmap,algo):

    assert len(frames) == len(bitmap)
    
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)

    if not os.path.exists(f"{FOLDER}{prefix}\\"):
        os.mkdir(f"{FOLDER}{prefix}\\")

    frames_path = f"frames_dataset\\SumMe\\{prefix}"
    selected_frames = []
    img_array = []

    for i,v in enumerate(bitmap):
        if v == 1: selected_frames.append(frames[i])

    for f in selected_frames:
        img =  cv2.imread(f"{frames_path}\\{f}")
        height, width, _ = img.shape
        size = (width,height)
        img_array.append(img)

    print(f"Selected frames to create video {len(selected_frames)}")

    out = cv2.VideoWriter(f"{FOLDER}{prefix}\\{prefix}_{algo}.avi",cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

