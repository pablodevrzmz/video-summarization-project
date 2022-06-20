from frames_manager.manager import extract_frames
from model.model import extract_features
from clustering import clustering
from evaluate import evaluate
from summarizer import summarizer
import os
import sys
import numpy as np

'''
Dataset summary
{'Air_Force_One.mp4': 4494, 'Base jumping.mp4': 4729, 'Bearpark_climbing.mp4': 3341, 'Bike Polo.mp4': 3064, 'Bus_in_Rock_Tunnel.mp4': 5131, 'car_over_camera.mp4': 4382, 'Car_railcrossing.mp4': 5075, 'Cockpit_Landing.mp4': 9046, 'Cooking.mp4': 1280, 'Eiffel Tower.mp4': 4971, 'Excavators river crossing.mp4': 9721, 'Fire Domino.mp4': 1612, 'Jumps.mp4': 950, 'Kids_playing_in_leaves.mp4': 3187, 'Notre_Dame.mp4': 4608, 'Paintball.mp4': 6096, 'paluma_jump.mp4': 2574, 'playing_ball.mp4': 3119, 'Playing_on_water_slide.mp4': 3065, 'Saving dolphines.mp4': 6683, 'Scuba.mp4': 2221, 'St Maarten Landing.mp4': 1751, 'Statue of Liberty.mp4': 3863, 'Uncut_Evening_Flight.mp4': 9672, 'Valparaiso_Downhill.mp4': 5178}
'''

CLUSTERS_PERCENTAGE = 0.15
SMOOTH_RATE = 25
NUMBER_OF_FRAME_CHUNKS = 25
NUMBER_OF_FLATTEN_ARRAYS_CHUNKS = 5

VIDEOS = [
    "Air_Force_One.mp4",
    "Base jumping.mp4",
    "Bearpark_climbing.mp4",
    "Bike Polo.mp4",
    "Bus_in_Rock_Tunnel.mp4",
    "car_over_camera.mp4",
    "Car_railcrossing.mp4",
    "Cockpit_Landing.mp4",
    "Cooking.mp4",
    "Eiffel Tower.mp4",
    "Excavators river crossing.mp4",
    "Fire Domino.mp4",
    "Jumps.mp4",
    "Kids_playing_in_leaves.mp4",
    "Notre_Dame.mp4",
    "Paintball.mp4",
    "paluma_jump.mp4",
    "playing_ball.mp4",
    "Playing_on_water_slide.mp4",
    "Saving dolphines.mp4",
    "Scuba.mp4",
    "St Maarten Landing.mp4",
    "Statue of Liberty.mp4",
    "Uncut_Evening_Flight.mp4",
    "Valparaiso_Downhill.mp4",
]


def split_dataset(dataset_path):

    if not os.path.exists("frames_dataset"):
        os.mkdir("frames_dataset")
        os.mkdir("frames_dataset\\SumMe")

    frames_counter = {}

    for v in VIDEOS:
        folder_name = v.replace(" ","_").replace(".mp4", "")
        folder_path = f"frames_dataset\\SumMe\\{folder_name}\\"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Folder '{folder_path}' created!")
            frames_counter[v] = extract_frames(f"{dataset_path}{v}", folder_path,folder_name, verbose=False)
        else:
            print(f"Folder {folder_path} already exists")

    return frames_counter

def run_video_sumarization(frames_path):

    # Extract feautures. Stage (2)
    data = extract_features(frames_path, NUMBER_OF_FRAME_CHUNKS, NUMBER_OF_FLATTEN_ARRAYS_CHUNKS)
    print(f"Features summary: final array shape: {data[0].shape}, used frames: {len(data[1])}")

    all_frames = data[2]

    print("\nRunning clustering")
    target_clusters = int(len(data[1])*CLUSTERS_PERCENTAGE) # 10% of frames

    algorithms = ["kmedoids","aglomerative"]
    methods = [clustering.run_k_medoids, clustering.run_aglomerative_clustering]

    for i,v in enumerate(algorithms):
        
        print(f"Running {v}...")
        _, center_indices = methods[i](data[0],target_clusters)
        
        # Save bitmap for evaluation
        final_bit_map = summarization_bit_map(all_frames,data[1],center_indices,SMOOTH_RATE)
        np.savetxt(f"{frames_path}{v}.csv",np.array(final_bit_map),delimiter=",")
        
        # Create videos
        print(f"Creating video resulting of {v}...")
        video_name = [e for e in frames_path.split("\\") if len(e) > 0].pop()
        summarizer.create_video_from_frames(all_frames,video_name,final_bit_map,v)
    
    # Run evaluation
    for algo in algorithms:
        f1_score, y, y_hat = evaluate.get_f1_score(f"{frames_path}{algo}.csv","./scores_dataset/SumMe/" + video_name + ".csv")
        print(f"{algo}: F1 Score = ",f1_score)

def generate_dataset(path):
    # Stage (1)
    frames_per_video = split_dataset(path)
    print(frames_per_video)

def summarization_bit_map(all_frames,selected_frames,centers_indices,smooth_rate_sec):
    final_map = [0]*len(all_frames)

    center_frames = [selected_frames[i] for i in centers_indices]

    center_pos = []

    # Turn on center flags
    for f in center_frames:
        pos = all_frames.index(f)
        final_map[pos] = 1
        center_pos.append(pos)

    # Turn on smooth_rate_sec before and after flags
    for i,v in enumerate(final_map):
        if i in center_pos:
            i_back = i
            rate = smooth_rate_sec
            while i_back >=0 and rate >=0:
                final_map[i_back] = 1
                rate -= 1
                i_back -= 1
            i_front = i
            rate = smooth_rate_sec
            while i_front < len(final_map) and rate >=0:
                final_map[i_front] = 1
                rate -= 1
                i_front += 1

    return final_map


if __name__ == "__main__":

    dataset_path = sys.argv[1] # frames_per_video = split_dataset("C:\\Users\\XPC\\Downloads\\SumMe\\videos\\
    video_frame_path = sys.argv[2] # ./frames_dataset/SumMe/Jumps/

    # Preprocessing steps
    generate_dataset(dataset_path)

    # Summarize video frames
    run_video_sumarization(video_frame_path)