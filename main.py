from frames_manager.manager import extract_frames
from model.model import extract_features
import os


'''
{'Air_Force_One.mp4': 4494, 'Base jumping.mp4': 4729, 'Bearpark_climbing.mp4': 3341, 'Bike Polo.mp4': 3064, 'Bus_in_Rock_Tunnel.mp4': 5131, 'car_over_camera.mp4': 4382, 'Car_railcrossing.mp4': 5075, 'Cockpit_Landing.mp4': 9046, 'Cooking.mp4': 1280, 'Eiffel Tower.mp4': 4971, 'Excavators river crossing.mp4': 9721, 'Fire Domino.mp4': 1612, 'Jumps.mp4': 950, 'Kids_playing_in_leaves.mp4': 3187, 'Notre_Dame.mp4': 4608, 'Paintball.mp4': 6096, 'paluma_jump.mp4': 2574, 'playing_ball.mp4': 3119, 'Playing_on_water_slide.mp4': 3065, 'Saving dolphines.mp4': 6683, 'Scuba.mp4': 2221, 'St Maarten Landing.mp4': 1751, 'Statue of Liberty.mp4': 3863, 'Uncut_Evening_Flight.mp4': 9672, 'Valparaiso_Downhill.mp4': 5178}
'''

def split_dataset(dataset_path):

    if not os.path.exists("frames_dataset\\SumMe"):
        os.mkdir("frames_dataset\\SumMe")

    videos = [
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

    frames_counter = {}

    for v in videos:
        folder_name = v.replace(" ","_").replace(".mp4", "")
        folder_path = f"frames_dataset\\SumMe\\{folder_name}\\"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Folder '{folder_path}' created!")
            frames_counter[v] = extract_frames(f"{dataset_path}{v}", folder_path,folder_name, verbose=False)
        else:
            print(f"Folder {folder_path} already exists")

    return frames_counter

if __name__ == "__main__":

    frames_per_video = split_dataset("C:\\Users\\XPC\\Downloads\\SumMe\\videos\\")
    print(frames_per_video)
    #extract_features('./frames_dataset/monkeys/')