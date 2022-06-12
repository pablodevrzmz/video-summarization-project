from frames_manager.manager import extract_frames
from model.model import extract_features

if __name__ == "__main__":
    #extract_frames("videos_dataset/monkeys/monkeys.mp4","frames_dataset/monkeys")
    extract_features('./frames_dataset/monkeys/')