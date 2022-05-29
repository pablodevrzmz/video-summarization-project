import cv2


def extract_frames( video_path, output_folder ):

    print("Proccesing vide0 %s" % video_path)
    
    capture = cv2.VideoCapture(video_path)

    frames_counter = 0

    while True:

        success, frame = capture.read()
    
        if success:
            cv2.imwrite(f'{output_folder}/{frames_counter}.jpg', frame)
    
        else:
            break

        print("Processing frame %d" % frames_counter)
    
        frames_counter+=1

    capture.release()

    print("Total frames extracted: %d" % frames_counter)