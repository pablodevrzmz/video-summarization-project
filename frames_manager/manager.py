import cv2


def extract_frames( video_path, output_folder, prefix, verbose=True ):

    print("Proccesing video %s" % video_path)
    
    capture = cv2.VideoCapture(video_path)

    frames_counter = 0

    while True:

        success, frame = capture.read()
    
        if success:
            output_name = f'{output_folder}{prefix}-{frames_counter+1}.jpg'
            cv2.imwrite(output_name, frame)
    
        else:
            break
        
        if verbose:
            print("Processing frame: %s" % output_name)
    
        frames_counter+=1

    capture.release()

    print("Total frames extracted: %d" % frames_counter)
    return frames_counter