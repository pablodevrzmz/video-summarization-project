import cv2


def extract_frames( video_path, output_folder, prefix, verbose=True ):

    print("Proccesing video %s" % video_path)
    
    capture = cv2.VideoCapture(video_path)

    frames_counter = 0

    while True:

        success, frame = capture.read()
    
        if success:
            output_name = f'{output_folder}{prefix}-{frames_counter+1}.jpg'
            frame = __resize_img(frame,50)
            cv2.imwrite(output_name, frame)
    
        else:
            break
        
        if verbose:
            print("Processing frame: %s" % output_name)
    
        frames_counter+=1

    capture.release()

    print("Total frames extracted: %d" % frames_counter)
    return frames_counter

def __resize_img(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)