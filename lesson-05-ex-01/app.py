import argparse
import cv2
import numpy as np

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Handle an input stream")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc)
    args = parser.parse_args()

    return args


def capture_stream(args):
    ### TODO: Handle image, video or webcam
    image_flag = False
    if args.i == "CAM":
        args.i = 0
    elif args.i.endswith('jpg'):
        image_flag = True
    
    ### TODO: Get and open video capture
    capture = cv2.VideoCapture(args.i)
    capture.open(args.i)
    
    out = None
    if not image_flag:
        out = cv2.VideoWriter("out.mp4", 0x00000021, 30, (100, 100))
        
    while capture.isOpened():
        framesRemaining, frame = capture.read()
        if not framesRemaining:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Re-size the frame to 100x100
        ### NOTE: I'm resizing only the videos
        if not image_flag:
            frame = cv2.resize(frame, (100, 100))

        ### TODO: Add Canny Edge Detection to the frame, 
        ###       with min & max values of 100 and 200
        frame = cv2.Canny(frame, 100, 200)
        ###       Make sure to use np.dstack after to make a 3-channel image
        frame = np.dstack((frame, frame, frame))

        ### TODO: Write out the frame, depending on image or video
        if image_flag:
            cv2.imwrite("out.jpg", frame)
        else:
            out.write(frame)
        
        # break if escape key is pressed
        if key_pressed == 27:
            break

    ### TODO: Close the stream and any windows at the end of the application
    if not image_flag:
        out.release()
    capture.release()
    cv2.destroyAllWindows()

def main():
    args = get_args()
    capture_stream(args)


if __name__ == "__main__":
    main()
