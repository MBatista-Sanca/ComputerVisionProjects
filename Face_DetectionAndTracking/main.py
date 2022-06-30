import cv2 as cv
import numpy as np

def main():
    cap = cv.VideoCapture(0)
    print(cap.isOpened())

    while cap.isOpened():
        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        
        cv.imshow("Frame", frame_gray)


        if cv.waitKey(1) & 0xFF == ord('q'):
            return 0

if __name__ == "__main__":
    main()

