import sys
import cv2
import numpy as np


def trackbar_update(self):
    pass


def create_trackbars():

    hsv_threshold = np.array(([0, 0, 0], [179, 255, 255]))
    hsv_label = [['H Low', 'S Low', 'V Low'], ['H High', 'S High', 'V High']]

    # Create color trackbars on the GUI
    for i in range(0, 3):
        cv2.createTrackbar(hsv_label[0][i], 'Image', hsv_threshold[0, i], hsv_threshold[1, i], trackbar_update)
        cv2.createTrackbar(hsv_label[1][i], 'Image', hsv_threshold[1, i], hsv_threshold[1, i], trackbar_update)

    return(hsv_threshold, hsv_label)


def update_threshold(frame, hsv_threshold, hsv_label):

    # Get trackbar's input with color range values
    for i in range(0, 3):
        hsv_threshold[0, i] = cv2.getTrackbarPos(hsv_label[0][i], 'Image')
        hsv_threshold[1, i] = cv2.getTrackbarPos(hsv_label[1][i], 'Image')

    # Create HSV color mask
    color_mask = cv2.inRange(frame, hsv_threshold[0, :], hsv_threshold[1, :])

    # Apply mask to the original image
    segmented_img = cv2.bitwise_and(frame, frame, mask=color_mask)

    return(segmented_img)


def pre_processing(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    return(img)


def capture(argv):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Image')

    # Create trackbars for color range input
    hsv_threshold, hsv_label = create_trackbars()

    while(True):
        # Capture video frame-by-frame
        ret, frame = cap.read()

        # Perform some pre-processing
        processed_img = pre_processing(frame)

        # Get threshold color values and segment image
        segmented_img = update_threshold(processed_img, hsv_threshold, hsv_label)

        # Convert back to RGB for better human visualization
        segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_HSV2BGR)

        # Display the frame
        cv2.imshow('Image', segmented_img)

        # Wait for exit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture and handle exit
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture(sys.argv)
