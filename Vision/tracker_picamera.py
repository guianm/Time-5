import time
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera


# Constants for image resizing
CAM_RESOLUTION = (320, 240)
CAM_FRAMERATE = 24

# Create color dictionary
color = dict()

# Define a new color:
# color['name'] = np.array((H_min, S_min, V_min), (H_max, S_max, V_max))
color['pink'] = np.array(((140, 40, 118), (173, 255, 255)))
color['red'] = np.array(((0, 131, 101), (179, 255, 255)))


def pre_processing(img):
    """Perform format conversion and cleaning operations."""

    # Convert to HSV format
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    return(hsv_img)


def morpho_operation(img_mask):
    """Perform morphology operations on the given image mask."""

    # Perform an opening on the color mask
    kernel = np.ones((11, 11), np.uint8)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)

    return(img_mask)


def find_targets(img_mask, original_img, draw_contours=False):
    """Computes the borders of mask and draw circles around each one."""

    # Compute the contours of each color area of the mask
    image, contours, hierarquy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if draw_contours is True:
        cv2.drawContours(original_img, contours, -1, (0, 0, 255), 2)

    # Skip color
    if len(contours) > 0:
        # Find the contour with maximum area
        best_cnt = max(contours, key=cv2.contourArea)
        # ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(best_cnt)
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        circle_img = cv2.circle(original_img, centroid, 10, (0, 255, 0), 2)
            # (x, y), radius = cv2.minEnclosingCircle(c)
            # center = (int(x), int(y))
            # radius = int(radius)
            # circle_img = cv2.circle(original_img, center, radius, (0, 255, 0), 2)


def main():

    # Instanciate camera
    camera = PiCamera()
    camera.resolution = CAM_RESOLUTION
    camera.framerate = CAM_FRAMERATE

    rawCapture = PiRGBArray(camera, size=CAM_RESOLUTION)

    # Camera warm-up
    time.sleep(0.1)
    cv2.namedWindow('Camera')

    # Aquire frame from the camera
    for camera_frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = camera_frame.array

        # Perform pre-processing steps
        proc_img = pre_processing(frame)

        # Perform color detection for each color on the dictionary
        for c in color.values():

            # Generate color mask
            img_mask = cv2.inRange(proc_img, c[0, :], c[1, :])

            # Eliminate mask noise
            img_mask = morpho_operation(img_mask)
            cv2.imshow('Mask_morpho', img_mask)

            # Find image contours
            find_targets(img_mask, frame)

        # Display the frame
        cv2.imshow('Camera', frame)

        # Clear stream
        rawCapture.truncate(0)

        # Wait for exit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Handle cv2 exit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
