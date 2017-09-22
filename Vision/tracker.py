import cv2
import numpy as np

# Constants for image resizing
DEF_RESOLUTION = (320, 240)

# Create color dictionary
color = dict()

# Define a new color:
# color['name'] = np.array((H_min, S_min, V_min), (H_max, S_max, V_max))
color['pink'] = np.array(((140, 40, 118), (173, 255, 255)))
color['red'] = np.array(((0, 131, 101), (179, 255, 255)))


class Object:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_coordinates(self):
        return((self.x, self.y))

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y


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


def set_targets(img_mask, original_img):
    """Computes the borders of mask and draw circles around each one."""

    # Compute the contours of each color area of the mask
    image, contours, hierarquy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_img, contours, -1, (0, 0, 255), 3)

    # Draw a circle around each mask area
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        circle_img = cv2.circle(original_img, center, radius, (0, 255, 0), 2)


def main():

    # Aquire image from video source
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Camera')

    while(True):
        # Capture video frame-by-frame
        ret, frame = cap.read()

        # Resize captured image
        frame = cv2.resize(frame, DEF_RESOLUTION)

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
            set_targets(img_mask, frame)

        # Display the frame
        cv2.imshow('Camera', frame)

        # Wait for exit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture and handle exit
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
