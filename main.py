import numpy as np
import cv2


def is_pixel_blue(r: int, g: int, b: int) -> bool:
    return b > r + 100 and b > g + 100


def to_blue_pixel_binary_image(image: cv2.UMat) -> None:
    value_to_add = 100
    mask_b_gt_r = image[:, :, 0] > np.clip(image[:, :, 2], 0, 255 - value_to_add) + value_to_add
    mask_b_gt_g = image[:, :, 0] > np.clip(image[:, :, 1], 0, 255 - value_to_add) + value_to_add
    mask = mask_b_gt_r & mask_b_gt_g
    image[mask] = [255, 255, 255]
    image[~mask] = [0, 0, 0]


def capture_from_camera():
    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()

        to_blue_pixel_binary_image(frame)

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_from_camera()
