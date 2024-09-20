import numpy as np
import cv2


def is_pixel_blue(r: int, g: int, b: int) -> bool:
    return b > r + 100 and b > g + 100


def to_blue_pixel_binary_image_bgr(image: cv2.UMat) -> None:
    value_to_add = 100
    mask_b_gt_r = image[:, :, 0] > np.clip(
        image[:, :, 2], 0, 255 - value_to_add) + value_to_add
    mask_b_gt_g = image[:, :, 0] > np.clip(
        image[:, :, 1], 0, 255 - value_to_add) + value_to_add
    mask = mask_b_gt_r & mask_b_gt_g
    image[mask] = [255, 255, 255]
    image[~mask] = [0, 0, 0]
    return image


def to_blue_pixel_binary_image_hsl(image: cv2.UMat) -> None:
    image_in_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask_h_lower = image_in_hls[:, :, 0] > 90
    mask_h_upper = image_in_hls[:, :, 0] < 160
    mask_l_lower = image_in_hls[:, :, 1] > 50
    mask_l_upper = image_in_hls[:, :, 1] < 150
    mask_s = image_in_hls[:, :, 2] > 120
    mask = (
        mask_h_lower
        & mask_h_upper
        & mask_l_lower
        & mask_l_upper
        & mask_s
    )
    image[mask] = [255, 0, 0]
    image[~mask] = [0, 0, 0]
    return image


def capture_from_camera():
    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    # frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0,
    #                       (frame_width, frame_height))

    while True:
        ret, frame = cam.read()

        binary_image = to_blue_pixel_binary_image_hsl(frame.copy())

        # Write the frame to the output file
        # out.write(frame)

        # Display the captured frame
        cv2.imshow('Camera', frame)
        cv2.imshow('Camera2', binary_image)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_from_camera()
