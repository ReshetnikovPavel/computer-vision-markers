import numpy as np
import cv2


def is_pixel_blue(r: int, g: int, b: int) -> bool:
    return b > r + 100 and b > g + 100


def to_blue_pixel_binary_image_bgr(image: cv2.UMat) -> cv2.UMat:
    value_to_add = 100
    mask_b_gt_r = image[:, :, 0] > np.clip(
        image[:, :, 2], 0, 255 - value_to_add) + value_to_add
    mask_b_gt_g = image[:, :, 0] > np.clip(
        image[:, :, 1], 0, 255 - value_to_add) + value_to_add
    mask = mask_b_gt_r & mask_b_gt_g
    image[mask] = [255, 255, 255]
    image[~mask] = [0, 0, 0]
    return image


def to_blue_pixel_binary_image_hsl(image: cv2.UMat) -> cv2.UMat:
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask_h_lower = hls_image[:, :, 0] > 90
    mask_h_upper = hls_image[:, :, 0] < 160
    mask_l_lower = hls_image[:, :, 1] > 50
    mask_l_upper = hls_image[:, :, 1] < 150
    mask_s = hls_image[:, :, 2] > 110
    mask = (
        mask_h_lower
        & mask_h_upper
        & mask_l_lower
        & mask_l_upper
        & mask_s
    )
    image[mask] = [255, 255, 255]
    image[~mask] = [0, 0, 0]
    return image


def denoise_binary(binary_image):
    kernel = np.array([[0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0]],
                      dtype=np.uint8)
    return cv2.erode(cv2.dilate(binary_image, kernel=kernel), kernel=kernel)


def capture_from_camera():
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        binary_image = denoise_binary(to_blue_pixel_binary_image_hsl(frame.copy()))
        cv2.imshow('Camera', frame)
        cv2.imshow('Camera2', binary_image)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_from_camera()
