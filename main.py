import numpy as np
import cv2


# Функция как была в примере, ее не использую
def is_pixel_blue(r: int, g: int, b: int) -> bool:
    return b > r + 100 and b > g + 100


# Более эффективная реализация предыдущей функции,
# только для всего изображения, используя операции над numpy массивами
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


# Т.к. мне не понравилось, что предыдущая функция работала очень плохо
# при разном освещении, я решил написать, используя
# hsl с подобранными руками значениями. Так синие предметы находились лучше
def to_blue_pixel_binary_image_hsl(image: cv2.UMat) -> cv2.UMat:
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask_h_lower = hls_image[:, :, 0] > 90
    mask_h_upper = hls_image[:, :, 0] < 160
    mask_l_lower = hls_image[:, :, 1] > 40
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


# Изображение содержало шум, поэтому я решил, что будет лучше его убрать
def denoise_binary(binary_image: cv2.UMat) -> cv2.UMat:
    denoised = binary_image
    kernel = np.ones(5, dtype=np.uint8)
    # На stackoverflow посоветовали ипользовать эту функцию,
    # она показала себя довольно неплохо
    denoised = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones(10, dtype=np.uint8)
    denoised = cv2.erode(denoised, kernel=kernel)
    denoised = cv2.dilate(denoised, kernel=kernel)
    return denoised


def get_bounding_rects(binary_image: cv2.UMat) -> list[cv2.typing.Rect]:
    # входное изображение здесь все еще было в BGR
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rects = []
    image_height, image_width = binary_image.shape
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        # Иногда все же синие шумы или отсветы
        # пробираются через фильтрацию шума,
        # поэтому маленькие прямоугольники я не учитываю
        if w * h > 0.001 * image_width * image_width:
            rects.append(rect)
    return rects


def add_bounding_rects(img: cv2.UMat, rects: list[cv2.typing.Rect]) -> cv2.UMat:
    for x, y, w, h in rects:
        # Рисую прямоугольники зеленым
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img


def process(frame: cv2.UMat) -> cv2.UMat:
    binary_image = to_blue_pixel_binary_image_hsl(frame.copy())
    binary_image = denoise_binary(binary_image)

    rects = get_bounding_rects(binary_image)
    binary_image = add_bounding_rects(binary_image, rects)
    frame = add_bounding_rects(frame, rects)

    return cv2.hconcat([frame, binary_image])


def capture_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open the video cam')
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f'Frame size: {width} x {height}')

    window = 'Blue markers detector'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    while True:
        success, frame = cap.read()
        if not success:
            print('Cannot read a frame from video stream')
            break
        cv2.imshow(window, process(frame))
        if cv2.waitKey(1) == 27:
            print('ESC key is pressed by user')
            break

    cap.release()
    cv2.destroyAllWindows()


def capture_from_file(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print('Cannot open the video file')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Frames per second: {fps}')

    window = 'Blue markers detector'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    while True:
        success, frame = cap.read()
        if not success:
            print('Cannot read the frame from video file')
            break
        cv2.imshow(window, process(frame))
        if cv2.waitKey(1) == 27:
            print('ESC key is pressed by user')
            break


if __name__ == "__main__":
    capture_from_camera()
    # capture_from_file(input())
