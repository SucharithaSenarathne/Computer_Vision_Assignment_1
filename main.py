import cv2
import numpy as np

def load_image(path='sample.jpg'):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def reduce_intensity_levels(img, levels):
    step = 256 // levels
    return ((img // step) * step).astype(np.uint8)

def spatial_average(img, kernel_size):
    return cv2.blur(img, (kernel_size, kernel_size))

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h))

def block_average(img, block_size):
    h, w = img.shape
    h_trim, w_trim = h - h % block_size, w - w % block_size
    img_cropped = img[:h_trim, :w_trim]
    reshaped = img_cropped.reshape(h_trim // block_size, block_size, w_trim // block_size, block_size)
    block_avg = reshaped.mean(axis=(1, 3)).astype(np.uint8)
    return np.kron(block_avg, np.ones((block_size, block_size), dtype=np.uint8))

if __name__ == "__main__":
    img = load_image('sample.jpg')

    # Task 1
    for levels in [2, 4, 8, 16]:
        out = reduce_intensity_levels(img, levels)
        cv2.imwrite(f'intensity_{levels}.png', out)

    # Task 2
    for size in [3, 10, 20]:
        out = spatial_average(img, size)
        cv2.imwrite(f'avg_{size}x{size}.png', out)

    # Task 3
    out_45 = rotate_image(img, 45)
    out_90 = rotate_image(img, 90)
    cv2.imwrite('rot_45.png', out_45)
    cv2.imwrite('rot_90.png', out_90)

    # Task 4
    for block in [3, 5, 7]:
        out = block_average(img, block)
        cv2.imwrite(f'block_{block}x{block}.png', out)

    print("processing completed!")
