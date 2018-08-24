from numpy import sqrt, average, diff


def quality_measure(frame):
    dx = diff(frame)[1:, :]  # remove the first row
    dy = diff(frame, axis=0)[:, 1:]  # remove the first column
    sharpness_x = average(sqrt(dx ** 2))
    sharpness_y = average(sqrt(dy ** 2))
    sharpness = min(sharpness_x, sharpness_y)
    return sharpness


def quality_measure_alternative(frame, black_threshold=40.):
    sum_horizontal = sum(sum(abs(frame[:, 2:] - frame[:, :-2]) / (frame[:, 1:-1] + 0.0001) * (
            frame[:, 1:-1] > black_threshold)))
    sum_vertical = sum(sum(abs(frame[2:, :] - frame[:-2, :]) / (frame[1:-1, :] + 0.0001) * (
            frame[1:-1, :] > black_threshold)))
    return min(sum_horizontal, sum_vertical)


def local_contrast(frame, stride):
    frame_strided = frame[::stride, ::stride]
    dx = diff(frame_strided)[1:, :]  # remove the first row
    dy = diff(frame_strided, axis=0)[:, 1:]  # remove the first column
    dnorm = sqrt(dx ** 2 + dy ** 2)
    sharpness = average(dnorm)
    return sharpness

def insert_cross(frame, y_center, x_center, cross_half_len, color):
    if color == 'white':
        rgb = [255, 255, 255]
    elif color == 'red':
        rgb = [255, 0, 0]
    elif color == 'green':
        rgb = [0, 255, 0]
    elif color == 'blue':
        rgb = [0, 0, 255]
    else:
        rgb = [255, 255, 255]
    for y in range(y_center - cross_half_len, y_center + cross_half_len + 1):
        frame[y, x_center] = rgb
    for x in range(x_center - cross_half_len, x_center + cross_half_len + 1):
        frame[y_center, x] = rgb