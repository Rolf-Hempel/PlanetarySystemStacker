from numpy import sqrt, average, diff


def quality_measure(frame):
    dx = diff(frame)[1:, :]  # remove the first row
    dy = diff(frame, axis=0)[:, 1:]  # remove the first column
    sharpness_x = average(sqrt(dx ** 2))
    sharpness_y = average(sqrt(dy ** 2))
    sharpness = min(sharpness_x, sharpness_y)
    return sharpness


def local_contrast(frame, stride):
    frame_strided = frame[::stride, ::stride]
    dx = diff(frame_strided)[1:, :]  # remove the first row
    dy = diff(frame_strided, axis=0)[:, 1:]  # remove the first column
    dnorm = sqrt(dx ** 2 + dy ** 2)
    sharpness = average(dnorm)
    return sharpness
