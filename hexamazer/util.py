import cv2
import math


def centroid(cnt):
    """X, Y coordinates of the centroid of a contour"""
    moments = cv2.moments(cnt)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx, cy


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def fmt_time(s, minimal=False):
    """
    Args:
        s: time in seconds (float for fractional)
        minimal: Flag, if true, only return strings for times > 0, leave rest outs
    Returns: String formatted 99h 59min 59.9s, where elements < 1 are left out optionally.
    """
    ms = s - int(s)
    s = int(s)
    if s < 60 and minimal:
        return "{s:02.3f}s".format(s=s + ms)

    m, s = divmod(s, 60)
    if m < 60 and minimal:
        return "{m:02d}min {s:02.3f}s".format(m=m, s=s + ms)

    h, m = divmod(m, 60)
    return "{h:02d}h {m:02d}min {s:02.3f}s".format(h=h, m=m, s=s + ms)


def overlay(frame, text, x=3, y=3, f_scale=1., color=None, origin='left', thickness=1):
    """Overlay text onto image. Newline characters are used to split the text string and put on 'new lines'.

    Args:
        frame: numpy array of image
        text: string of (multi-line) text
        x: start of text overlay in x
        y: start of text overlay in y
        f_scale: font scale
        color: Foreground color
        origin: Left or right align of coordinates
        thickness: line thickness
        """
    if color is None:
        if frame.ndim < 3:
            color = (255,)
        else:
            color = (255, 255, 255)

    color_bg = [0 for _ in list(color)]

    f_h = int(13 * f_scale)
    x_ofs = x
    y_ofs = y + f_h

    lines = text.split('\n')

    for n, line in enumerate(lines):
        text_size, _ = cv2.getTextSize(line, fontFace=cv2.FONT_HERSHEY_PLAIN,
                                       fontScale=f_scale, thickness=thickness + 1)
        if origin == 'right':
            text_x = x_ofs - text_size[0]
        else:
            text_x = x_ofs

        # draw text outline
        cv2.putText(frame,
                    line, (text_x, y_ofs + n * f_h),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=f_scale,
                    color=color_bg,
                    lineType=cv2.LINE_AA,
                    thickness=thickness + 1)

        # actual text
        cv2.putText(frame,
                    line, (text_x, y_ofs + n * f_h),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=f_scale,
                    color=color,
                    lineType=cv2.LINE_AA,
                    thickness=thickness)
