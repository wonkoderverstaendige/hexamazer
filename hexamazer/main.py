import cv2
import math
import time
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

kernel_3 = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)

LED_POSITIONS = [(377, 445), (538, 715)]
LED_THRESHOLD = 70

MIN_MOUSE_AREA = 50
MIN_DIST_TO_NODE = 100

STACKING_FUN = np.vstack

NODES_A = {1: {'use': None, 'x': 73, 'y': 66},
           2: {'use': None, 'x': 211, 'y': 6},
           3: {'use': None, 'x': 345, 'y': 81},
           4: {'use': None, 'x': 486, 'y': 23},
           5: {'use': None, 'x': 613, 'y': 100},
           6: {'use': None, 'x': 48, 'y': 233},
           7: {'use': None, 'x': 334, 'y': 233},
           8: {'use': None, 'x': 620, 'y': 251},
           9: {'use': None, 'x': 186, 'y': 313},
           10: {'use': None, 'x': 476, 'y': 322},
           11: {'use': None, 'x': 765, 'y': 335},
           12: {'use': None, 'x': 165, 'y': 481},
           13: {'use': None, 'x': 472, 'y': 495},
           14: {'use': None, 'x': 768, 'y': 508},
           15: {'use': None, 'x': 314, 'y': 573},
           16: {'use': None, 'x': 625, 'y': 592}}

NODES_B = {6: {'use': None, 'x': 186, 'y': 611},
           7: {'use': None, 'x': 448, 'y': 615},
           8: {'use': None, 'x': 718, 'y': 625},
           9: {'use': None, 'x': 309, 'y': 689},
           10: {'use': None, 'x': 583, 'y': 690},
           12: {'use': None, 'x': 310, 'y': 831},
           13: {'use': None, 'x': 591, 'y': 837},
           15: {'use': None, 'x': 453, 'y': 913},
           16: {'use': None, 'x': 746, 'y': 912},
           17: {'use': None, 'x': 43, 'y': 689},
           18: {'use': None, 'x': 22, 'y': 828},
           19: {'use': None, 'x': 150, 'y': 908},
           20: {'use': None, 'x': 148, 'y': 1078},
           21: {'use': None, 'x': 297, 'y': 1171},
           22: {'use': None, 'x': 460, 'y': 1082},
           23: {'use': None, 'x': 614, 'y': 1173},
           24: {'use': None, 'x': 760, 'y': 1080}}


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
    if color is None:
        if frame.ndim < 3:
            color = (255)
        else:
            color = (255, 255, 255)
    color_bg = [0 for _ in color]

    f_h = int(13 * f_scale)
    x_ofs = x
    y_ofs = y + f_h

    lines = text.split('\n')

    for n, line in enumerate(lines):
        text_size, _ = cv2.getTextSize(line, fontFace=cv2.FONT_HERSHEY_PLAIN,
                                       fontScale=f_scale, thickness=thickness+1)
        if origin == 'right':
            text_x =  x_ofs - text_size[0]
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


def centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

class CameraView:
    def __init__(self, name, x, y, width, height, num_frames, led_pos, nodes, thresh_mask=100, thresh_detect=35):
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.thresh_mask = thresh_mask
        self.thresh_detect = 255 - thresh_detect  # because we invert the image before thresholding

        self.led_pos = (led_pos[0] - x, led_pos[1] - y)
        self.nodes = nodes

        self.use_val_frame = True

        self.frame = {}

        self.results = pd.DataFrame(index=range(num_frames),
                                    columns=['largest_area', 'largest_x', 'largest_y',
                                             'sum_area', 'led_state'])

    def update(self, frame, n):
        sub_frame = frame[self.y:self.y + self.height, self.x:self.x + self.width].copy()
        self.frame['raw'] = sub_frame
        self.frame['grey'] = cv2.cvtColor(self.frame['raw'], cv2.COLOR_BGR2GRAY)
        self.frame['hsv'] = cv2.cvtColor(self.frame['raw'], cv2.COLOR_BGR2HSV)
        self.frame['hue'] = self.frame['hsv'][:, :, 0]
        self.frame['sat'] = self.frame['hsv'][:, :, 1]
        self.frame['val'] = self.frame['hsv'][:, :, 2]

        if not self.use_val_frame:
            foi = self.frame['grey']
        else:
            foi = self.frame['val']

        # on first frame, create mask
        if n == 1:
            _, mask = cv2.threshold(foi, self.thresh_mask, 255, cv2.THRESH_BINARY)
            self.frame['mask'] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_3)

        # apply mask to either greyscale or (HS)Value image
        masked = cv2.bitwise_not(foi) * (self.frame['mask'] // 255)
        self.frame['masked'] = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel_3)

        # threshold masked image to find mouse
        _, thresh = cv2.threshold(self.frame['masked'], self.thresh_detect, 255, cv2.THRESH_BINARY)
        self.frame['thresh'] = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_3)

        # find contours
        _, contours, hierarchy = cv2.findContours(self.frame['thresh'], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find largest contour
        largest_cnt, largest_area = None, 0
        sum_area = 0
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area > MIN_MOUSE_AREA:
                sum_area += area
                if area > largest_area:
                    largest_area = area
                    largest_cnt = cnt

        # store per-frame results
        self.results['largest_area'][n] = largest_area
        self.results['sum_area'][n] = sum_area
        cv2.drawContours(self.frame['raw'], contours, -1, (150, 150, 0), 3)

        closest_node = None
        closest_distance = 1e12

        if largest_cnt is not None:
            cx, cy = centroid(largest_cnt)
            self.results['largest_x'][n] = cx
            self.results['largest_y'][n] = cy
            cv2.drawContours(self.frame['raw'], [largest_cnt], 0, (0, 0, 255), 3)
            overlay(self.frame['raw'],
                    text='{}, {}\nA: {}'.format(cx, cy, largest_area),
                    x=(min(cx + 15, 700)),
                    y=cy + 15)

            # Find closest node
            for node_id, node in self.nodes.items():
                dist = distance(cx, cy, node['x'] - self.x, node['y'] - self.y)
                if dist < closest_distance and dist < MIN_DIST_TO_NODE:
                    closest_distance = dist
                    closest_node = node_id

        #Label nodes
        for node_id, node in self.nodes.items():
            color = (255, 0, 0) if node_id == closest_node else (255, 255, 255)
            cv2.circle(self.frame['raw'], (node['x'] - self.x, node['y'] - self.y), MIN_DIST_TO_NODE//2, color)

            overlay(self.frame['raw'], text=str(node_id), color=color,
                    x=node['x'] - self.x, y=node['y'] - self.y, f_scale=2.)

        # detect LED
        led_state = self.frame['grey'][self.led_pos[1], self.led_pos[0]] > LED_THRESHOLD
        overlay(self.frame['raw'], text='ON' if led_state else 'OFF',
                x=self.led_pos[0] + 5, y=self.led_pos[1] + 5, f_scale=.6)
        self.results['led_state'][n] = led_state

    def store(self, path):
        self.results.to_pickle(str(path) + '.{}.hxm_pickle'.format(self.name))


class HexAMazer:
    frame_types = ['raw', 'grey', 'val', 'thresh', 'mask', 'masked', 'hsv', 'hue', 'sat']

    def __init__(self, vid_path, display=True, start_frame=0):
        self.__start_frame = start_frame

        self.path = Path(vid_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

        self.capture = cv2.VideoCapture(vid_path)
        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.reset_capture()

        self.frame = None
        self.disp_frame = None

        self.__padding = math.floor((math.log(self.num_frames, 10))) + 1
        self.__replay_fps = int(self.capture.get(cv2.CAP_PROP_FPS))

        self.cam_views = [CameraView(x=0, y=0, width=800, height=600, name='top', nodes=NODES_A,
                                     num_frames=self.num_frames, led_pos=LED_POSITIONS[0]),
                          CameraView(x=0, y=600, width=800, height=600, name='bottom', nodes=NODES_B,
                                     num_frames=self.num_frames, led_pos=LED_POSITIONS[1])]
        self.trials = []
        self.current_trial = None

        self.alive = True
        self.paused = False
        self.force_move_frames = 0
        self.display = display
        self.rotate_frame = False
        self.frame_jump_distance = 30
        self.showing = 0

        self.pressed_key = -1
        self.loop()

    def loop(self):
        frame_proc_time = time.time()
        while self.alive:
            # if new frames need to be handled
            if (not self.paused) or self.force_move_frames:
                if self.force_move_frames:
                    self.move_rel(self.force_move_frames - 1)
                    self.force_move_frames = 0
                self.frame = self.grab()

            if self.frame is None:
                print('No frame returned.')
                self.quit()
                break

            if self.rotate_frame:
                self.frame = np.rot90(self.frame).copy()

            curr_pos = self.frame_pos()
            for cv in self.cam_views:
                cv.update(self.frame, curr_pos)

            if self.display:
                self.disp_frame = self.gather_cam_views()

                # Add text overlay
                try:
                    character = chr(self.pressed_key) if self.pressed_key > 0 else None
                except ValueError:
                    character = '??'
                elapsed =  (time.time() - frame_proc_time) * 1000
                ui_wait = 1000 / self.__replay_fps
                frame_proc_time = time.time()
                overlay_str = '{t}\n' \
                              '#{n:0{pad}d}\n' \
                              'frame: {key}\n' \
                              'paused: {pause}\n' \
                              't_wait: {wait_time:.0f} ms\n' \
                              't_loop: {proc_time:3.0f} ms\n'\
                              'input: {pressed} {char_pressed}' \
                    .format(n=curr_pos,
                            pad=self.__padding,
                            t=fmt_time(curr_pos / 15.),
                            key=self.frame_types[self.showing],
                            pause=self.paused,
                            wait_time=ui_wait,
                            pressed=self.pressed_key if self.pressed_key > 0 else None,
                            char_pressed='({})'.format(character) if self.pressed_key > 0 else '',
                            proc_time=elapsed)
                overlay(self.disp_frame, x=self.frame_width, text=overlay_str, origin='right')

                overlay_str = 'Trial active: {trial} ({n_trials} total)'.format(
                    trial=self.current_trial + 1 if self.current_trial is not None else None,
                    n_trials=len(self.trials))
                overlay(self.disp_frame, text=overlay_str, x=self.frame_width // 4, f_scale=1.5)

                cv2.imshow('Hex-A-Mazer', self.disp_frame)
                self.process_key(cv2.waitKey(int(1000 / self.__replay_fps)))

    def process_key(self, key):
        if key < 0:
            return
        self.pressed_key = key

        if key in [27, ord('q')]:
            self.quit()

        # switch to different processing stage
        elif 59 > key >= 49:
            self.showing = min(int(chr(key)), len(self.frame_types)) - 1

        # pause
        elif key == 32:
            self.paused = not self.paused

        # move 1 frame forward
        elif key == ord('.'):
            self.force_move_frames = 1

        # move 1 frame back
        elif key == ord(','):
            self.force_move_frames = -1

        # up, speed up
        elif key == 38:
            self.__replay_fps = min(1000, self.__replay_fps + 5)
            # print('Up')

        # down, slow down
        elif key == 40:
            self.__replay_fps = max(5, self.__replay_fps - 5)
            # print('Down')

        # left, jump X frames back
        elif key == ord('<'):
            self.force_move_frames = -self.frame_jump_distance

        # right, jump X frames forward
        elif key == ord('>'):
            self.force_move_frames = self.frame_jump_distance

        # replay speeds
        elif key == ord('s'):
            self.__replay_fps = 5
        elif key == ord('n'):
            self.__replay_fps = 15
        elif key == ord('f'):
            self.__replay_fps = 30
        elif key == ord('h'):
            self.__replay_fps = 500

        # set trial start/end labels
        elif key == ord('t'):
            n = self.frame_pos()
            if self.current_trial is None:
                self.current_trial = len(self.trials)
                self.trials.append([n, n])
            else:
                self.trials[self.current_trial][1] = n
                self.current_trial = None

    def gather_cam_views(self):
        frame_type = self.frame_types[self.showing]
        try:
            sub_frames = [cv.frame[frame_type] for cv in self.cam_views]
        except KeyError:
            sub_frames = [cv.frame['raw'] for cv in self.cam_views]

        if not self.rotate_frame:
            disp_frame = STACKING_FUN(sub_frames)
        else:
            disp_frame = STACKING_FUN(sub_frames)
        return disp_frame

    def frame_pos(self):
        return int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))

    def reset_capture(self):
        self.move_to(self.__start_frame)

    def move_to(self, n):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, n)

    def move_rel(self, delta):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos() + delta)

    def grab(self):
        rv, frame = self.capture.read()
        return frame if rv else None

    def quit(self):
        self.alive = False
        cv2.destroyAllWindows()
        self.capture.release()

        for cv in self.cam_views:
            cv.store(self.path)

        with open(str(self.path) + 'trials.csv', 'w') as trials_csv:
            for t in self.trials:
                trials_csv.write('{start}, {end}'.format(start=t[0], end=t[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')

    cli_args = parser.parse_args()
    hx = HexAMazer(cli_args.path)
