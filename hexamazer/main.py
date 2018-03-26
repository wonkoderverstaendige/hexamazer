import cv2
import math
import numpy as np
import pandas as pd

kernel_3 = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)

LED_POSITIONS = [(377, 445), (538, 715)]
LED_THRESHOLD = 70

MIN_MOUSE_AREA = 50

def fmt_time(s, minimal=False):
    """
    Args:
        s: time in seconds (float for fractional)
        minimal: Flag, if true, only return strings for times > 0, leave rest outs
    Returns: String formatted 99h 59min 59.9s, where elements < 1 are left out optionally.
    """
    ms = s-int(s)
    s = int(s)
    if s < 60 and minimal:
        return "{s:02.3f}s".format(s=s+ms)

    m, s = divmod(s, 60)
    if m < 60 and minimal:
        return "{m:02d}min {s:02.3f}s".format(m=m, s=s+ms)

    h, m = divmod(m, 60)
    return "{h:02d}h {m:02d}min {s:02.3f}s".format(h=h, m=m, s=s+ms)


def overlay(frame, text, x=3, y=3, f_scale=1.):
    f_h = int(13 * f_scale)
    x_ofs = x
    y_ofs = y + f_h
    lines = text.split('\n')

    if frame.ndim < 3:
        color = (255)
        color_bg = (0)
    else:
        color = (255, 255, 255)
        color_bg = (0, 0, 0)

    for n, line in enumerate(lines):
        cv2.putText(frame,
                    line, (x_ofs, y_ofs + n * f_h),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=f_scale,
                    color=color_bg,
                    lineType=cv2.LINE_AA,
                    thickness=3)
        cv2.putText(frame,
                    line, (x_ofs, y_ofs + n * f_h),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=f_scale,
                    color=color,
                    lineType=cv2.LINE_AA)

def centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


class CameraView:
    def __init__(self, x, y, width, height, num_frames, led_pos, thresh_mask=100, thresh_detect=35):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.thresh_mask = thresh_mask
        self.thresh_detect = 255 - thresh_detect  # because we invert the image before thersholding

        self.led_pos = (led_pos[0] - x, led_pos[1] - y)

        self.use_val_frame = True

        self.frame = {}


        self.results = pd.DataFrame(index=range(num_frames))
        print(self.results)
        self.result_position = np.empty((num_frames, 2))
        self.result_largest_area = np.empty(num_frames)
        self.result_sum_area = np.empty(num_frames)
        self.led_state = np.empty(num_frames)



    def update(self, frame, n):
        sub_frame = frame[self.y:self.y + self.height, self.x:self.x + self.width].copy()
        self.frame['raw'] = sub_frame
        self.frame['grey'] = cv2.cvtColor(self.frame['raw'], cv2.COLOR_BGR2GRAY)
        self.frame['hsv'] = cv2.cvtColor(self.frame['raw'], cv2.COLOR_BGR2HSV)
        self.frame['hue'] = self.frame['hsv'][:,:,0]
        self.frame['sat'] = self.frame['hsv'][:,:,1]
        self.frame['val'] = self.frame['hsv'][:,:,2]

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

        # thresold masked image to find mouse
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
        self.result_largest_area[n] = largest_area
        self.result_sum_area[n] = sum_area
        cv2.drawContours(self.frame['raw'], contours, -1, (150, 150, 0), 3)
        if largest_cnt is not None:
            cx, cy = centroid(largest_cnt)
            self.result_position[n, :] = cx, cy
            cv2.drawContours(self.frame['raw'], [largest_cnt], 0, (0, 0, 255), 3)
            overlay(self.frame['raw'],
                    text='{}, {}\nA: {}'.format(cx, cy, largest_area),
                    x=(min(cx + 15, 700)),
                    y=cy + 15)

        # detect LED
        led_state = self.frame['grey'][self.led_pos[1], self.led_pos[0]] > LED_THRESHOLD
        overlay(self.frame['raw'], text='ON' if led_state else 'OFF',
                x=self.led_pos[0] + 5, y=self.led_pos[1] + 5, f_scale=.6)
        self.led_state[n] = led_state

class HexAMazer:
    frame_types = ['raw', 'grey', 'val', 'thresh', 'mask', 'masked', 'hsv', 'hue', 'sat']

    def __init__(self, vid_path, display=True, start_frame=0):
        cv2.destroyAllWindows()
        self.__start_frame = start_frame

        self.capture = cv2.VideoCapture(vid_path)
        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.reset_capture()

        self.frame = None
        self.disp_frame = None

        self.__padding = math.floor((math.log(self.num_frames, 10))) + 1
        self.__replay_fps = int(self.capture.get(cv2.CAP_PROP_FPS))

        self.cam_views = [CameraView(x=0, y=0, width=800, height=600, num_frames=self.num_frames, led_pos=LED_POSITIONS[0]),
                          CameraView(x=0, y=600, width=800, height=600, num_frames=self.num_frames, led_pos=LED_POSITIONS[1])]
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
        while self.alive:
            # if new frames need to be handled
            if (not self.paused) or self.force_move_frames:
                if self.force_move_frames:
                    self.move_rel(self.force_move_frames-1)
                    self.force_move_frames = 0
                self.frame = self.grab()


            if self.frame is None:
                print('No frame returned.')
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
                    character = chr(self.pressed_key)
                except ValueError:
                    character = '??'
                overlay_str = '#{n:0{pad}d}\n' \
                              '{t}\n' \
                              'frame: {key}\n' \
                              'paused: {pause}\n' \
                              'play fps: {fps}\n' \
                              'pressed: {pressed} {char_pressed}' \
                                   .format(n=curr_pos,
                                            pad=self.__padding,
                                            t=fmt_time(curr_pos / 15.),
                                            key=self.frame_types[self.showing],
                                            pause=self.paused,
                                            fps=self.__replay_fps,
                                            pressed=self.pressed_key if self.pressed_key > 0 else '',
                                            char_pressed='({})'.format(
                                              character) if self.pressed_key > 0 else '')

                overlay(self.disp_frame, text=overlay_str)

                overlay_str = 'Trial active: {trial} ({n_trials} total)'.format(trial=self.current_trial+1 if self.current_trial is not None else None,
                                                                           n_trials=len(self.trials))
                overlay(self.disp_frame, text=overlay_str, x=self.frame_width//4, f_scale=1.5)

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
            self.__replay_fps += 5
            #print('Up')

        # down, slow down
        elif key == 40:
            self.__replay_fps = max(5, self.__replay_fps - 5)
            #print('Down')

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
        elif key == ord('F'):
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
            disp_frame = np.vstack(sub_frames)
        else:
            disp_frame = np.vstack(sub_frames)
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

if __name__ == '__main__':
    hx = HexAMazer('C:/Users/reichler/data/hex_maze/x265.mp4')