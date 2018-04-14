import cv2
import math
import time
import numpy as np
import argparse
from pathlib import Path

from hexamazer.CameraView import CameraView
from hexamazer.util import fmt_time, overlay

CAPTURE_MODULE = cv2.VideoCapture

DEFAULT_STACKING = np.vstack
DEFAULT_REPLAY_WAIT = 30

AUTOPLAY = False

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

LED_POSITIONS = [(377, 445), (538, 715)]


class HexAMazer:
    frame_types = ['raw', 'grey', 'val', 'thresh', 'mask', 'masked', 'hsv', 'hue', 'sat']

    def __init__(self, vid_path, display=True, start_frame=0, stacking_fun=DEFAULT_STACKING):
        self.__start_frame = start_frame

        self.path = Path(vid_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

        self.stacking_fun = stacking_fun

        self.capture = CAPTURE_MODULE(vid_path)
        # if hasattr(self.capture, 'start'):
        #     self.capture.start()

        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame = None
        self.disp_frame = None

        self.__padding = math.floor((math.log(self.num_frames, 10))) + 1
        self.__replay_fps = DEFAULT_REPLAY_WAIT  # int(self.capture.get(cv2.CAP_PROP_FPS))

        self.cam_views = [CameraView(x=0, y=0, width=800, height=600, name='top', nodes=NODES_A,
                                     num_frames=self.num_frames, led_pos=LED_POSITIONS[0]),
                          CameraView(x=0, y=600, width=800, height=600, name='bottom', nodes=NODES_B,
                                     num_frames=self.num_frames, led_pos=LED_POSITIONS[1])]
        self.trials = []
        self.current_trial = None

        self.alive = True
        self.paused = AUTOPLAY
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
            if not self.paused or self.force_move_frames:
                if self.force_move_frames:
                    self.move_rel(self.force_move_frames - 1)
                    self.force_move_frames = 0
                rv, frame = self.grab()

                if rv:
                    self.frame = frame
                else:
                    if self.frame is None:
                        raise IOError('Frame acquisition failed')
                    self.paused = True

                if self.rotate_frame:
                    self.frame = np.rot90(self.frame).copy()

            curr_pos = self.frame_pos()
            if self.frame is not None:
                for cv in self.cam_views:
                    cv.update(self.frame, curr_pos, self.current_trial)

            if self.display:
                self.disp_frame = self.gather_cam_views()

                # Add text overlay
                try:
                    character = chr(self.pressed_key) if self.pressed_key > 0 else None
                except ValueError:
                    character = '??'
                elapsed = (time.time() - frame_proc_time) * 1000
                ui_wait = 1000 / self.__replay_fps
                frame_proc_time = time.time()
                overlay_str = '{t}\n' \
                              '#{n:0{pad}d}\n' \
                              'frame: {key}\n' \
                              'paused: {pause}\n' \
                              't_wait: {wait_time:.0f} ms\n' \
                              't_loop: {proc_time:3.0f} ms\n' \
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
                key = cv2.waitKey(int(1000 / self.__replay_fps))
                self.process_key(key)

    def process_key(self, key):
        if key < 0:
            return
        self.pressed_key = key

        if key in [27, ord('q')]:
            self.quit()

        # switch to different processing stage
        elif 58 > key >= 49:
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

        # if not self.rotate_frame:
        #     disp_frame = self.stacking_fun(sub_frames)
        # else:
        #     disp_frame = self.stacking_fun(sub_frames)

        disp_frame = self.stacking_fun(sub_frames)
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
        return self.capture.read()

    def quit(self):
        self.alive = False
        cv2.destroyAllWindows()
        # if hasattr(self.capture, 'stop'):
        #     self.capture.stop()

        for cv in self.cam_views:
            cv.store(self.path)

        trials_csv_path = str(self.path) + '.trials.csv'
        with open(trials_csv_path, 'w') as trials_csv:
            for t in self.trials:
                trials_csv.write('{start}, {end}\n'.format(start=t[0], end=t[1]))
        print('Trial data written to {}'.format(trials_csv_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    # parser.add_argument('-M', '--matlab', action='store_true', help='Store output as matlab .mat file')
    parser.add_argument('-H', '--horizontal', action='store_true', help='Show sub-frames side-by-side')

    cli_args = parser.parse_args()
    stacking = np.hstack if cli_args.horizontal else DEFAULT_STACKING

    HexAMazer(cli_args.path, stacking_fun=stacking)


if __name__ == '__main__':
    main()
