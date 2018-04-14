import cv2
import numpy as np
import pandas as pd

from hexamazer.util import centroid, overlay, distance

kernel_3 = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)

LED_THRESHOLD = 70

MIN_MOUSE_AREA = 50
MIN_DIST_TO_NODE = 100

THICKNESS_MINOR_CONTOUR = 1
THICKNESS_MAJOR_CONTOUR = 1
DRAW_MINOR_CONTOURS = False
DRAW_MAJOR_CONTOURS = False

TRAIL_LENGTH = 30
DRAW_TRAIL = True


class CameraView:
    def __init__(self, name, x, y, width, height, num_frames, led_pos, nodes, thresh_mask=100, thresh_detect=35):
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.thresh_mask = thresh_mask
        self.thresh_detect = 255 - thresh_detect  # because we invert the image before applying the threshold

        self.led_pos = (led_pos[0] - x, led_pos[1] - y)
        self.nodes = nodes

        self.use_val_frame = True

        self.frame = {}

        self.results = pd.DataFrame(index=range(1, num_frames + 1),
                                    columns=['processed', 'x', 'y', 'led', 'trial'])
        self.results['processed'] = 0

    def update(self, frame, n, trial):
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

        # draw all contours
        if DRAW_MINOR_CONTOURS:
            cv2.drawContours(self.frame['raw'], contours, -1, (150, 150, 0), THICKNESS_MINOR_CONTOUR)

        closest_node = None
        closest_distance = 1e12

        if largest_cnt is not None:
            # center coordinates of contour
            cx, cy = centroid(largest_cnt)
            self.results.loc[n, 'x'] = cx
            self.results.loc[n, 'y'] = cy

            # draw largest contour and contour label
            if DRAW_MAJOR_CONTOURS:
                cv2.drawContours(self.frame['raw'], [largest_cnt], 0, (0, 0, 255), THICKNESS_MAJOR_CONTOUR)
            overlay(self.frame['raw'],
                    text='{}, {}\nA: {}'.format(cx, cy, largest_area),
                    x=(min(cx + 15, 700)),
                    y=cy + 15)
            cv2.circle(self.frame['raw'], (cx, cy), 3, color=(255, 255, 255))

            # Find closest node
            for node_id, node in self.nodes.items():
                dist = distance(cx, cy, node['x'] - self.x, node['y'] - self.y)
                if dist < closest_distance and dist < MIN_DIST_TO_NODE:
                    closest_distance = dist
                    closest_node = node_id

        points = self.results.loc[max(n - TRAIL_LENGTH, 1):n, ['x', 'y']].values
        if DRAW_TRAIL and len(points) > 1:
            for p_idx in range(len(points) - 1):
                try:
                    x1, y1 = map(int, tuple(points[p_idx, :]))
                    x2, y2 = map(int, tuple(points[p_idx + 1, :]))
                except ValueError:
                    pass
                else:
                    cv2.line(self.frame['raw'], (x1, y1), (x2, y2), color=(255, 255, 255))

        # Label nodes
        for node_id, node in self.nodes.items():
            color = (255, 0, 0) if node_id == closest_node else (255, 255, 255)
            cv2.circle(self.frame['raw'], (node['x'] - self.x, node['y'] - self.y), MIN_DIST_TO_NODE // 2, color)

            overlay(self.frame['raw'], text=str(node_id), color=color,
                    x=node['x'] - self.x, y=node['y'] - self.y, f_scale=2.)

        # detect LED
        led_val = self.frame['grey'][self.led_pos[1], self.led_pos[0]]
        led_state = led_val > LED_THRESHOLD
        overlay(self.frame['raw'], text='ON' if led_state else 'OFF',
                x=self.led_pos[0] + 5, y=self.led_pos[1] + 5, f_scale=.6)
        self.results.loc[n, 'led'] = led_val

        self.results.loc[n, 'processed'] = 1
        self.results.loc[n, 'trial'] = trial

    def store(self, path):
        out_path = str(path) + '.{}.csv'.format(self.name)
        self.results.to_csv(out_path, index_label='frame')
        print('Results for <{}> stored at {}'.format(self.name, out_path))
