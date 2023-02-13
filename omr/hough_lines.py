import cv2 
import numpy as np

def detect_staffs(img, rho=1, theta=np.pi/180, threshold=25, min_line_length=None, max_line_gap=2):
    lines = detect_lines(img, rho, theta, threshold, min_line_length, max_line_gap)
    if len(np.array(lines).shape) == 0:
        print('No lines detected!')
        return [], []
    _, lines, centroids = merge_lines(img, lines)
    staffline_groups = detect_staffline_groups(centroids)
    staff_lines, bboxes = to_staffs(lines, staffline_groups)
    return staff_lines, bboxes

# code for detecting lines inspired by: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
def detect_lines(img, rho=1, theta=np.pi/180, threshold=25, min_line_length=None, max_line_gap=2):
    neg_gray = 255 - cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if min_line_length == None:
        min_line_length = img.shape[1] // 3

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    return np.squeeze(cv2.HoughLinesP(neg_gray, rho, theta, threshold, np.array([]), min_line_length, max_line_gap))

def bbox_to_line(bbox):
    x1, y1, w, h, _ = bbox
    hh = h // 2
    return [x1, y1 + hh, x1 + w, y1 + hh]

def merge_lines(img, lines):
    
    # draw lines on empty img
    line_image = np.zeros(img.shape, img.dtype)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # do a connected component search and convert resulting 
    # areas (i.e. their bboxes) to lines
    tmp = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    output = cv2.connectedComponentsWithStats(tmp, .0000001)
    (_, labels, stats, centroids) = output
    lines = np.array([bbox_to_line(bbox) for bbox in stats])
    
    # skip first result which is the entire image
    lines = lines[1:]
    centroids = centroids[1:]
    
    return labels, lines, centroids


def detect_staffline_groups(centroids):
    ys = sorted([(idx, c[1]) for idx, c in enumerate(centroids)], key=lambda x: x[1])

    staffline_groups = []
    cur_group = []
    last_delta = -1
    epsilon = 2
    for i in range(len(ys) - 1):
        (idx1, y1), (idx0, y0) = ys[i + 1], ys[i]
        delta = y1 - y0
        if abs(last_delta - delta) > epsilon:
            cur_group = [idx0, idx1]
            last_delta = delta
        else:
            if len(cur_group) == 0:
                print("Warning: Exceeding 5 staff lines!", ys[i])
            cur_group.append(idx1)
        if len(cur_group) == 5:
            staffline_groups.append(cur_group)
            cur_group = []
            last_delta = delta
    return staffline_groups

def to_staffs(lines, staffline_groups):
    lines = np.copy(lines)
    out_lines = []
    bboxes = []
    for group in staffline_groups:
        lines_group = lines[group]
        
        # clamp each staff line to median x0 and x1
        x0_median, x1_median = [int(np.median(lines_group[:,i])) for i in [0, 2]]
        staff_lines = [(x0_median, line[1], x1_median, line[3]) for line in lines_group]
        out_lines.append(staff_lines)
        
        # bounding box estimation for the staff
        max_y = np.max(lines_group[:,1::2])
        min_y = np.min(lines_group[:,1::2])
        bbox = [x0_median, min_y, x1_median, max_y]
        bboxes.append(bbox)
        
    out_lines = np.array(out_lines)
    bboxes = np.array(bboxes).astype(np.float32)
    return out_lines, bboxes

