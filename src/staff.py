from rle import *
from commonfunctions import *
from collections import Counter
import cv2




row_percentage = 0.3


def enhance_staff_detection(img):
    """Enhanced staff line detection using adaptive thresholding and morphological operations"""
    # Apply adaptive thresholding for better binarization in varying lighting conditions
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to enhance line detection
    # Use horizontal kernel to detect staff lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return lines


def enhance_staff_removal(img_with_staff, thickness):
    """Enhanced staff line removal with better preservation of musical symbols"""
    img = img_with_staff.copy()
    rows, cols = img.shape
    
    # Apply adaptive thresholding first
    img_enhanced = enhance_staff_detection(img)
    
    # Use horizontal projection with adaptive threshold
    projected = []
    for i in range(rows):
        proj_sum = 0
        for j in range(cols):
            proj_sum += img_enhanced[i][j] == 0  # Count black pixels (staff lines)
        
        # Adaptive threshold based on local context
        local_threshold = 0.15 * cols  # More conservative threshold
        if proj_sum <= local_threshold:
            # Check neighboring rows for staff line consistency
            consistent = True
            for di in [-1, 1]:
                ni = i + di
                if 0 <= ni < rows:
                    neighbor_proj = sum(img_enhanced[ni][j] == 0 for j in range(cols))
                    if neighbor_proj > local_threshold * 1.5:
                        consistent = False
                        break
            
            if consistent:
                # Remove staff line while preserving symbols
                img[i, :] = 1
                # Apply selective opening to preserve note heads
                kernel_size = min(3 * thickness, 15)
                if kernel_size > 0:
                    kernel = np.ones((kernel_size, 1))
                    img[i:i+kernel_size, :] = binary_opening(img[i:i+kernel_size, :], kernel)
    
    return img


def calculate_thickness_spacing(rle, most_common):
    bw_patterns = [most_common_bw_pattern(col, most_common) for col in rle]
    bw_patterns = [x for x in bw_patterns if x]  # Filter empty patterns

    flattened = []
    for col in bw_patterns:
        flattened += col

    pair, count = Counter(flattened).most_common()[0]

    line_thickness = min(pair)
    line_spacing = max(pair)

    return line_thickness, line_spacing


def whitene(rle, vals, max_height):
    rlv = []
    for length, value in zip(rle, vals):
        if value == 0 and length < 1.1*max_height:
            value = 1
        rlv.append((length, value))

    n_rle, n_vals = [], []
    count = 0
    for length, value in rlv:
        if value == 1:
            count = count + length
        else:
            if count > 0:
                n_rle.append(count)
                n_vals.append(1)

            count = 0
            n_rle.append(length)
            n_vals.append(0)
    if count > 0:
        n_rle.append(count)
        n_vals.append(1)

    return n_rle, n_vals


def remove_staff_lines(rle, vals, thickness, shape):
    n_rle, n_vals = [], []
    for i in range(len(rle)):
        rl, val = whitene(rle[i], vals[i], thickness)
        n_rle.append(rl)
        n_vals.append(val)

    return hv_decode(n_rle, n_vals, shape)


def remove_staff_lines_2(thickness, img_with_staff):
    """Enhanced staff line removal with adaptive thresholding"""
    img = img_with_staff.copy()
    rows, cols = img.shape
    
    # Use enhanced staff removal
    return enhance_staff_removal(img, thickness)


def get_rows(start, most_common, thickness, spacing):
    # start = start-most_common
    rows = []
    num = 6
    if start - most_common >= 0:
        start -= most_common
        num = 7
    for k in range(num):
        row = []
        for i in range(thickness):
            row.append(start)
            start += 1
        start += (spacing)
        rows.append(row)
    if len(rows) == 6:
        rows = [0] + rows
    return rows


def horizontal_projection(img):
    projected = []
    rows, cols = img.shape
    for i in range(rows):
        proj_sum = 0
        for j in range(cols):
            proj_sum += img[i][j] == 1
        projected.append([1]*proj_sum + [0]*(cols-proj_sum))
        if(proj_sum <= 0.1*cols):
            return i
    return 0


def get_staff_row_position(img):
    found = 0
    row_position = -1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] == 0):
                row_position = i
                found = 1
                break
        if found == 1:
            break
    return row_position


def coordinator(bin_img, horizontal):
    rle, vals = hv_rle(bin_img)
    most_common = get_most_common(rle)
    thickness, spacing = calculate_thickness_spacing(rle, most_common)
    start = 0
    if horizontal:
        no_staff_img = remove_staff_lines_2(thickness, bin_img)
        staff_lines = otsu(bin_img - no_staff_img)
        start = horizontal_projection(bin_img)
    else:
        no_staff_img = remove_staff_lines(rle, vals, thickness, bin_img.shape)
        no_staff_img = binary_closing(
            no_staff_img, np.ones((thickness+2, thickness+2)))
        no_staff_img = median(no_staff_img)
        no_staff_img = binary_opening(
            no_staff_img, np.ones((thickness+2, thickness+2)))
        staff_lines = otsu(bin_img - no_staff_img)
        staff_lines = binary_erosion(
            staff_lines, np.ones((thickness+2, thickness+2)))
        staff_lines = median(staff_lines, selem=square(21))
        start = get_staff_row_position(staff_lines)
    staff_row_positions = get_rows(
        start, most_common, thickness, spacing)
    staff_row_positions = [np.average(x) for x in staff_row_positions]
    return spacing, staff_row_positions, no_staff_img
