
import cv2
import os
import numpy as np
import json
# input and output folder
input_dir = "movie_trim"
input_image_dir = "first_frame"
output_dir = "movie_threshold"
roi_file = "roi_data.json"  # save file of ROI
trajectory_save_dir = "trace_data"  # save trace data

os.makedirs(output_dir, exist_ok=True)
os.makedirs(trajectory_save_dir, exist_ok=True)

# get MP4
mp4_trim_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
first_frame_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(".jpeg")]

## dict of ROI
roi_dict = {}

if os.path.exists(roi_file):
    with open(roi_file, "r") as f:
        already_roi = json.load(f)
    print(already_roi)
else:
    already_roi = {}

# global
line_points = []
roi_selected = False

def draw_line(event, x, y, flags, param):
    """ detect event """
    global line_points, roi_selected
    
    if event == cv2.EVENT_LBUTTONDOWN:  # click left
        if len(line_points) < 2:
            line_points.append((x, y))
            print(f"Point added: {x}, {y}")
    elif event == cv2.EVENT_RBUTTONDOWN and line_points:  # click right
        line_points.pop()
        print("Last point removed.")
    elif event == cv2.EVENT_MBUTTONDOWN and len(line_points) == 2:  
        roi_selected = True
        print("ROI selection complete.")

# save each sample
for input_file, input_image in zip(mp4_trim_files, first_frame_files):
    file_name = input_file.split(".")[0]
    file_label_list = file_name.split("_")
    sample_name = "_".join(file_label_list[0:3]) 
    date = file_label_list[2]

    if sample_name in already_roi:
        roi_dict[sample_name] = already_roi[sample_name]
        print('skipping make roi')
        continue

    # read background
    input_video_path = os.path.join(input_dir, input_file)
    cap = cv2.VideoCapture(input_video_path)
    ret, first_frame = cap.read()
    cap.release()
    
    line_points = []
    roi_selected = False

    cv2.namedWindow(f"Set roi {file_name}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"Set roi {file_name}", 1000, 800)
    cv2.setMouseCallback(f"Set roi {file_name}", draw_line)

    while True:
        temp_img = first_frame.copy()
        if len(line_points) == 2:
            cv2.line(temp_img, line_points[0], line_points[1], (0, 255, 0), 2)
        for point in line_points:
            cv2.circle(temp_img, point, 3, (255, 0, 0), -1)

        cv2.imshow(f"Set roi {file_name}", temp_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(line_points) == 2:  # Enterキーで確定
            roi_selected = True
            break

    cv2.destroyAllWindows()

    # calculate 
    (x1, y1), (x2, y2) = line_points
    edge_length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    scale_factor = 40 / edge_length_px  # scaling
    
    # make squere
    dx = x2 - x1
    dy = y2 - y1
    perpendicular = np.array([-dy, dx]) / edge_length_px * (40 / scale_factor)
    
    # Calculate the four points of a square
    roi_polygon = np.array([
        (x1, y1),
        (x2, y2),
        (x2 + perpendicular[0], y2 + perpendicular[1]),
        (x1 + perpendicular[0], y1 + perpendicular[1])
    ], dtype=np.int32)
    
    roi_dict[sample_name] = roi_polygon.tolist()

with open(roi_file, "w") as f:
    json.dump(roi_dict, f)  

print("Saved ROI data")
    


# Read ROI
with open(roi_file, "r") as f:
    roi_dict = json.load(f)

# each movie
for i, (input_file, input_image) in enumerate(zip(mp4_trim_files, first_frame_files)):
    file_name = input_file.split(".")[0]
    file_label_list = file_name.split("_")
    sample_name = "_".join(file_label_list[0:3])
    date = file_name.split("_")[2]

    # input and output path
    input_path = os.path.join(input_dir, input_file)
    input_image_path = os.path.join(input_image_dir, input_image)
    output_path = os.path.join(output_dir, f"{file_name}_threshold.mp4")

    # skip condition
    if os.path.exists(output_path):
        print(f"Skipping (already converted) mp4")
        continue  # skip


    # read background
    background = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # open movie
    cap = cv2.VideoCapture(input_path)

    # get params
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # high
    fps = cap.get(cv2.CAP_PROP_FPS)  

    # Applying pre-obtained ROI
    if sample_name in roi_dict:
        roi_polygon = np.array(roi_dict[sample_name], dtype=np.int32)

        # Calculate the bounding box (minimum rectangle) of the ROI
        x_min = np.min(roi_polygon[:, 0])
        y_min = np.min(roi_polygon[:, 1])
        x_max = np.max(roi_polygon[:, 0])
        y_max = np.max(roi_polygon[:, 1])

        # Calculate width and height
        w = x_max - x_min
        h = y_max - y_min
    else:
        print(f"ROI not found: {file_name}")
        continue


    # The physical size of the arena (e.g. 40cm x 40cm)
    arena_width_cm = 40
    arena_height_cm = 40

    # Pixel to cm conversion factor
    scale_x = arena_width_cm / w
    scale_y = arena_height_cm / h

    # List of mouse trails (in pixels)
    mouse_trajectory_px = []

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_thres = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # Convert the ROI coordinates into a NumPy array (to be applicable to fillPoly)
    roi_polygon_np = np.array([roi_polygon], dtype=np.int32)

    # Pre-create ROI mask (using video resolution)
    roi_mask = np.zeros((height, width), dtype=np.uint8)

    # Fill the ROI area with white (255)
    cv2.fillPoly(roi_mask, roi_polygon_np, 255)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(background, gray)）
        _, thresh = cv2.threshold(diff, 17, 255, cv2.THRESH_BINARY)
        thresh[roi_mask == 0] = 0  # Set pixels outside the ROI to 0
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        out_thres.write(thresh)

        # Detect contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contour (assuming it's a mouse)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Save only if inside ROI
                if x_min <= cx <= x_max and y_min <= cy <= y_max:
                    mouse_trajectory_px.append((cx, cy))

        # display in window
        cv2.namedWindow(f"Threshold {file_name}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Threshold {file_name}", 800, 600)
        cv2.imshow(f"Threshold {file_name}", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Convert mouse trajectories to a NumPy array (to make it 2D)
    mouse_trajectory_px = np.array(mouse_trajectory_px, dtype=np.int32).reshape(-1, 2)
    # Save as a 2D array
    np.save(os.path.join(trajectory_save_dir, f"{file_name}_trace.npy"), mouse_trajectory_px)


    cap.release()
    cv2.destroyAllWindows()


