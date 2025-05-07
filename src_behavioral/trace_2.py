
import cv2
import os
import numpy as np
import json
# 入力フォルダと出力フォルダの指定
input_dir = "movie_trim"
input_image_dir = "first_frame"
output_dir = "movie_threshold"
roi_file = "roi_data.json"  # ROI保存用ファイル
trajectory_save_dir = "trace_data"  # 軌跡データの保存

os.makedirs(output_dir, exist_ok=True)
os.makedirs(trajectory_save_dir, exist_ok=True)

# 指定フォルダ内のすべてのMP4ファイルを取得
mp4_trim_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
first_frame_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(".jpeg")]

## ROIを保存する辞書
roi_dict = {}

if os.path.exists(roi_file):
    with open(roi_file, "r") as f:
        already_roi = json.load(f)
    print(already_roi)
else:
    already_roi = {}

# グローバル変数（ROI選択用）
line_points = []
roi_selected = False

def draw_line(event, x, y, flags, param):
    """ マウスイベントでROIの基準となる1辺を選択する関数 """
    global line_points, roi_selected
    
    if event == cv2.EVENT_LBUTTONDOWN:  # 左クリックで点を追加
        if len(line_points) < 2:
            line_points.append((x, y))
            print(f"Point added: {x}, {y}")
    elif event == cv2.EVENT_RBUTTONDOWN and line_points:  # 右クリックで最後の点を削除
        line_points.pop()
        print("Last point removed.")
    elif event == cv2.EVENT_MBUTTONDOWN and len(line_points) == 2:  # 中クリックで確定
        roi_selected = True
        print("ROI selection complete.")

# ROIをサンプルごとに事前に取得して保存
for input_file, input_image in zip(mp4_trim_files, first_frame_files):
    file_name = input_file.split(".")[0]
    file_label_list = file_name.split("_")
    sample_name = "_".join(file_label_list[0:3]) 
    date = file_label_list[2]

    if sample_name in already_roi:
        roi_dict[sample_name] = already_roi[sample_name]
        print('skipping make roi')
        continue

    # 背景画像を読み込む
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

    # 選択した辺を基準に正方形のROIを計算
    (x1, y1), (x2, y2) = line_points
    edge_length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    scale_factor = 30 / edge_length_px  # 1辺30cmにスケーリング
    
    # 垂直方向のベクトルを求めて正方形を作る
    dx = x2 - x1
    dy = y2 - y1
    perpendicular = np.array([-dy, dx]) / edge_length_px * (30 / scale_factor)
    
    # 正方形の4点を計算
    roi_polygon = np.array([
        (x1, y1),
        (x2, y2),
        (x2 + perpendicular[0], y2 + perpendicular[1]),
        (x1 + perpendicular[0], y1 + perpendicular[1])
    ], dtype=np.int32)
    
    roi_dict[sample_name] = roi_polygon.tolist()

with open(roi_file, "w") as f:
    json.dump(roi_dict, f)  

print("ROIデータを保存しました。")
    


# ROIデータを読み込み
with open(roi_file, "r") as f:
    roi_dict = json.load(f)

# 各動画の処理
for i, (input_file, input_image) in enumerate(zip(mp4_trim_files, first_frame_files)):
    file_name = input_file.split(".")[0]
    file_label_list = file_name.split("_")
    sample_name = "_".join(file_label_list[0:3])
    date = file_name.split("_")[2]

    # 入力・出力パス
    input_path = os.path.join(input_dir, input_file)
    input_image_path = os.path.join(input_image_dir, input_image)
    output_path = os.path.join(output_dir, f"{file_name}_threshold.mp4")

    # すでにMP4ファイルが存在する場合はスキップ
    if os.path.exists(output_path):
        print(f"Skipping (already converted) mp4")
        continue  # スキップ


    # 背景画像を読み込む
    background = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # 動画を開く
    cap = cv2.VideoCapture(input_path)

    # フレーム数、幅、高さを取得
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # フレーム数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 幅
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高さ
    fps = cap.get(cv2.CAP_PROP_FPS)  # フレームレート（オプション）

    # 取得済みのROIを適用
    if sample_name in roi_dict:
        roi_polygon = np.array(roi_dict[sample_name], dtype=np.int32)

        # ROIのバウンディングボックス（最小矩形）を計算
        x_min = np.min(roi_polygon[:, 0])
        y_min = np.min(roi_polygon[:, 1])
        x_max = np.max(roi_polygon[:, 0])
        y_max = np.max(roi_polygon[:, 1])

        # 幅と高さを計算
        w = x_max - x_min
        h = y_max - y_min
    else:
        print(f"ROIが見つかりません: {file_name}")
        continue


    # アリーナの物理サイズ（例: 30cm × 30cm）
    arena_width_cm = 30
    arena_height_cm = 30

    # ピクセル→cm変換係数
    scale_x = arena_width_cm / w
    scale_y = arena_height_cm / h

    # マウス軌跡を記録するリスト（ピクセル単位）
    mouse_trajectory_px = []

    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # コーデック
    out_thres = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # ROIの座標をNumPy配列に変換（fillPolyに適用できる形にする）
    roi_polygon_np = np.array([roi_polygon], dtype=np.int32)

    # ROIマスクを事前に作成（動画の解像度を利用）
    roi_mask = np.zeros((height, width), dtype=np.uint8)

    # ROI領域を白 (255) に塗る
    cv2.fillPoly(roi_mask, roi_polygon_np, 255)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 背景との差分
        diff = cv2.absdiff(background, gray)

        # 閾値処理（二値化）
        _, thresh = cv2.threshold(diff, 17, 255, cv2.THRESH_BINARY)

        # ROI外を無条件で0にする
        thresh[roi_mask == 0] = 0  # ROIの外のピクセルを0に設定

        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        out_thres.write(thresh)

        # 輪郭を検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 最大の輪郭を取得（マウスと仮定）
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # ROI内にいる場合のみ保存
                if x_min <= cx <= x_max and y_min <= cy <= y_max:
                    mouse_trajectory_px.append((cx, cy))

        # ウィンドウに表示
        cv2.namedWindow(f"Threshold {file_name}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Threshold {file_name}", 800, 600)
        cv2.imshow(f"Threshold {file_name}", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # マウス軌跡を NumPy 配列に変換（2Dにする）
    mouse_trajectory_px = np.array(mouse_trajectory_px, dtype=np.int32).reshape(-1, 2)
    # 2D配列の形で保存
    np.save(os.path.join(trajectory_save_dir, f"{file_name}_trace.npy"), mouse_trajectory_px)


    cap.release()
    cv2.destroyAllWindows()

