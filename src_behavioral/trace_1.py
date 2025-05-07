import os
import subprocess
import ffmpeg

# Specifying the input and output folders
input_folder = "movie" 
output_folder = "movie_mp4"  

if not os.path.exists(output_folder):  # Create the output folder if it does not exist
    os.makedirs(output_folder)

mts_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mts")]  # Get all MTS files in a specified folder

for mts_file in mts_files:
    input_path = os.path.join(input_folder, mts_file)
    output_path = os.path.join(output_folder, os.path.splitext(mts_file)[0] + ".mp4")  # .mp4に変換

    # Skip if MP4 file already exists
    if os.path.exists(output_path):
        print(f"Skipping (already converted): {mts_file}")
        continue  # スキップ

    # Run ffmpeg command
    command = ["ffmpeg", "-i", input_path, "-vcodec", "copy", "-acodec", "copy", output_path]
    print(f"Converting: {mts_file} → {output_path}")
    subprocess.run(command, stdout=subprocess.PIPE, stderr=None)


# adjustment
input_dir = "movie_mp4"  
output_dir = "movie_adjust" 
os.makedirs(output_dir, exist_ok=True)
mp4_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]

setting = {'brightness': 0.30, 
            'contrast': 1.7,   
            'saturation': 1.5,
            'gamma': 1.05}


for file in mp4_files:
    file_name = file.split(".")[0]
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, f"{file_name}_adjusted.mp4")
    if os.path.exists(output_path):
        print(f"Skipping (already converted): {file}")
        continue  
    brightness = setting['brightness'] 
    contrast =  setting['contrast']  
    saturation = setting['saturation']
    gamma = setting['gamma']

    (
        ffmpeg
        .input(input_path)
        .filter("eq", brightness=brightness, contrast=contrast, saturation=saturation, gamma=gamma)
        .output(output_path)
        .run()
    )

# trim
input_dir = "movie_adjust"
output_trim_dir = "movie_trim"
output_image = "first_frame"
os.makedirs(output_trim_dir, exist_ok=True)
os.makedirs(output_image, exist_ok=True)
mp4_adjust_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]

for i, file in enumerate(mp4_adjust_files):
    file_name = file.split(".")[0]
    input_path = os.path.join(input_dir, file)
    output_first_image_path = os.path.join(output_image, f"{file_name}_first_frame.jpeg")
    if os.path.exists(output_first_image_path):
        print(f"Skipping (already converted): {file}")
        continue  

    # Save the start frame as an image
    (
        ffmpeg
        .input(input_path)  # トリム開始時刻のフレーム
        .output(output_first_image_path, vframes=1)   # 最初の1フレームだけ保存
        .run()
    )

for i, file in enumerate(mp4_adjust_files):
    file_name = file.split(".")[0]
    print(f"start: {file_name}")
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_trim_dir, f"{file_name}_trim.mp4")
    if os.path.exists(output_path):
        print(f"Skipping (already converted): {file}")
        continue 


    # Trim start time (in seconds)
    start_time = 10  
    duration = 120  
    (
        ffmpeg
        .input(input_path, ss=start_time)
        .output(output_path, t=duration)
        .run()
    )

    print(f"finish: {file_name})")