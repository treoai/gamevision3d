import cv2
import numpy as np

# Input and output paths
video_path1 = 'aaa_2.mp4'
video_path2 = 'bbb_0.mp4'
output_path = 'vert.mp4'

# Open video captures
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

# Get video properties
frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap1.get(cv2.CAP_PROP_FPS)
frame_count = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

for i in range(frame_count):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Compute vertical split position
    alpha = i / (frame_count - 1)
    split_col = int(frame_width * (1 - alpha))

    # Merge frames with vertical transition
    combined = np.hstack((frame1[:, :split_col], frame2[:, split_col:]))

    # Write to output
    out.write(combined)

# Release everything
cap1.release()
cap2.release()
out.release()
print("Video saved to", output_path)
