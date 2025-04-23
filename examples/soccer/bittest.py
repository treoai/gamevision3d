from mmpose.apis import MMPoseInferencer
import cv2
import matplotlib.pyplot as plt

# Initialize the ViTPose model
inferencer = MMPoseInferencer(
    pose2d='vitpose-b',  # You can change to vitpose-l or others
    det_model='rtmdet-nano'  # lightweight detector for person detection
)

# Run inference on an image
result_generator = inferencer('your_image.jpg', return_vis=True)
result = next(result_generator)

# Save or show the visualized result
vis_img = result['visualization'][0]
cv2.imwrite('output.jpg', vis_img)
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
