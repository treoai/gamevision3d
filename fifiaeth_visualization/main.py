import numpy as np
import cv2
from pathlib import Path
import torch
import tqdm

# Importing necessary modules from aitviewer
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.viewer import Viewer
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.node import Node

# Importing local modules
from lib.pitch import SoccerPitch
from lib.utils import project_points

PITCH = SoccerPitch()
SOCCER_FIELD_LINES = PITCH.lines


def create_billboard(camera, img_folder, distance=200, draw_fn=None):
    """Create a billboard from a sequence of images."""
    img_paths = sorted(img_folder.glob("*.jpg"))
    H, W = camera.rows, camera.cols
    pc = Billboard.from_camera_and_distance(
        camera, distance, W, H, textures=[str(path) for path in img_paths], image_process_fn=draw_fn
    )
    return pc

def convert_video_to_images(video_path, output_folder):
    """Convert video to images."""
    output_folder.mkdir(exist_ok=True, parents=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    with tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Converting video to images:") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fp = output_folder / f"{frame_id:06d}.jpg"
            if not fp.exists():
                cv2.imwrite(str(fp), frame)
            frame_id += 1
            pbar.update(1)
    cap.release()


def create_smpl_sequences(params, names=None, colors=None, post_fk_func=None):
    """Create SMPL sequences for animation."""
    num_subjects = len(params["global_orient"])
    
   
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=C.device)


    names = names if names is not None else [f"SMPL_{i}" for i in range(num_subjects)]
    colors = colors if colors is not None else [(0.5, 0.5, 0.5, 1) for _ in range(num_subjects)]

    smpl_seqs = []
    for i in range(num_subjects):
        smpl_seq = SMPLSequence(
            poses_body=params["body_pose"][i],
            smpl_layer=smpl_layer,
            poses_root=params["global_orient"][i],
            betas=params["betas"][i],
            trans=params["transl"][i],
            is_rigged=False,
            post_fk_func=post_fk_func,
            name=names[i],
            color=colors[i],
        )
        smpl_seq.mesh_seq.compute_vertex_and_face_normals.cache_clear()
        smpl_seqs.append(smpl_seq)
    return smpl_seqs


def draw_field(frame, calib, img_shape):
    for line in SOCCER_FIELD_LINES:
        line = project_points(line, **calib, img_shape=img_shape)
        line = line.astype(np.int32)
        cv2.polylines(frame, [line], False, (180, 30, 30), 3)
        
def draw_field_solid(frame, calib, img_shape):
    field_points = []
    for line in SOCCER_FIELD_LINES:
        line = project_points(line, **calib, img_shape=img_shape)
        line = line.astype(np.int32)
        for pt in line:
            x, y = pt
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                field_points.append([x, y])
    field_points = np.array(field_points)    
    # Calculate the convex hull of the points
    hull = cv2.convexHull(field_points)

    # Create a mask for the convex hull area (black image)
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)

    # Fill the mask with white in the convex hull region
    cv2.fillPoly(mask, [hull], 255)

    # Invert the mask (outside the hull becomes white)
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask: all outside hull becomes white
    frame[mask_inv == 255] = (250, 250, 250)  # Outside the hull

    # Draw the filled convex polygon on the image (inside the hull remains green)
    cv2.fillPoly(frame, [hull], (137, 255, 166))  # Green polygon
    for line in SOCCER_FIELD_LINES:
        line = project_points(line, **calib, img_shape=img_shape)
        line = line.astype(np.int32)
        cv2.polylines(frame, [line], False, (180, 180, 180), 4)
    return frame


def make_draw_func(camera=None):
    def _draw_func(img, current_frame_id):
        if camera:
            current_frame_id = min(current_frame_id, len(camera["K"]) - 1)
            img_shape = img.shape
            calib = {
                "R": camera["R"][current_frame_id],
                "t": camera["t"][current_frame_id],
                "k": camera["k"][current_frame_id],
                "f": camera["K"][current_frame_id][0, 0],
                "principal_points": camera["K"][current_frame_id][:2, 2],
            }
            draw_field(img, calib, img_shape)
            #draw_field_solid(img, calib, img_shape)
        return img

    return _draw_func


def make_post_fk_func(camera_params):
    R_all = torch.from_numpy(camera_params["R"]).float()
    t_all = torch.from_numpy(camera_params["t"]).float()
    k_all = torch.from_numpy(camera_params["k"]).float()

    def _post_fk_func(self, vertices, joints, current_frame_only):
        # apply rotation and translation
        nonlocal R_all, t_all, k_all
        R = R_all.to(vertices.device)
        t = t_all.to(vertices.device)
        k = k_all.to(vertices.device)
        if current_frame_only:
            R = R[self.current_frame_id][None]
            t = t[self.current_frame_id][None]
            k = k[self.current_frame_id][None]

        vertices_cam = (R[:, None] @ vertices[..., None]).squeeze(-1)
        vertices_cam += t[:, None]
        vertices_normalized = vertices_cam[..., :2] / vertices_cam[..., 2:]
        r = vertices_normalized.square().sum(-1, keepdims=True)
        # r.clamp_(0, 1)
        k1 = k[:, 0:1]
        k2 = k[:, 1:2]

        scale = 1 + k1[:, None] * r + k2[..., None] * r.square()
        vertices_cam[..., :2] *= scale

        # transform back to world coordinates
        vertices_cam -= t[:, None]
        vertices_cam = (R[:, None].transpose(-1, -2) @ vertices_cam[..., None]).squeeze(-1)
        vertices = vertices_cam
        return vertices, joints

    return _post_fk_func


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="ARG_CRO_220001")
    parser.add_argument("--output", type=str, default="ARG_CRO_220001.mp4", help="Path to save the animation as video (e.g., output.mp4)")

    args = parser.parse_args()

    # define constants
    C.z_up = True

    # Setup viewer and load data
    viewer = Viewer(size=(1920, 1080))
    clip_name = args.sequence
    data_dir = Path("D:/AnasAZ/Saudi/Data/FIFA_ETH/")
    smpl_param_path = data_dir / f"poses/{clip_name}.npz"
    video_path = data_dir / f"videos/{clip_name}.mp4"
    calibration_path = data_dir / f"cameras/{clip_name}.npz"
    if not smpl_param_path.exists():
        raise FileNotFoundError(f"SMPL parameters not found at {smpl_param_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration parameters not found at {calibration_path}")

    camera_params = dict(np.load(calibration_path))

    # Load SMPL parameters
    smpl_params = dict(np.load(smpl_param_path))
    colors = np.random.rand(len(smpl_params["betas"]), 4)
    colors[:, 3] = 1
    post_fk_func = make_post_fk_func(camera_params)
    smpl_seqs = create_smpl_sequences(smpl_params, colors=colors, post_fk_func=post_fk_func)

    # Setup camera and billboard
    camera = OpenCVCamera(camera_params["K"], camera_params["Rt"], 1920, 1080, viewer=viewer, name="Overlay")
    ## create a tmp folder and convert video to images
    img_folder = Path(f"outputs/{clip_name}")
    convert_video_to_images(video_path, img_folder)
    billboard = create_billboard(camera, img_folder, 200, make_draw_func(camera_params))
    viewer.scene.add(billboard)
    viewer.scene.add(camera)

    # Add SMPL sequences to the scene
    smpl_seq_node = Node(name="SMPL", n_frames=len(camera_params["K"]), is_selectable=False)
    for seq in smpl_seqs:
        smpl_seq_node.add(seq)
    viewer.scene.add(smpl_seq_node)

    # Configure lighting
    light = viewer.scene.lights[0]
    light.shadow_enabled = True
    light.azimuth = 270
    light.elevation = 0
    light.shadow_map_size = 64
    light.shadow_map_near = 0.01
    light.shadow_map_far = 50
    viewer.scene.lights[1].shadow_enabled = False

    # Finalize setup and run viewer
    viewer.scene.floor.enabled = False
    viewer.set_temp_camera(camera)
    viewer._init_scene()
    viewer.export_video("aaa.mp4",)
    viewer.run()
    
    # if args.output:
    #     # Set up video writer
    #     fps = 25  # Adjust if you know the real FPS
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     video_writer = cv2.VideoWriter(args.output, fourcc, fps, (1920, 1080))

    #     num_frames = len(camera_params["K"])

    #     for frame_id in tqdm.tqdm(range(num_frames), desc="Rendering to video"):
    #         # Set the current frame for all animated nodes
    #         viewer._init_scene()
    #         viewer.export_animation_range[-1] = viewer.scene.n_frames - 1
    #         viewer.timer.start()
    #         viewer._last_frame_rendered_at = viewer.timer.time
    #         current_time, delta = viewer.timer.next_frame()
    #         # Render the current frame
    #         img = viewer.render(current_time, delta)  # Returns RGB float32 image in [0, 1]
    #         img = (img * 255).astype(np.uint8)
    #         bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #         # Write frame to video
    #         video_writer.write(bgr_img)

    #     video_writer.release()
    #     print(f"Saved video to {args.output}")
    # else:
    #     viewer.run()
