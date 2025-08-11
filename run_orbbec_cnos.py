# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from hardware.hardware_orbbec import ORBBEC
from utils.keyboard import KeyManager

from estimater import *
from datareader import *
import argparse


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/T/mesh/XL-T.obj') #textured_simple.obj does not exist
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/T')
  parser.add_argument('--est_refine_iter', type=int, default=7) # default 5
  parser.add_argument('--track_refine_iter', type=int, default=5) # default 2
  parser.add_argument('--debug', type=int, default=3)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_orbbec')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  K = np.loadtxt(f'{args.test_scene_dir}/cam_K_depth.txt').reshape(3,3)

  YOLO_WORLD_PATH = "yolov8s-world.pt"
  YOLOV8_MODEL_PATH = "yolov8n.pt"
  TARGET_CLASS_NAME = "book" # YOLO cup scissors
  TARGET_CLASS_ID = 73 # YOLO 41 76

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  ### ORBBEC

  # reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
  devices = ORBBEC.get_devices(1, 360, 640)
  device = devices[0]
  assert device.connect(), f"Connection to {device.name} failed"


  logging.info("Starting pose estimation")

  km = KeyManager()

  i = 0
  while km.key != 'q':
    logging.info("Press q to stop.")
    MAX_ATTEMPTS = 20

    logging.info("Warming up camera...")
    for _ in range(MAX_ATTEMPTS):
        data = device.get_sensors()
        if data and data.get("rgb") is not None:
            logging.info("RGB stream is live.")
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("Failed to receive RGB frame after warm-up.")
    # time.sleep(1)
    data = device.get_sensors()
    color = data['rgb']
    depth = data['d'].astype(np.int16)
    depth = depth / 1000.0 # convert to meters

    # test with different resolution
    color_resized = Image.fromarray(color).resize((320, 180), resample=Image.BILINEAR)
    color_resized = np.array(color_resized)

    if i==0:
      # CNOS Mask detection
      detection = np.load(f"{args.test_scene_dir}/cnos_output/cnos_results/detection.npz")
      mask = detection["segmentation"][0] # shape from (1, 360, 640) to (360, 640)

      assert mask is not None
      logging.info("Mask loaded successfully.")

      # plt.imshow(predicted_mask, cmap='gray')
      # plt.title("Mask")
      # plt.axis('off')
      # plt.show()

      pose = est.register(K=K, rgb=cv2.cvtColor(color,cv2.COLOR_RGB2BGR), depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      
      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

  # for i in range(len(reader.color_files)):
    # logging.info(f'i:{i}')
    # color = reader.get_color(i)
    # depth = reader.get_depth(i)
    # if i==0:
    #   mask = reader.get_mask(0).astype(bool)
    #   pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    else:
      pose = est.track_one(rgb=cv2.cvtColor(color,cv2.COLOR_RGB2BGR), depth=depth, K=K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/pose_{i}.txt', pose.reshape(4,4))


    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=2, transparency=0, is_input_rgb=False)
      # cv2.imshow('1', vis[...,::-1])
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      # plt.imshow(vis[..., ::-1])
      # plt.axis('off') 
      # plt.show()

    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/pose_{i}.png', vis[..., ::-1])
    i = i + 1

