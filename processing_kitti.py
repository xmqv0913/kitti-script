import os
import random
import glob
from re import S
from typing_extensions import Self

import cv2 as cv
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from scan import LaserScan


def check_and_mkdir(path):
  if(not os.path.exists(path)):
    os.mkdir(path)


def excute_commond(cmd):
  print(cmd)
  os.system(cmd)


def copy_with_list(src_road_dir, src_depth_dir, dst_dir, name_list):
  check_and_mkdir(dst_dir)
  calib_dir = os.path.join(dst_dir, "calib")
  check_and_mkdir(calib_dir)
  depth_dir = os.path.join(dst_dir, "depth_u16")
  check_and_mkdir(depth_dir)
  gt_dir = os.path.join(dst_dir, "gt_image_2")
  check_and_mkdir(gt_dir)
  image_dir = os.path.join(dst_dir, "image_2")
  check_and_mkdir(image_dir)
  for name in name_list:
    excute_commond("cp {} {}".format(
        os.path.join(src_road_dir, "calib", name+".txt"),
        os.path.join(calib_dir, name+".txt")
    ))

    its = name.split("_")
    road_name = its[0]+"_road_"+its[1]+".png"
    lane_name = its[0]+"_lane_"+its[1]+".png"
    excute_commond("cp {} {}".format(
        os.path.join(src_road_dir, "gt_image_2", road_name),
        os.path.join(gt_dir, road_name)
    ))
    excute_commond("cp {} {}".format(
        os.path.join(src_road_dir, "gt_image_2", lane_name),
        os.path.join(gt_dir, lane_name)
    ))
    excute_commond("cp {} {}".format(
        os.path.join(src_road_dir, "image_2", name+".png"),
        os.path.join(image_dir, name+".png")
    ))
    excute_commond("cp {} {}".format(
        os.path.join(src_depth_dir, name+".png"),
        os.path.join(depth_dir, name+".png")
    ))


def copy_to_roadseg():
  data_road_dir = "/Users/xiaoming/dataset/RoadLaneDetectionEvaluation/data_road"
  data_depth_dir = "/Users/xiaoming/dataset/RoadLaneDetectionEvaluation/depth_u16"

  dst_dir = 'datasets/kitti'
  check_and_mkdir(dst_dir)

  # random sampling
  names = os.listdir(os.path.join(data_road_dir, "training", "image_2"))
  names = [name[:-4] for name in names]
  all_len = len(names)
  val_len = int(all_len*0.2)
  all_idx = range(all_len)
  val_idx = random.sample(all_idx, val_len)
  val_idx = set(val_idx)
  val_names = []
  train_names = []
  for i in all_idx:
    if i in val_idx:
      val_names.append(names[i])
    else:
      train_names.append(names[i])

  # copy to dst
  dst_train_dir = os.path.join(dst_dir, "traning")
  copy_with_list(os.path.join(data_road_dir, "training"), os.path.join(
      data_depth_dir, "training"), dst_train_dir, train_names)
  dst_val_dir = os.path.join(dst_dir, "validation")
  copy_with_list(os.path.join(data_road_dir, "training"), os.path.join(
      data_depth_dir, "training"), dst_val_dir, val_names)

  test_list = os.listdir(os.path.join(data_road_dir, "testing", "image_2"))
  test_list = [name[:-4] for name in test_list]
  dst_test_dir = os.path.join(dst_dir, "testing")
  copy_with_list(os.path.join(data_road_dir, "testing"), os.path.join(
      data_depth_dir, "testing"), dst_test_dir, test_list)

  print("done!")


def show_gt():
  # gt_dir = 'testresults/kitti/test_kitti'
  gt_dir = 'datasets/kitti/validation/gt_image_2'
  im2_dir = 'datasets/kitti/validation/image_2'
  out_dir = 'debug/show_test'
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  names = os.listdir(im2_dir)
  for idx, name in enumerate(names):
    print("{}/{}".format(idx, len(names)))
    if name != "um_000093.png":
      continue
    name_items = name.split('_')
    gt_name = '{}_road_{}'.format(name_items[0], name_items[1])
    gt_im = cv.imread(os.path.join(gt_dir, gt_name), cv.IMREAD_COLOR)
    gt_mask = gt_im[:, :, 0] > 128
    gt_im[:, :, 0] = 0
    gt_im[:, :, 1] = 0
    gt_im[:, :, 2][gt_mask] = 255
    im2 = cv.imread(os.path.join(im2_dir, name), cv.IMREAD_UNCHANGED)
    show_im = im2.copy()
    show_im[gt_mask] = 0.5*im2[gt_mask]+0.5*gt_im[gt_mask]
    cv.imwrite(os.path.join(out_dir, name), show_im)


def interpolation_depth():
  to_do_flag = 0



def load_cam(path):
  def _read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
      for line in f.readlines():
        key, value = line.split(':', 1)
        # The only non-float values in these files are dates, which
        # we don't care about anyway
        try:
          data[key] = np.array([float(x) for x in value.split()])
        except ValueError:
          pass
    return data

  rawdata = _read_calib_file(path)
  data = {}
  P0 = np.reshape(rawdata['P0'], (3, 4))
  P1 = np.reshape(rawdata['P1'], (3, 4))
  P2 = np.reshape(rawdata['P2'], (3, 4))
  P3 = np.reshape(rawdata['P3'], (3, 4))
  R0_rect = np.reshape(rawdata['R0_rect'], (3, 3))
  Tr_velo_to_cam = np.reshape(rawdata['Tr_velo_to_cam'], (3, 4))
  data['P0'] = P0
  data['P1'] = P1
  data['P2'] = P2
  data['P3'] = P3
  data['R0_rect'] = R0_rect
  data['Tr_velo_to_cam'] = Tr_velo_to_cam
  return data

def aug_44(m):
  m44 = np.identity(4)
  h,w = m.shape
  m44[:h, :w]=m
  return m44

def get_xyz_im(uvz, cam, invalid_mask):
  uv = uvz[:,:2]/uvz[:,2].reshape(-1,1)
  h,w= invalid_mask.shape
  mask=(uv[:,0]>=-100) & (uv[:,0]<w+100) & (uv[:,1]>=-100) & (uv[:,1]<h+100) & (uvz[:,2]>1.0) & (uvz[:,2]<65)
  uv=uv[mask]
  uvz=uvz[mask]
  cam_inv_t = np.transpose(np.linalg.inv(cam[:3,:3]))
  xyz = (uvz@cam_inv_t)
  xyz = np.clip(xyz, -70,70)
  vu = uv[:,::-1]
  tree = cKDTree(vu)
  grid_y, grid_x = np.mgrid[0:h,0:w]
  to_be_query = np.c_[grid_y.ravel(), grid_x.ravel()]
  _, idxs = tree.query(to_be_query, k=1)
  xyz_im = np.zeros((h,w,3), dtype=np.uint16)
  for i in range(to_be_query.shape[0]):
    xyz_im[to_be_query[i][0], to_be_query[i][1]]=(xyz[idxs[i]]+70)*450
  xyz_im[:,:,0][invalid_mask]=0
  xyz_im[:,:,1][invalid_mask]=0
  xyz_im[:,:,2][invalid_mask]=0
  return xyz_im

def get_scan_grad(bin_path:str, scaner: LaserScan):
  scaner.open_scan(bin_path)
  xyz_im = scaner.proj_xyz
  mask_im = scaner.proj_mask
  xyz = []
  left_z = []
  left_x =[]
  right_z = []
  right_x = []
  for scan_id in range(xyz_im.shape[0]):
    scan_xyz = xyz_im[scan_id]
    scan_mask = mask_im[scan_id]
    scan_xyz=scan_xyz[scan_mask]
    yaw = -np.arctan2(scan_xyz[:,1], scan_xyz[:,0])*180/np.pi # just for debug
    scan_len = scan_xyz.shape[0]
    half_win = 5
    scan_z = scan_xyz[:,2]
    scan_x = scan_xyz[:,0]
    z_left_grad = np.zeros(scan_len)
    z_right_grad = np.zeros(scan_len)
    x_left_grad = np.zeros(scan_len)
    x_right_grad = np.zeros(scan_len)
    idxs = np.arange(scan_len)
    for i in range(1, half_win+1):
      left_idx = (idxs-i)%scan_len
      right_idx = (idxs+i)%scan_len
      z_left_grad+=abs(scan_z[left_idx]-scan_z)/abs(yaw[left_idx]-yaw)
      x_left_grad+=abs(scan_x[left_idx]-scan_x)/abs(yaw[left_idx]-yaw)
      z_right_grad+=abs(scan_z[right_idx]-scan_z)/abs(yaw[right_idx]-yaw)
      x_right_grad+=abs(scan_x[right_idx]-scan_x)/abs(yaw[right_idx]-yaw)
    xyz.append(scan_xyz)
    left_z.append(z_left_grad)
    left_x.append(x_left_grad)
    right_z.append(z_right_grad)
    right_x.append(x_right_grad)
  if False:
  # just debug
    show_im = np.zeros(xyz_im.shape[:2])
    for row in range(xyz_im.shape[0]):
      show_im[row][mask_im[row]]=left_z[row]
    show_im = (show_im*10).astype(np.uint8)
    name = bin_path.split("/")[-1][:-3]
    debug_out_path = os.path.join("debug/show_grad/show_left_z", name+".png")
    cv.imwrite(debug_out_path, show_im)
  return xyz, left_z, left_x, right_z, right_x

  

def process_lidar(im2_dir, velodyne_dir, cam_dir, out_dir, pre_depth_dir):
  im2_names = os.listdir(im2_dir)
  names = [name[:-4] for name in im2_names]
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  out_depth_dir = os.path.join(out_dir, "depth1000")
  if not os.path.exists(out_depth_dir):
    os.mkdir(out_depth_dir)
  
  scaner = LaserScan(project=True, W=4096)

  for name in names:
    print("processing {}".format(name))
    bin_path = os.path.join(velodyne_dir,"{}.bin".format(name))
    xyz, left_z, left_x, right_z, right_x=get_scan_grad(bin_path, scaner)
    calib_path = os.path.join(cam_dir, "{}.txt".format(name))
    calib_data = load_cam(calib_path)
    trans = calib_data['P2']@ aug_44(calib_data['R0_rect']) @ aug_44(calib_data['Tr_velo_to_cam'])

    if False:
      pre_depth_path = os.path.join(pre_depth_dir, "{}.png".format(name))
      pre_depth = cv.imread(pre_depth_path, cv.IMREAD_ANYDEPTH)
      invalid_mask = (pre_depth==0)
      # We discard the lidar intensity information and reuse its address for storing homogeneous coordinates
      road_pt = np.fromfile(bin_path, dtype=np.float32).reshape(-1,4)
      road_pt[:,3]=1.0
      uvz=road_pt @ np.transpose(trans)
      xyz_im = get_xyz_im(uvz, calib_data['P2'],invalid_mask)
      out_xyz_path = os.path.join(out_dir, "{}.png".format(name))
      cv.imwrite(out_xyz_path, xyz_im)
      depth_u16 = (xyz_im[:,:,2]/450-70)*1000
      depth_u16 = depth_u16.astype(np.uint16)
      out_debug_path = out_xyz_path = os.path.join(out_dir, "{}_debugz.png".format(name))
      cv.imwrite(out_debug_path, depth_u16)
      continue
      uvz[:,:2]=uvz[:,:2]/uvz[:,2].reshape(-1,1)

      # interpolate and project to image
      im2_path = os.path.join(im2_dir, "{}.png".format(name))
      im2 = cv.imread(im2_path)
      h,w,_ = im2.shape
      mask=(uvz[:,0]>=0.001) & (uvz[:,0]<w-0.001) & (uvz[:,1]>=0.001) & (uvz[:,1]<h-0.001) & (uvz[:,2]>1.0) & (uvz[:,2]<65)
      uvz=uvz[mask]
      pts = uvz[:,:2][:,::-1]

      #for debug
      if True:
        debug_u16 = np.zeros((h,w), dtype=np.uint16)
        pts=pts.astype(np.uint16)
        for i in range(pts.shape[0]):
          debug_u16[pts[i][0], pts[i][1]]=int(uvz[i][2]*1000)
        out_depth_path = os.path.join(out_depth_dir, "{}_debug.png".format(name))
        cv.imwrite(out_depth_path, debug_u16)

      inter_type = 'linear'
      vals = 1000.0/uvz[:,2]
      grid_y, grid_x = np.mgrid[0:h,0:w]
      depth_inv = griddata(pts, vals, (grid_y, grid_x), inter_type, 0.0)
      mask = depth_inv>0.0001

      depth_u16 = np.zeros((h,w), dtype=np.uint16)
      depth_u16[mask] = (1000.0*1000.0/depth_inv[mask]).astype(np.uint16)
      
      depth_u16[invalid_mask]=0
      
      out_depth_path = os.path.join(out_depth_dir, "{}_{}.png".format(name, inter_type))
      cv.imwrite(out_depth_path, depth_u16)
      
      flag=0

def to_gray(root):
  paths = glob.glob(os.path.join(root, "*.png"))
  for path in paths:
    bev_prob = cv.imread(path, cv.IMREAD_GRAYSCALE)
    cv.imwrite(path, bev_prob)
  flag = 0


if __name__ == "__main__":
  #shit = cv.imread("/Users/xiaoming/dataset/RoadLaneDetectionEvaluation/data_road/testresults/pre_depth/test_240/baseline_bev_test/um_road_000017.png", cv.IMREAD_UNCHANGED)
  #to_gray("/Users/xiaoming/dataset/RoadLaneDetectionEvaluation/data_road/testresults/xyz/test_310/baseline_bev_test")
  #to_gray("/Users/xiaoming/dataset/RoadLaneDetectionEvaluation/data_road/testresults/pre_depth/test_240/baseline_bev_test")
  
  #show_gt()
  # interpolation_depth()
  if True:
    road_root = "/Users/xiaoming/dataset/RoadLaneDetectionEvaluation/data_road"
    velodyne_root = "/Users/xiaoming/dataset/RoadLaneDetectionEvaluation/data_road_velodyne"
    depth_root = "/Users/xiaoming/dataset/RoadLaneDetectionEvaluation/depth_u16/"
    item_name = "testing"
    pre_depth_dir = os.path.join(depth_root,item_name)
    out_dir = os.path.join(road_root,item_name, "xyz")
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)
    im2_dir = os.path.join(road_root, item_name,"image_2")
    velodyne_dir = os.path.join(velodyne_root, item_name, "velodyne")
    cam_dir = os.path.join(road_root, item_name, "calib")
    process_lidar(im2_dir, velodyne_dir, cam_dir, out_dir, pre_depth_dir)
