## Visualize the training data, including point clouds with texture/ points clouds with sematic label/ 
## groundtruth volume/ and fused result for all above

from dvis import dvis
import open3d as o3d
import numpy as np
import struct
import yaml
from tqdm import tqdm
import argparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--iffusion', type=bool, default=False, help='if the data need to be fused')
    parser.add_argument('--startframe', type=int, default=0, help='Start frame to visualization')
    parser.add_argument('--endframe', type=int, default=1, help='End frame to visualization')
    parser.add_argument('--seqpath', type=str, default='dataset/sequences/00/', help='choose the path of the sequence to visualize')
    return parser.parse_args()

def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1
  return uncompressed

def read_poses(path):
    poses = []
    f = open(path)
    lines = f.readlines()
    for line in lines:
        linesplit = line.split()
        pose = np.array([[float(linesplit[0]),float(linesplit[1]),float(linesplit[2]), float(linesplit[3])],
                [float(linesplit[4]),float(linesplit[5]),float(linesplit[6]),float(linesplit[7])],
                [float(linesplit[8]),float(linesplit[9]),float(linesplit[10]),float(linesplit[11])],
                [0.0,0.0,0.0,1.0]])
        poses.append(pose)
    return poses

def read_bin(path):
    pc_list = []
    with open(path,'rb') as f:
        content = f.read()
        points = struct.iter_unpack('ffff',content)
        for point in points:
            pc_list.append([point[0], point[1],point[2]])
    return np.asarray(pc_list, dtype = np.float32)


def read_label(path):
    label_list = []
    with open(path,'rb') as f:
        content = f.read()
        labels = struct.iter_unpack('I',content)
        for label in labels:
            label_list.append(label[0])
    return label_list

def read_calib(path):
    calibs = []
    f = open(path)
    lines = f.readlines()
    for line in lines:
        linesplit = line[4:].split()
        calib = np.array([[float(linesplit[0]),float(linesplit[1]),float(linesplit[2]), float(linesplit[3])],
                [float(linesplit[4]),float(linesplit[5]),float(linesplit[6]),float(linesplit[7])],
                [float(linesplit[8]),float(linesplit[9]),float(linesplit[10]),float(linesplit[11])]])
        calibs.append(calib)
    return calibs

def idx2str(idx):
    if idx<10:
        return '00000'+str(idx)
    elif idx<100:
        return '0000'+str(idx)
    elif idx<1000:
        return '000'+str(idx)
    else:
        return '00'+str(idx)

def dot(transform, pts):
    if pts.shape[1] == 3:
        pts = np.concatenate([pts,np.ones((len(pts),1))],1)
    return (transform @ pts.T).T

def add_noise_to_point_cloud(point_cloud, variance=0.01, steps=1, repeat=True):
    # Separate the XYZ and RGB parts
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:]

    # Initialize the new point cloud with 10 times the number of points
    if repeat:
        xyz = np.repeat(xyz, 2, axis=0)
        rgb = np.repeat(rgb, 2, axis=0)

    # Add Gaussian noise to the XYZ coordinates
    for step in tqdm(range(steps)):
        noise = np.random.normal(0, np.sqrt(variance), xyz.shape)
        xyz += noise

    # Combine the noisy XYZ coordinates with the original RGB values
    noisy_point_cloud = np.hstack((xyz, rgb))
    
    return noisy_point_cloud

def semantic_propagation(pts, semantic_pts):
    # Separate XYZ and RGB parts
    xyz_a = pts[:, :3]
    xyz_b = semantic_pts[:, :3]
    rgb_b = semantic_pts[:, 3:]

    # Build a KD-tree for point cloud B
    tree = cKDTree(xyz_b)

    # Find the nearest neighbors in B for each point in A
    distances, indices = tree.query(xyz_a)

    # Get the RGB values of the nearest neighbors
    rgb_a = rgb_b[indices]

    # Combine the XYZ coordinates of A with the RGB values from B
    semantic_cloud = np.hstack((xyz_a, rgb_a))
    
    return semantic_cloud

def flip_y(pts):
    pts[:, 1] = -pts[:, 1]
    return pts

# Function to compute unidirectional Chamfer Distance
def chamfer_distance(source_points, target_points):
    source_kdtree = cKDTree(source_points)
    dist, _ = source_kdtree.query(target_points)
    return dist

# Function to compute color codes based on errors with adjusted colormap
def compute_color_codes(errors, colormap='jet'):
    norm_errors = (errors - errors.min()) / ((errors.max() - errors.min()) / 7)  # Adjusted to map only the first 1/3 of the range
    norm_errors = np.clip(norm_errors, 0, 1)  # Ensure the values are within [0, 1]
    cmap = plt.get_cmap(colormap)
    colors = cmap(norm_errors)
    return colors[:, :3]  # Return RGB values

# Main function to process point clouds and return results
def process_point_clouds(source_points, target_points):
    # Compute Chamfer Distance errors
    errors = chamfer_distance(source_points, target_points)
    # Compute color codes based on errors
    color_codes = compute_color_codes(errors)
    # Combine original points with color codes
    result = np.hstack((target_points, color_codes))
    return result

# Function to plot colorbar with adjusted ticks
def plot_colorbar(colormap='jet', errors=None):
    fig, ax = plt.subplots(figsize=(1, 6))  # Adjusted for a thinner colorbar

    # Normalize intensity range
    norm = plt.Normalize(vmin=0, vmax=1)
    # Create colorbar
    cmap = plt.get_cmap(colormap)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)

    # Add labels and title
    cb.set_label('Intensity')
    cb.ax.set_title('Color Bar')

    # Adjust ticks to reflect original error values
    if errors is not None:
        ticks = np.linspace(0, 1, num=5)
        tick_labels = (ticks * ((errors.max() - errors.min()) / 7)) + errors.min()
        cb.set_ticks(ticks)
        cb.set_ticklabels(["{:.2f}".format(label) for label in tick_labels])

    # Show and save the color bar
    plt.savefig("jet_colorbar_adjusted.png")
    # plt.show()

def main():
    args = parse_args()
    velo2cam = np.array([[7.533745000000e-03,-9.999714000000e-01,-6.166020000000e-04,-4.069766000000e-03],
                        [1.480249000000e-02,7.280733000000e-04,-9.998902000000e-01,-7.631618000000e-02],
                        [9.998621000000e-01,7.523790000000e-03,1.480755000000e-02,-2.717806000000e-01],
                        [0,0,0,1]])          
    sem_pallete = yaml.safe_load(open('ws_helin/semantic-kitti.yaml', 'r'))["color_map"]
    path = args.seqpath
    path_poses = path + 'poses.txt'
    poses = read_poses(path_poses)
    sem_colors_multiframs = np.array([[0,0,0]])
    pts_multiframs = np.array([[0,0,0]])
    startframe = args.startframe
    endframe = args.endframe

    ## diff pts
    diff_path = "SemantickittiVisTool/000123_diff.ply"
    diff_pts = o3d.io.read_point_cloud(diff_path)
    diff_pts_array = np.asarray(diff_pts.points)

    ## refine pts
    refine_path = "SemantickittiVisTool/000123_refine.ply"
    refine_pts = o3d.io.read_point_cloud(refine_path)
    refine_pts_array = np.asarray(refine_pts.points)


    for i in tqdm(range(startframe, endframe)):
        path_pts = path + 'velodyne/' + idx2str(i) +'.bin'
        path_labels = path + 'labels/' + idx2str(i) +'.label' 

        # point cloud
        input_pts = read_bin(path_pts)
        pose_velo = np.linalg.inv(velo2cam).dot(np.linalg.inv(poses[123])).dot(poses[i]).dot(velo2cam)
        input_pts_global = dot(pose_velo, input_pts)[:,0:3]

        # semantic pallet
        labels = read_label(path_labels)
        semcolors = np.zeros([len(input_pts),3])
        for idx, pointlabel in enumerate(labels):
            semantic_pointlabel = pointlabel & 0xFF #take out lower 16 bits
            if semantic_pointlabel in sem_pallete:
                semcolors[idx] = np.asarray(sem_pallete[semantic_pointlabel][::-1])
            else:
                semcolors[idx] = np.array([0,0,0])

        if i==123:
            input_sem_pts = np.concatenate([input_pts_global, semcolors],1)
            # dvis(flip_y(input_pts_global),l=1,t=1, vs=0.1,name='input_pts'+str(i))
            # dvis(flip_y(input_sem_pts), l=2, t=1,vs=0.1,name='input_sem_pts'+str(i))

            # ## add noisy
            # noisy1_sem_pts = add_noise_to_point_cloud(input_sem_pts, variance=0.1, steps=5)
            # dvis(flip_y(noisy1_sem_pts),l=1,t=2,vs=0.1,name='noisy1 semantic scan'+str(i))
            # noisy2_sem_pts = add_noise_to_point_cloud(input_sem_pts, variance=0.1, steps=30)
            # dvis(flip_y(noisy2_sem_pts),l=2,t=2,vs=0.1,name='noisy2 semantic scan'+str(i))
            # noisy3_sem_pts = add_noise_to_point_cloud(input_sem_pts, variance=0.1, steps=100)
            # dvis(flip_y(noisy3_sem_pts),l=3,t=2,vs=0.1,name='noisy3 semantic scan'+str(i))

        sem_colors_multiframs = np.concatenate([sem_colors_multiframs, semcolors],0)
        pts_multiframs = np.concatenate([pts_multiframs, input_pts_global],0)

    
    
    #filter
    distance_filter = np.sqrt(pts_multiframs[:, 0]**2 + pts_multiframs[:, 1]**2) < 50
    filtered_pts_multiframs = pts_multiframs[distance_filter]
    filtered_sem_colors_multiframs = sem_colors_multiframs[distance_filter]

    gt_sem_pts = np.concatenate([filtered_pts_multiframs, filtered_sem_colors_multiframs],1)
    dvis(flip_y(gt_sem_pts), l=1,t=0,vs=0.1, ms=1000000,name='gt semantic pts')
    # denoisy1_gt = add_noise_to_point_cloud(gt_sem_pts, variance=0.1, steps=3, repeat=False)
    # dvis(denoisy1_gt, l=2,t=0,vs=0.1, ms=1000000,name='gt noisy1')
    # denoisy2_gt = add_noise_to_point_cloud(gt_sem_pts, variance=0.1, steps=10, repeat=False)
    # dvis(denoisy2_gt, l=3,t=0,vs=0.1, ms=1000000,name='gt noisy2')
    # denoisy3_gt = add_noise_to_point_cloud(gt_sem_pts, variance=0.1, steps=50, repeat=False)
    # dvis(denoisy3_gt, l=4,t=0,vs=0.1, ms=1000000,name='gt noisy3')
    denoisy4_gt = add_noise_to_point_cloud(gt_sem_pts, variance=0.1, steps=500, repeat=False)
    dvis(denoisy4_gt, l=5,t=0,vs=0.1, ms=1000000,name='gt noisy4')


    # result = process_point_clouds(filtered_pts_multiframs, refine_pts_array)
    # dvis(flip_y(result), l=4,t=3,vs=0.1, ms=1000000,name='error intensity')
    # errors = chamfer_distance(filtered_pts_multiframs, refine_pts_array)
    # plot_colorbar('jet', errors)

    # ## semantic propagation
    # diff_sem_pts = semantic_propagation(diff_pts_array, gt_sem_pts)
    # dvis(flip_y(diff_sem_pts), l=4,t=1,vs=0.1, ms=1000000,name='diff semantic pts')
    # refine_sem_pts = semantic_propagation(refine_pts_array, gt_sem_pts)
    # dvis(flip_y(refine_sem_pts), l=5,t=1,vs=0.1, ms=1000000,name='refine semantic pts')

    # ## add noisy
    # denoisy1_sem_pts = add_noise_to_point_cloud(diff_sem_pts, variance=0.1, steps=3, repeat=False)
    # dvis(flip_y(denoisy1_sem_pts),l=1,t=3,vs=0.1,name='denoisy1 semantic scan'+str(i))
    # denoisy2_sem_pts = add_noise_to_point_cloud(diff_sem_pts, variance=0.1, steps=10,repeat=False)
    # dvis(flip_y(denoisy2_sem_pts),l=2,t=3,vs=0.1,name='denoisy2 semantic scan'+str(i))
    # denoisy3_sem_pts = add_noise_to_point_cloud(diff_sem_pts, variance=0.1, steps=50,repeat=False)
    # dvis(flip_y(denoisy3_sem_pts),l=3,t=3,vs=0.1,name='denoisy3 semantic scan'+str(i))
    

if __name__=="__main__":
    main()