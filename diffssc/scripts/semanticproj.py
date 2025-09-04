## Visualize the training data, including point clouds with texture/ points clouds with sematic label/ 
## groundtruth volume/ and fused result for all above

# from dvis import dvis
import numpy as np
import struct
import yaml
from tqdm import tqdm
from matplotlib import image 
import argparse
from PIL import Image

THRESHOULDH = 4
THRESHOULDW = 2

SEQ = ['00','01','02','03','04','05','06','07','08','09','10']

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--startframe', type=int, default=0, help='Start frame to visualization')
    parser.add_argument('--endframe', type=int, default=10, help='End frame to visualization')
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

def main():
    args = parse_args()
    velotocam0 = np.array([[7.533745000000e-03,-9.999714000000e-01,-6.166020000000e-04,-4.069766000000e-03],
                        [1.480249000000e-02,7.280733000000e-04,-9.998902000000e-01,-7.631618000000e-02],
                        [9.998621000000e-01,7.523790000000e-03,1.480755000000e-02,-2.717806000000e-01],
                        [0,0,0,1]])
    cam2tocam0 = np.array([[1,0,0,-0.06],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
    cam3tocam0 = np.array([[1,0,0,0.48],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
    R0_rect = np.array([[9.999239000000e-01,9.837760000000e-03,-7.445048000000e-03,0],
                        [-9.869795000000e-03,9.999421000000e-01,-4.278459000000e-03,0],
                        [7.402527000000e-03,4.351614000000e-03,9.999631000000e-01,0],
                        [0,0,0,1]])
    sem_pallete = yaml.safe_load(open('ws_helin/semantic-kitti.yaml', 'r'))["color_map"]
    imgh, imgw, _ = image.imread('./dataset/sequences/00/image_2/000000.png').shape

    for seq in SEQ:
        path = 'dataset/sequences/'+seq+'/'
        path_poses = path + 'poses.txt'
        path_calib = path + 'calib.txt'
        poses = read_poses(path_poses)
        p2 = read_calib(path_calib)[2]
        imgsemantic = np.zeros([imgh,imgw,3], dtype = int)
        startframe = 0
        endframe = len(poses)-2
        
        for i in tqdm(range(startframe, endframe,5)):
            pos_scans_velo = np.array([[0,0,0,0]])
            label_scans = np.array([0])
            sem_colors = np.array([[0,0,0]])
            imgsemanticmask = np.zeros([imgh,imgw,2], dtype = int)
            imgsemanticmask[:,:,1] = 1e6
            for j in range(i, i+6):
                # point cloud
                path_pts = path + 'velodyne/' + idx2str(j) +'.bin'
                currentscan = read_bin(path_pts)
                pose_velo = np.linalg.inv(velotocam0).dot(poses[j].dot(velotocam0))
                currentscan_velo = dot(pose_velo, currentscan)
                pos_scans_velo = np.concatenate([pos_scans_velo, currentscan_velo],0)
                # label
                path_labels = path + 'labels/' + idx2str(j) +'.label'  
                currentlabel = np.asarray(read_label(path_labels))
                
                # visualize the point cloud
                currentsemcolor = np.zeros([len(currentscan),3])
                for idx, pointlabel in enumerate(currentlabel):
                    currentlabel[idx] = pointlabel & 0xFF #take out lower 16 bits             
                        
                #     if currentlabel[idx] in sem_pallete:
                #         currentsemcolor[idx] = np.asarray(sem_pallete[currentlabel[idx]][::-1]) #reverse the list from bgr to rgb
                #     else:
                #         currentsemcolor[idx] = np.array([0,0,0])
                # sem_colors = np.concatenate([sem_colors, currentsemcolor],0)
                label_scans = np.concatenate([label_scans, currentlabel])
            # dvis(np.concatenate([pos_scans_velo[:,:3], sem_colors],1),l=2,t =i,vs=0.10, ms=1000000,name='pts/fused semantic pts')  
            # for the dynamic object, only take the current scan
            pos_scans_velo = np.concatenate([pos_scans_velo[:len(currentlabel)],pos_scans_velo[len(currentlabel):][label_scans[len(currentlabel):]<100]],0)
            label_scans = np.concatenate([label_scans[:len(currentlabel)],label_scans[len(currentlabel):][label_scans[len(currentlabel):]<100]])

            pose_cam2 = np.linalg.inv(cam2tocam0).dot(np.linalg.inv(poses[i]).dot(velotocam0))
            pos_scans_cam = dot(pose_cam2, pos_scans_velo)
            label_scans = label_scans[pos_scans_cam[:,2]>0]
            pos_scans_cam = pos_scans_cam[pos_scans_cam[:,2]>0]
            pos_img = p2 @ R0_rect @ pos_scans_cam.T
            pos_img[:2] /= pos_img[2,:]
            pos_img = pos_img.T
            
            # chop
            label_scans = label_scans[pos_img[:,0]>THRESHOULDW]
            pos_img = pos_img[pos_img[:,0]>THRESHOULDW]

            label_scans = label_scans[pos_img[:,0]<imgw-THRESHOULDW]
            pos_img = pos_img[pos_img[:,0]<imgw-THRESHOULDW]

            label_scans = label_scans[pos_img[:,1]>THRESHOULDH]
            pos_img = pos_img[pos_img[:,1]>THRESHOULDH]

            label_scans = label_scans[pos_img[:,1]<imgh-THRESHOULDH]
            pos_img = pos_img[pos_img[:,1]<imgh-THRESHOULDH]

            # sparse to dense
            for idx, posimg in enumerate(pos_img):
                if posimg[2] < imgsemanticmask[int(posimg[1]),int(posimg[0]),1] and label_scans[idx] in sem_pallete.keys():
                    imgsemanticmask[int(posimg[1])-THRESHOULDH:int(posimg[1])+THRESHOULDH,int(posimg[0])-THRESHOULDW:int(posimg[0])+THRESHOULDW,0] = label_scans[idx]
                    imgsemanticmask[int(posimg[1])-THRESHOULDH:int(posimg[1])+THRESHOULDH,int(posimg[0])-THRESHOULDW:int(posimg[0])+THRESHOULDW,1] = posimg[2]
            # save the mask array
            np.save('2dsemseg/'+seq+'/mask/'+idx2str(i)+'.npy',imgsemanticmask[:,:,0].astype(np.uint8))
            # write the img
            for u in range(imgw):
                for v in range(imgh):
                    imgsemantic[v,u,:] = sem_pallete[imgsemanticmask[v,u,0]][::-1] #reverse the list from bgr to rgb
            img = Image.fromarray(imgsemantic.astype(np.uint8)).convert('RGB')
            img.save('2dsemseg/'+seq+'/image/'+idx2str(i)+'.png')
    
if __name__=="__main__":
    main()