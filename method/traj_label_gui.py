import msgpack
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from mpl_toolkits import mplot3d
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import rdp
import os
import cv2
from autolab_core import RigidTransform
import math
import matplotlib.image as mpimg
from matplotlib.widgets import Button
from tqdm import tqdm
from scipy import interpolate

def angle(dir):
    dir2 = dir[1:]
    dir1 = dir[:-1]
    return np.arccos((dir1*dir2).sum(axis=1)/(
        np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

def nn_line(xx,yy):
    # points=np.c_[x,y]
    # clf = NearestNeighbors(2).fit(points)
    # G = clf.kneighbors_graph()
    # T = nx.from_scipy_sparse_matrix(G)
    # paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
    # mindist = np.inf
    # minidx = 0

    # for i in range(len(points)):
    #     p = paths[i]           # order of nodes
    #     ordered = points[p]    # ordered nodes
    #     # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
    #     cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
    #     if cost < mindist:
    #         mindist = cost
    #         minidx = i
    # opt_order = paths[minidx]
    # xx = x[opt_order]
    # yy = y[opt_order]
    points = np.vstack((xx,yy)).T
    tolerance = 1
    min_angle = np.pi*0.22
    simplified = np.array(rdp.rdp(points.tolist(), tolerance))
    sx, sy = simplified.T
    directions = np.diff(simplified, axis=0)
    theta = angle(directions)
    idx = np.where(theta>min_angle)[0]+1
    org_idx=[]
    for i in range(idx.size):    
        mindist = np.inf
        minidx=0
        for j in range(xx.size):
            d=math.dist([sx[idx[i]],sy[idx[i]]],[xx[j],yy[j]])
            if (d<mindist):
                mindist=d
                minidx=j
        org_idx.append(minidx)
    return xx,yy,org_idx

# def get_kp(msg_path):
#     with open(msg_path, "rb") as f:
#         u = msgpack.Unpacker(f)
#         msg = u.unpack()

#     keyfrms = msg["keyframes"]

#     keyfrm_points = []
#     for keyfrm in keyfrms.values():
#         # get conversion from camera to world
#         trans_cw = np.matrix(keyfrm["trans_cw"]).T
#         rot_cw = R.from_quat(keyfrm["rot_cw"]).as_matrix()
#         # compute conversion from world to camera
#         rot_wc = rot_cw.T
#         trans_wc = - rot_wc * trans_cw
#         keyfrm_points.append((trans_wc[0, 0], trans_wc[1, 0], trans_wc[2, 0]))
#     keyfrm_points = np.array(keyfrm_points)
#     keyfrm_points = np.delete(keyfrm_points, 1, 1)
#     return keyfrm_points

def key_img(msg_path, video_path):

    # Read file as binary and unpack data using MessagePack library
    with open(msg_path, "rb") as f:
        u = msgpack.Unpacker(f)
        msg = u.unpack()

    # The point data is tagged "landmarks"
    key_frames = msg["keyframes"]

    print("Point cloud has {} points.".format(len(key_frames)))

    key_frame = {int(k): v for k, v in key_frames.items()}
    
    video_name = video_path.split("/")[-1][:-4]
    if not os.path.exists(video_name):
        os.mkdir(video_name)

    vidcap = cv2.VideoCapture(video_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS)) + 1
    count = 0

    tss=[]
    keyfrm_points=[]
    for key in sorted(key_frame.keys()):
        point = key_frame[key]

        # position capture
        trans_cw = np.matrix(point["trans_cw"]).T
        rot_cw = R.from_quat(point["rot_cw"]).as_matrix()

        rot_wc = rot_cw.T
        trans_wc = - rot_wc * trans_cw
        keyfrm_points.append((trans_wc[0, 0], trans_wc[1, 0], trans_wc[2, 0]))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, fps * float(point["ts"]))
        tss.append(point["ts"])

        # image capture
        success, image = vidcap.read()

        if not success:
            print("capture failed")
        else:
            cv2.imwrite(os.path.join(video_name, str(count) +".jpg"), image)
        count+=1
    keyfrm_points = np.array(keyfrm_points)
    keyfrm_points = np.delete(keyfrm_points, 1, 1)
    return keyfrm_points,tss

mp1="/home/diwei/cv/stella_vslam/build/0706a_wtc.msg"
mp2="/home/diwei/cv/stella_vslam/build/0706b_wtc.msg"
vp1="/home/diwei/cv/video/dataset_wtc_new/0706_wtc_134923_a.mp4"
vp2="/home/diwei/cv/video/dataset_wtc_new/0706_wtc_160107_b.mp4"
kp1,ts1=key_img(mp1,vp1)
kp2,ts2=key_img(mp2,vp2)
x1,y1,id1=nn_line(kp1[:,0],kp1[:,1])
x2,y2,id2=nn_line(kp2[:,0],kp2[:,1])
new_id2=[]
for i in range(len(id1)):
    t=id1[i]/float(x1.size)*float(x2.size)
    new_id2.append(min(id2, key=lambda x:abs(x-t)))
id2=new_id2
if len(id1)==0 and len(id2)==0:
    id1.append(len(kp1)-3)
    id2.append(len(kp2)-3)
print(id1,id2)
fig=plt.figure(figsize=(20,8))
gs=fig.add_gridspec(3,6)
ax1=fig.add_subplot(gs[:2,:3])
ax2=fig.add_subplot(gs[:2,3:])
ax3=fig.add_subplot(gs[2,0])
ax4=fig.add_subplot(gs[2,1])
ax5=fig.add_subplot(gs[2,2])
ax6=fig.add_subplot(gs[2,3])
ax7=fig.add_subplot(gs[2,4])
ax8=fig.add_subplot(gs[2,5])
ax=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
ax[0].plot(x1, y1, 'b-', label='path')
ax[0].plot(x1[id1], y1[id1], 'bo', markersize = 10, label='turning points')
ax[0].plot(x1[id1[0]], y1[id1[0]], 'go', markersize = 10, label='turning points')
ax[0].set_title("Trajectory 1")
ax[0].axis('equal')
ax[1].plot(x2, y2, 'r-', label='path')
ax[1].plot(x2[id2], y2[id2], 'ro', markersize = 10, label='turning points')
ax[1].plot(x2[id2[0]], y2[id2[0]], 'go', markersize = 10, label='turning points')
ax[1].set_title("Trajectory 2")
ax[1].axis('equal')

cp=os.path.dirname(os.path.abspath(__file__))
img = mpimg.imread(cp+'/'+vp1.split('/')[-1][:-4]+'/'+str(id1[0])+'.jpg')
ax[2].axis("off")
ax[2].set_title("Trajectory 1 Image")
ax[2].imshow(img)
st=max(id2[0]-2,0)
for i in range(5):
    img = mpimg.imread(cp+'/'+vp2.split('/')[-1][:-4]+'/'+str(st+i)+'.jpg')
    ax[3+i].axis("off")
    ax[3+i].imshow(img)
ind=0
flag=0

def next(event):
    global st
    st = min(st+5,x2.size)
    for i in range(5):
        img = mpimg.imread(cp+'/'+vp2.split('/')[-1][:-4]+'/'+str(st+i)+'.jpg')
        ax[3+i].clear()
        ax[3+i].axis("off")
        ax[3+i].imshow(img)
    plt.draw()

def prev(event):
    global st
    st = max(st-5,0)
    for i in range(5):
        img = mpimg.imread(cp+'/'+vp2.split('/')[-1][:-4]+'/'+str(st+i)+'.jpg')
        ax[3+i].clear()
        ax[3+i].axis("off")
        ax[3+i].imshow(img)
    plt.draw()
   
def write_img(st,ed,nimg,fps,vn,vc,count):
    nt=float(ed-st)/float(nimg)
    for i in range(nimg):
        vc.set(cv2.CAP_PROP_POS_FRAMES,fps*(st+i*nt))
        success, image = vc.read()
        if not success:
            print("capture failed")
        else:
            cv2.imwrite(os.path.join(vn, str(count) +".jpg"), image)
        count+=1

def trans_mat(primary,secondary):
    n = primary.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(primary)
    Y = pad(secondary)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))
    return transform

def onclick_select(event):
    global st,ind,flag,id1,id2
    if ind>=len(id2):
        print("label ended")
        return
    if event.inaxes == ax[3]:
        id2[ind]=st
        ind+=1
    elif event.inaxes == ax[4]:
        id2[ind]=st+1
        ind+=1
    elif event.inaxes == ax[5]:
        id2[ind]=st+2
        ind+=1
    elif event.inaxes == ax[6]:
        id2[ind]=st+3
        ind+=1
    elif event.inaxes == ax[7]:
        id2[ind]=st+4
        ind+=1
    else:
        return
    if ind>=len(id2):
        if flag==0:
            flag=1
            print(id1,id2)
            # id1=[13, 27, 35, 50, 60, 76, 90, 101, 113, 157, 176, 191, 215, 237, 247, 262, 276]
            # id2=[13, 29, 38, 54, 66, 82, 98, 109, 122, 163, 182, 197, 215, 235, 247, 261, 275]
            fw1=open("coordinate1.txt","w")
            fw2=open("coordinate2.txt","w")
            ff=open("test.txt","w")
            ff.write(str(x1)+'\n')
            ff.write(str(y1)+'\n')
            ff.write(str(x2)+'\n')
            ff.write(str(y2)+'\n')
            print("complete")
            vc1=cv2.VideoCapture(vp1)
            vc2=cv2.VideoCapture(vp2)
            fps1=int(vc1.get(cv2.CAP_PROP_FPS)) + 1
            fps2=int(vc2.get(cv2.CAP_PROP_FPS)) + 1
            ct=0
            id1.append(x1.size-1)
            id2.append(x2.size-1)
            pxi1=[]
            pyi1=[]
            pxi2=[]
            pyi2=[]
            ax[0].clear()
            ax[0].set_title("Trajectory pairs")
            ax[0].axis('equal')
            sec=[]
            pri=[]
            for i in range(len(id2)):
                if i==0:
                    ids=0
                    st1=0
                    st2=0
                    xf1=x1[:id1[i]]
                    yf1=y1[:id1[i]]
                    xf2=x2[:id2[i]]
                    yf2=y2[:id2[i]]
                    sec.append([x1[0],y1[0],0.])
                    pri.append([x2[0],y2[0],0.])
                else:
                    ids=id1[i-1]
                    st1=ts1[id1[i-1]]
                    st2=ts2[id2[i-1]]
                    xf1=x1[id1[i-1]:id1[i]]
                    yf1=y1[id1[i-1]:id1[i]]
                    xf2=x2[id2[i-1]:id2[i]]
                    yf2=y2[id2[i-1]:id2[i]]
                    sec.append([x1[id1[i-1]],y1[id1[i-1]],0.])
                    pri.append([x2[id2[i-1]],y2[id2[i-1]],0.])
            sec.append([x1[x1.size-1],y1[y1.size-1],0.])
            pri.append([x2[x2.size-1],y2[y2.size-1],0.])
            tm=trans_mat(np.array(pri),np.array(sec))
            for i in range(len(id2)):
                if i==0:
                    ids=0
                    st1=0
                    st2=0
                    xf1=x1[:id1[i]]
                    yf1=y1[:id1[i]]
                    xf2=x2[:id2[i]]
                    yf2=y2[:id2[i]]
                else:
                    ids=id1[i-1]
                    st1=ts1[id1[i-1]]
                    st2=ts2[id2[i-1]]
                    xf1=x1[id1[i-1]:id1[i]]
                    yf1=y1[id1[i-1]:id1[i]]
                    xf2=x2[id2[i-1]:id2[i]]
                    yf2=y2[id2[i-1]:id2[i]]
                ed1=ts1[id1[i]]
                ed2=ts2[id2[i]]
                nimg=int(min(ed1-st1,ed2-st2)*2)
                tck, u =interpolate.splprep([xf1,yf1],s=0)
                xi1, yi1 = interpolate.splev(np.linspace(0, 1, nimg), tck)
                # shift_x=xf1[0]-xf2[0]
                # shift_y=yf1[0]-yf2[0]
                # xf2=xf2+shift_x
                # yf2=yf2+shift_y
                # x_scale = (xf1[-1] - xf2[0]) / (xf2[-1] - xf2[0])
                # y_scale = (yf1[-1] - yf2[0]) / (yf2[-1] - yf2[0])
                # new_x = (xf2 - xf2[0]) * x_scale + xf2[0]
                # new_y = (yf2 - yf2[0]) * y_scale + yf2[0]
                tck, u =interpolate.splprep([xf2,yf2],s=0)
                xi2, yi2 = interpolate.splev(np.linspace(0, 1, nimg), tck)
                c2=[]
                for i in range(len(xi2)):
                    c2.append([xi2[i],yi2[i],0.])
                t=tm(np.array(c2))
                xi2=t[:,0]
                yi2=t[:,1]
                # ax[0].plot(xi1,yi1,'ro',markersize=1)
                # ax[0].plot(xi2,yi2,'bo',markersize=1)
                # plt.draw()
                # plt.pause(0.001)
                # text = input("Is the current pair of segments correct? (y or n)\n")
                # if text=="n":
                #     xi2=xi1
                #     yi2=yi1
                pxi1.extend(xi1.tolist())
                pyi1.extend(yi1.tolist())
                pxi2.extend(xi2.tolist())
                pyi2.extend(yi2.tolist())
                ax[0].clear()
                ax[0].set_title("Trajectory pairs")
                ax[0].plot(pxi1,pyi1,'r-',markersize=1)
                ax[0].plot(pxi2,pyi2,'b-',markersize=1)
                ax[0].axis('equal')
                plt.draw()
                for j in range(nimg):
                    fw1.write(str(xi1[j])+' '+str(yi1[j])+'\n')
                for j in range(nimg):
                    fw2.write(str(xi2[j])+' '+str(yi2[j])+'\n')
                write_img(st1,ed1,nimg,fps1,vp1.split('/')[-1][:-4],vc1,ct)
                write_img(st2,ed2,nimg,fps2,vp2.split('/')[-1][:-4],vc2,ct)
                ct+=nimg

        print("label ended")
        return
    ax[0].clear()
    ax[1].clear()
    ax[0].set_title("Trajectory 1")
    ax[1].set_title("Trajectory 2")

    ax[0].plot(x1, y1, 'b-')
    ax[0].plot(x1[id1], y1[id1], 'bo', markersize = 10)
    for i in range(ind):
        ax[0].plot(x1[id1[i]], y1[id1[i]], 'go', markersize = 10)
    ax[0].plot(x1[id1[ind]], y1[id1[ind]], 'mo', markersize = 10)
    ax[0].axis('equal')

    ax[1].plot(x2,y2,'r-')
    ax[1].plot(x2[id2], y2[id2], 'ro', markersize = 10)
    for i in range(ind):
        ax[1].plot(x2[id2[i]], y2[id2[i]], 'go', markersize = 10)
    ax[1].plot(x2[id2[ind]], y2[id2[ind]], 'mo', markersize = 10)
    ax[1].axis('equal')

    cp=os.path.dirname(os.path.abspath(__file__))
    img = mpimg.imread(cp+'/'+vp1.split('/')[-1][:-4]+'/'+str(id1[ind])+'.jpg')
    ax[2].clear()
    ax[2].set_title("Trajectory 1 Image")
    ax[2].axis("off")
    ax[2].imshow(img)
    st=max(id2[ind]-2,0)
    for i in range(5):
        img = mpimg.imread(cp+'/'+vp2.split('/')[-1][:-4]+'/'+str(st+i)+'.jpg')
        ax[3+i].clear()
        ax[3+i].axis("off")
        ax[3+i].imshow(img)
    plt.draw()
    

fig.canvas.mpl_connect("button_press_event",onclick_select)
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(prev)
plt.show()