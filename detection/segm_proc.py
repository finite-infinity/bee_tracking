import numpy as np
from sklearn.decomposition import PCA
import math

SZ_MIN = 50
SZ_MAX = 1000

#def points_dst(prs, lbs):
#    res = np.zeros((len(prs), len(lbs)))
#    for ip in range(len(prs)):
#        for il in range(len(lbs)):
#            res[ip,il] = np.sqrt((prs[ip][0]-lbs[il][0])**2+(prs[ip][1]-lbs[il][1])**2)
#    return res
#
#def remove_el(v, i):
#    v1 = v[:i]
#    v1.extend(v[(i+1):])
#    return v1
#
#def find_labels_from_arr(labels_point, labels_segm):
#    lxs, lys = np.where(labels_point != -1)
#    lbs = []
#    for j in range(len(lxs)):
#        lbs.append(lxs[j], lys[j], labels_segm[lxs[j], lys[j]], labels_point[lxs[j],lys[j]]))
#    return lbs

#mark区域m中包含(i1,i2)的一连通区域
def mark_region(i1, i2, m, regions, nb):
    q = [(i1,i2)]
    while (len(q) > 0):  
        (i1,i2) = q.pop(0)  #删掉q中第一个元素，并赋值给(i1，i2) 
                            #一直给周边赋-1，中心赋nb，直到m中这一片连通区域都赋值为nb，边框为-1
        
        regions[i1,i2] = nb   #给个区域号
        
        #将四周区域为0的全赋值-1
        if (i1 > 0) and (regions[i1-1,i2] == 0) and m[i1-1,i2]:
            regions[i1-1,i2] = -1  
            q.append((i1-1, i2))
        if (i2 > 0) and (regions[i1,i2-1] == 0) and m[i1,i2-1]:
            regions[i1,i2-1] = -1
            q.append((i1, i2-1))
        if (i1 < m.shape[0]-1) and (regions[i1+1,i2] == 0) and m[i1+1,i2]:
            regions[i1+1,i2] = -1
            q.append((i1+1, i2))
        if (i2 < m.shape[1]-1) and (regions[i1,i2+1] == 0) and m[i1,i2+1]:
            regions[i1,i2+1] = -1
            q.append((i1, i2+1))
    return regions

#m是预测的区域，将m分为几片连通区域，返回区域数和区域集合（边界-1，区域内rg_nb）
def connected_components(m): # due to problems with opencv2
    rg_nb = 0 #区域号从1开始
    regions = np.zeros(m.shape)
    for i1 in range(m.shape[0]):  
        for i2 in range(m.shape[1]):
            if (regions[i1,i2] == 0) and m[i1,i2]:  
                rg_nb = rg_nb + 1  #mark区域rg_nb
                regions = mark_region(i1, i2, m, regions, rg_nb)
    return rg_nb+1, regions

#计算区域主轴：每个区域中点的第一主成分的角度（通过区域形状计算的）
def find_center(regions, rg):
    ys, xs = np.where(regions == rg)  #这里x,y是逆时针转了一下的（默认x轴为pi/2位置）
    sz = len(xs)
    res = (-1, -1, -1)
    if (sz > SZ_MIN) and (sz <= SZ_MAX):
        m = np.zeros((sz, 2))
        m[:,0] = -xs
        m[:,1] = ys
        pca = PCA(n_components=2)  #特征主成分分析，将m降维为2维（保留一个方向）
        pca.fit(m)
        #array,shape(n_components, n_features), 降维后各主成分方向，并按照方差值排序
        p = pca.components_  

        a = math.pi / 2
        if p[0,1] != 0:  #y轴有分量
            a = (math.atan(p[0,0]/p[0,1]) + math.pi) % math.pi  #反正切，算夹角
        res = (np.mean(xs), np.mean(ys), a)  #区域中心点坐标+主成分角度
    return res


def find_type(pred, regions, rg):
    v = pred[regions == rg]   #包含1/2/3
    res = float(np.sum(v == 2)) / np.prod(np.shape(v))  #v的元素中2所占比例（正样本）
    return res


def find_angle(pred, regions, rg):
    v = pred[regions == rg]  
    v = v[v >= 0]   #指定区域角预测值
    if v.size == 0: 
        return 0
    v[v > 1] = 1   #v看成概率
    return np.percentile(v, 99) * 2 * math.pi  #v的99%分位数*pi/2，得预测角


def extract_positions(pred_class, pred_angle):
    region_nbs, regions = connected_components(pred_class > 0)  #区域号，区域
    res = []
    for rg in range(1, region_nbs):
        (x, y, ax) = find_center(regions, rg)  #中心位置加主轴角度
        if ax != -1:   
            a = find_angle(pred_angle, regions, rg)  #预测角
            cl = find_type(pred_class, regions, rg)  #预测类（未经阈值判断）
            res.append((x, y, cl, a, ax))   #中心位置、预测类、预测角、主轴
    return res


def angle_diff(a1, a2):  #角a1、a2的差
    if a2 > a1:
        a1, a2 = a2, a1
    d = a1 - a2
    if d > 180:
        a1 = -(360 - a1)
        d = a2 - a1
    return d


