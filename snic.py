#%% Initialization
import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from time import time
from scipy.io import savemat

import utils


#%% Core Functions
def find_seeds(width, height, numk):
    sz = width * height
    gridstep = int(np.sqrt(sz / numk) + 0.5)
    halfstep = gridstep / 2
    xsteps = int(width / gridstep)
    ysteps = int(height / gridstep)
    # no fix for gridstep
    numk_new = xsteps * ysteps
    seeds = np.zeros((numk_new, 2), dtype=np.int) # row and col
    for y in range(0, ysteps):
        for x in range(0, xsteps):
            seeds[y * xsteps + x, 0] = halfstep + y * gridstep
            seeds[y * xsteps + x, 1] = halfstep + x * gridstep
    #print('gridstep: %d' % gridstep)
    return numk_new, seeds

def snic(img, numk, compactness, verbose=False):
    # heap log (d, k, i, j, pop(0)/push(1))
    f = open('snic.log', 'w')

    # CIELab
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # reduce range of lightness, or else boundaries would be wavy
    img[:,:,0] = img[:,:,0] * 100 / 255

    # constants
    h = img.shape[0]
    w = img.shape[1]
    numk, seeds = find_seeds(w, h, numk)
    sz = h * w
    di = [-1, 0, 1, 0, -1, 1, 1, -1]
    dj = [0, -1, 0, 1, -1, -1, 1, 1]
    connectivity = 4
    invwt = compactness * compactness * numk / sz

    # variants
    labels = -1 * np.ones((h, w), np.int)   # pixel labels
    kf = np.zeros((numk, 5), np.double)     # sp features: l,a,b,i,j
    ks = np.zeros(numk, np.int)             # sp sizes
    heap = []

    for k in range(numk):
        heap.append((0, k, seeds[k, 0], seeds[k, 1]))
    heapq.heapify(heap)
    while len(heap) > 0:
        d, k, i, j = heapq.heappop(heap)
        if verbose:
            f.write('%.3f,%d,%d,%d,%d\n' % (d, k, i, j, 0))
        if labels[i, j] < 0: # takes longer time if removed
            labels[i, j] = k
            l, a, b = img[i, j, :].tolist()
            kf[k, :] += l, a, b, i, j
            ks[k] += 1
            for n in range(connectivity):
                ii = i + di[n]
                jj = j + dj[n]
                if ii < 0 or jj < 0 or ii >= h or jj >= w:
                    continue
                if labels[ii, jj] < 0:
                    pf = (*(img[ii, jj, :].tolist()), ii, jj) # pixel features
                    dist = kf[k, :] - [f * ks[k] for f in pf]
                    dist = [d * d for d in dist]
                    colordist = sum(dist[0:3])
                    spacedist = sum(dist[3:5])
                    slicdist = (colordist + spacedist * invwt) / (ks[k] * ks[k])
                    heapq.heappush(heap, (slicdist, k, ii, jj))
                    if verbose:
                        f.write('%.3f,%d,%d,%d,%d\n' % (slicdist, k, ii, jj, 1))
    f.close()
    return labels

def snico(img, numk):
    # CIELab
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # reduce range of lightness, or else boundaries would be wavy
    img[:,:,0] = img[:,:,0] * 100 / 255

    # constants
    h = img.shape[0]
    w = img.shape[1]
    numk, seeds = find_seeds(w, h, numk)
    di = [-1, 0, 1, 0, -1, 1, 1, -1]
    dj = [0, -1, 0, 1, -1, -1, 1, 1]
    connectivity = 4
    area = h * w / numk
    DCN = 20 # default color nomalization factor

    maxcdsq = np.zeros(numk) # max color distances squared
    for iter in range(2):
        labels = -1 * np.ones((h, w), np.int)   # pixel labels
        kf = np.zeros((numk, 5), np.double)     # sp features: l,a,b,i,j
        ks = np.zeros(numk, np.int)             # sp sizes
        heap = []
        for k in range(numk):
            heap.append((0, k, seeds[k, 0], seeds[k, 1]))
        heapq.heapify(heap)
        while len(heap) > 0:
            d, k, i, j = heapq.heappop(heap)
            if labels[i, j] < 0: # takes longer time if removed
                labels[i, j] = k
                l, a, b = img[i, j, :].tolist()
                kf[k, :] = (kf[k, :] * ks[k] + (l, a, b, i, j)) / (ks[k] + 1)
                ks[k] += 1
                for n in range(connectivity):
                    ii = i + di[n]
                    jj = j + dj[n]
                    if ii < 0 or jj < 0 or ii >= h or jj >= w:
                        continue
                    if labels[ii, jj] < 0:
                        pf = (*(img[ii, jj, :].tolist()), ii, jj) # pixel features
                        dist = kf[k, :] - pf
                        dist = [d * d for d in dist]
                        colordist = sum(dist[0:3])
                        spacedist = sum(dist[3:5])
                        if iter == 0:
                            slicdist = colordist / DCN / DCN + spacedist / area
                            if maxcdsq[k] < colordist:
                                maxcdsq[k] = colordist
                        else:
                            if maxcdsq[k] == 0:
                                slicdist = spacedist / area
                            else:
                                slicdist = colordist / maxcdsq[k] + spacedist / area
                        heapq.heappush(heap, (slicdist, k, ii, jj))
    return labels

def show_seeds(img, numk):
    h = img.shape[0]
    w = img.shape[1]
    numk_new, seeds = find_seeds(w, h, numk)
    for k in range(numk_new):
        i, j = seeds[k, :]
        cv2.circle(img, (j, i), 1, [255, 0, 0], -1)
    plt.imshow(img)
    plt.show()
    print(seeds)


#%% Testing
def export_labels_mat():
    img = cv2.imread('example.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    labels = snic(img, 200, 20)
    savemat('matlab/reimpl_labels.mat', {"reimpl_labels":labels})

def test(zero_param=False):
    img = cv2.imread('j20.jpg')
    #img = img[0:100,0:100]
    print('%dx%d' % (img.shape[0], img.shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t0 = time()
    if zero_param:
        labels = snico(img, 200)
    else:
        labels = snic(img, 200, 20)
    print('Time used: %.3fs' %(time() - t0))
    utils.show_bounaries(img, labels)

if __name__ =='__main__':
    test(True)
    # TODO: visualize algorithm process