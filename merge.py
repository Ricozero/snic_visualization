import time
import cv2
import numpy as np

import utils
import snic


DMAX = np.inf

def get_gradmag(img_gray):
    gradx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=7)
    grady = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=7)
    gradmag = cv2.magnitude(gradx, grady) / 255
    return gradmag

def get_hsv_histogram(img_hsv, sp_labels):
    n_sp = sp_labels.max() + 1
    sp_pcs = np.zeros((n_sp, 1), dtype=np.uint16)
    sp_hsv_sums = np.zeros((n_sp, 8, 4, 4), dtype=np.uint16)
    for row in range(sp_labels.shape[0]):
        for col in range(sp_labels.shape[1]):
            sp_idx = sp_labels[row, col]
            sp_pcs[sp_idx] += 1
            x = img_hsv[row][col]
            sp_hsv_sums[sp_idx][int(x[0]/32)][int(x[1]/64)][int(x[2]/64)] += 1
    sp_hsv_sums = sp_hsv_sums.reshape(n_sp, 8 * 4 * 4)
    return sp_hsv_sums / sp_pcs

def get_rgb_histogram(img, sp_labels):
    n_sp = sp_labels.max() + 1
    sp_pcs = np.zeros((n_sp, 1), dtype=np.uint16)
    sp_rgb_sums = np.zeros((n_sp, 8, 8, 8), dtype=np.uint16)
    for row in range(sp_labels.shape[0]):
        for col in range(sp_labels.shape[1]):
            sp_idx = sp_labels[row, col]
            sp_pcs[sp_idx] += 1
            x = img[row][col]
            sp_rgb_sums[sp_idx][int(x[0]/32)][int(x[1]/32)][int(x[2]/32)] += 1
    sp_rgb_sums = sp_rgb_sums.reshape(n_sp, 8 * 8 * 8)
    return sp_rgb_sums / sp_pcs

def get_rgb_average(img, sp_labels):
    n_sp = sp_labels.max() + 1
    sp_pcs = np.zeros((n_sp, 1), dtype=np.uint16)
    sp_rgb_sums = np.zeros((n_sp, 3), dtype=np.uint16)
    for row in range(sp_labels.shape[0]):
        for col in range(sp_labels.shape[1]):
            sp_idx = sp_labels[row, col]
            sp_pcs[sp_idx] += 1
            sp_rgb_sums[sp_idx] += img[row][col]
    return sp_rgb_sums / sp_pcs

def euler_distance(avec, bvec):
    return np.sum(np.sqrt(np.abs(avec - bvec)))

def superpixel_adjacency_matrix(sp_labels, sp_feature, img_gradmag):
    n_sp = sp_labels.max() + 1
    # Stores dissimilarity
    Ma = DMAX * np.ones((n_sp, n_sp), np.float)
    # Stores border information
    Mb = np.zeros((n_sp, n_sp), np.float)
    # Stores border pixel count
    Mc = np.zeros((n_sp, n_sp), np.int)
    # Consider adjacency in four directions
    # Ignore bottom right pixel
    for i in range(sp_labels.shape[0] - 1):
        for j in range(sp_labels.shape[1] - 1):
            c = sp_labels[i, j] # Center
            r = sp_labels[i, j + 1] # Right
            d = sp_labels[i + 1, j] # Down
            # Horizontal
            if c != r:
                if Ma[c, r] == DMAX:
                    Ma[c, r] = Ma[r, c] = euler_distance(sp_feature[c, :], sp_feature[r, :])
                Mb[c, r] += img_gradmag[i, j]
                Mb[r, c] += img_gradmag[i, j]
                Mc[c, r] += 1
                Mc[r, c] += 1
            # Vertical
            if c != d:
                if Ma[c, d] == DMAX:
                    Ma[c, d] = Ma[d, c] = euler_distance(sp_feature[c, :], sp_feature[d, :])
                Mb[c, d] += img_gradmag[i, j]
                Mb[d, c] += img_gradmag[i, j]
                Mc[c, d] += 1
                Mc[d, c] += 1
    return Ma, Mb, Mc

def choose_neighbor(a, b, c):
    # Normalize border gradient
    with np.errstate(divide='ignore', invalid='ignore'):
        measure = b / c
    measure[np.isnan(measure)] = 0
    # Fuse border gradient and color distance
    measure += a
    return np.argmin(measure)

def merge_superpixel(sp_labels, sp_feature, img_gradmag, percentage, save_turns=False):
    Ma, Mb, Mc = superpixel_adjacency_matrix(sp_labels, sp_feature, img_gradmag)
    n_sp = Ma.shape[0]
    # Merge state, indicates a sp is merged with which sp or not
    # Make sure ms[i] <= i
    ms = np.array(range(n_sp))

    # Pixel counts
    sp_pcs = np.zeros(sp_labels.max() + 1, dtype=np.uint16)
    for i in range(sp_labels.shape[0] - 1):
        for j in range(sp_labels.shape[1] - 1):
            ind = sp_labels[i][j]
            sp_pcs[ind] += 1

    # Loop until no loop exists in NNG
    flag = True
    turn = 0
    count_sum = 0
    ms_turns = []
    while flag:
        flag = False
        turn += 1
        count = 0
        for i in range(n_sp):
            # Merged already, skip
            if ms[i] < i:
                continue
            # The nearest sp's index
            # Only choose from larger index to avoid repeating
            # Only possible to choose the first sp's index of a merged region,
            # because other merged sp's distances are marked DMAX
            ind = i + choose_neighbor(Ma[i, i:], Mb[i, i:], Mc[i, i:])
            # Two sps form a loop
            if Ma[i, ind] < DMAX and choose_neighbor(Ma[ind, :ind], Mb[ind, :ind], Mc[ind, :ind]) == i:
                # Latter sp merges with former sp
                ms[ind] = i
                # Use weighed average to update merged sp's feature
                sp_feature[i, :] = (sp_feature[i, :] * sp_pcs[i] + sp_feature[ind, :] * sp_pcs[ind]) / (sp_pcs[i] + sp_pcs[ind])
                # Update the ith sp's adjecncy
                for j in range(n_sp):
                    if (Ma[i, j] < DMAX or Ma[ind, j] < DMAX) and j != i and j != ind:
                        Ma[i, j] = Ma[j, i] = euler_distance(sp_feature[i, :], sp_feature[j, :])
                Mb[i, ind] = Mb[ind, i] = 0
                Mb[i, :] = Mb[i, :] + Mb[ind, :]
                Mb[:, i] = Mb[:, i] + Mb[:, ind]
                Mc[i, ind] = Mc[ind, i] = 0
                Mc[i, :] = Mc[i, :] + Mc[ind, :]
                Mc[:, i] = Mc[:, i] + Mc[:, ind]
                # Clear merged sp's neighbors
                Ma[:, ind] = Ma[ind, :] = DMAX
                Mb[:, ind] = Mb[ind, :] = 0
                Mc[:, ind] = Mc[ind, :] = 0
                # State update
                flag = True
                count += 1
        count_sum += count
        if save_turns:
            ms_turns.append(ms.copy())
        #print('Turn %d: %d' % (turn, count))
        if count_sum / n_sp > percentage:
            break

    # Intermediate results
    if save_turns:
        for m in ms_turns:
            for i in range(n_sp):
                k = i
                while m[k] < k:
                    k = m[k]
                m[i] = k

        msp_labels_list = []
        for m in ms_turns:
            msp_labels = np.zeros_like(sp_labels)
            for i in range(sp_labels.shape[0]):
                for j in range(sp_labels.shape[1]):
                    msp_labels[i, j] = m[sp_labels[i, j]]
            msp_labels_list.append(msp_labels.copy())
        return msp_labels_list

    # Merged sps link to the sp of smallest index
    for i in range(n_sp):
        k = i
        while ms[k] < k:
            k = ms[k]
        ms[i] = k

    # Rename sps to make the indices continuous
    cur = 0
    for i in range(n_sp):
        if ms[i] == i:
            ms[i] = cur
            cur += 1
        else:
            ms[i] = ms[ms[i]]

    # Create a merged superpixel label map
    msp_labels = np.zeros_like(sp_labels)
    for i in range(sp_labels.shape[0]):
        for j in range(sp_labels.shape[1]):
            msp_labels[i, j] = ms[sp_labels[i, j]]

    return msp_labels

def merge(img, sp_labels, percentage=0.8):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gradmag = get_gradmag(img_gray)
    sp_feature = get_rgb_histogram(img, sp_labels)
    msp_labels = merge_superpixel(sp_labels, sp_feature, img_gradmag, percentage)
    return msp_labels


if __name__ == "__main__":
    img_path = 'bee.jpg'
    img = cv2.imread(img_path)
    #img = img[:, 200:]
    print('%dx%d' % (img.shape[0], img.shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gradmag = get_gradmag(img_gray)

    t = []
    t.append(time.time())
    sp_labels = snic.snico(img, 200)
    t.append(time.time())
    #sp_feature = get_hsv_histogram(img_hsv, sp_labels)
    #sp_feature = get_rgb_average(img, sp_labels)
    sp_feature = get_rgb_histogram(img, sp_labels)
    t.append(time.time())
    print('Nsp: %d' % (sp_labels.max() + 1))

    no_turns = True
    if no_turns:
        msp_labels = merge_superpixel(sp_labels, sp_feature, img_gradmag, percentage=0.95)
        t.append(time.time())
        for i in range(1, len(t)):
            print('%.2f' % (t[i] - t[i - 1]), end=' ')
        print()
        utils.show_bounaries(img, sp_labels)
        utils.show_bounaries(img, msp_labels)
    else:
        msp_labels_list = merge_superpixel(sp_labels, sp_feature, img_gradmag, save_turns=True)
        t.append(time.time())
        for i in range(1, len(t)):
            print('%.2f' % (t[i] - t[i - 1]), end=' ')
        print()
        for i, msp_labels in enumerate(msp_labels_list):
            utils.show_bounaries(img, msp_labels)
