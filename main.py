import cv2
import time

import snic
import merge
import utils


def process_image(img_path, numk=200, compactness=20, percentage=0.95):
    img = cv2.imread(img_path)
    print('Processing %s (%dx%d).' % (img_path, img.shape[0], img.shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t = []
    t.append(time.time())
    labels = snic.snic(img, numk, compactness)
    utils.save_boundaries(img, labels, img_path[:img_path.rfind('.')] + '_snic.png')
    t.append(time.time())
    labels = snic.snico(img, numk)
    utils.save_boundaries(img, labels, img_path[:img_path.rfind('.')] + '_snico.png')
    t.append(time.time())
    labels = merge.merge(img, labels, percentage)
    utils.save_boundaries(img, labels, img_path[:img_path.rfind('.')] + '_snico_merge.png')
    t.append(time.time())

    print('Time: ', end='')
    for i in range(1, len(t)):
        print('%.2f' % (t[i] - t[i - 1]), end=' ')
    print()

# data_folder = 'images'
# for fn in os.listdir(data_folder):
#     if fn.find('_') != -1:
#         continue
#     process_image(data_folder + '/' + fn)

#process_image('images/118035.jpg', percentage=0.95)
#process_image('images/124084.jpg', percentage=0.95)
#process_image('images/135069.jpg', percentage=0.98)