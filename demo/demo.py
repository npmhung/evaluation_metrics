from time import time
from metrics import *
from glob import glob
import os
from utils.via_io import VIAReader
import cv2 as cv
from shutil import rmtree

DEMO_PATH = './demo'
DEBUG = False
set_debug(DEBUG)

print('Debug mode: ', DEBUG)


def test_iou(imgs, gts, preds, names):
    base = os.path.join(DEMO_PATH, 'output')
    if not os.path.exists(base):
        os.makedirs(base)
    for im, gt, pred, name in zip(imgs, gts, preds, names):
        print('Image:', name)
        # break
        try:
            # create gt & prediction map to use iou
            _gt = draw_box(np.zeros_like(im), gt, 1)
            _pred = draw_box(np.zeros_like(im), pred, 1)
            if DEBUG:
                tmp = np.stack([_pred, np.zeros_like(_pred), _gt], 2)*255
                cv.imwrite(os.path.join(base, name+'.png'), tmp)
            # print('dafadf',pred)
            # print(_gt.dtype, np.unique(_gt, return_counts=True))
            print(mean_iou(_gt, _pred, 2, [1,1]))
        except Exception as e:
            print(e)
        #     pass
        # break


def test_err(imgs, gts, preds, names, err_type):
    base = os.path.join(DEMO_PATH, 'output')
    if not os.path.exists(base):
        os.makedirs(base)
    for name, gt, pred, img in zip(names, gts, preds, imgs):
        try:
            print('Image:', name)

            err = err_type()
            # print(out)
            start_time = time()
            # print(gt)
            err, db_imgs = err(gt, pred, img)

            print('Time: %f'%(time()-start_time))
            print(err)
            out_folder = os.path.join(base, name)
            if os.path.exists(out_folder):
                rmtree(out_folder)
            os.makedirs(out_folder)

            if DEBUG:
                for idx, im in enumerate(db_imgs):
                    # print(os.path.join(out_folder, '%d.png'%idx))
                    cv.imwrite(os.path.join(out_folder, '%d.png'%idx), im)
        except Exception as e:
            print(e)
        # return



def load_data(path):
    im_path = os.path.join(path, 'imgs')
    gt_path = os.path.join(path, 'gts')
    pr_path = os.path.join(path, 'preds')
    ext = 'png'
    ll = sorted([os.path.splitext(os.path.basename(im))[0] for im in glob(os.path.join(im_path, '*.%s'%ext))])
    # print(ll, os.path.join(im_path, '*.%s'%ext))
    imgs = []
    gts = []
    preds = []
    names = []
    for iname in ll:
        gtp = os.path.join(gt_path, iname + '.csv')
        prp = os.path.join(pr_path, iname + '.csv')
        if os.path.exists(gtp) and os.path.exists(prp):
            gt = VIAReader(gtp).getBoxes()
            pr = VIAReader(prp).getBoxes()
            im = cv.imread(os.path.join(im_path, iname + '.%s' % ext))
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            # print(im.shape)
            imgs.append(im)
            # print(gt)
            gts.append(np.array(gt))
            preds.append(np.array(pr))
            names.append(iname)

    return imgs, gts, preds, names


def main():
    imgs, gts, preds, names = load_data(DEMO_PATH)
    # test_err(imgs, gts, preds, names, merge_error) #test merge
    # test_err(imgs, gts, preds, names, split_error)  # test split
    # test_iou(imgs, gts, preds, names)

if __name__ == '__main__':
    main()

