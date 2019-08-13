import python_y4m.y4m as y4m
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, wait
import glob
import zipfile
# sys.path.append('..')


def process_frame(frame, save_prefix):
    # 针对图片做一些处理
    # print(yuv)
    yuv = frame[0]  # 原始数据
    headers = frame[1]  # 头部字典
    count = frame[2]  # 第几帧
    print(count)

    y = yuv[0:len(yuv) // 3 * 2]
    u = yuv[len(yuv) // 3 * 2:len(yuv) // 6 * 5]
    v = yuv[len(yuv) // 6 * 5:]
    y = [int(e) for e in y]
    u = [int(e) for e in u]
    v = [int(e) for e in v]
    u = np.array(u, dtype=np.uint8)
    y = np.array(y, dtype=np.uint8)
    v = np.array(v, dtype=np.uint8)

    y = y.reshape(headers['H'], headers['W'])
    u = u.reshape(headers['H'] // 2, headers['W'] // 2)
    v = v.reshape(headers['H'] // 2, headers['W'] // 2)
    cb = np.zeros((headers['H'], headers['W']), dtype=np.uint8)
    cr = np.zeros((headers['H'], headers['W']), dtype=np.uint8)
    cb[0::2, 0::2] = u
    cb[1::2, 0::2] = u
    cb[0::2, 1::2] = u
    cb[1::2, 1::2] = u

    cr[0::2, 0::2] = v
    cr[1::2, 0::2] = v
    cr[0::2, 1::2] = v
    cr[1::2, 1::2] = v

    ycbcr = np.stack((y, cb, cr), -1)
    np.save(save_prefix + '_{:03d}.npy'.format(count), ycbcr)

    # ycrcb = np.stack((y, cr, cb), -1)
    # print(np.mean(cb))
    # print(np.mean(cr))
    # print(np.min(cb))
    # print(np.min(cr))
    # print(np.min(y))

    # print(type(ycbcr))
    # bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    # bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    #
    # print(ycrcb.shape)
    # cv2.imshow('y', bgr)
    # cv2.waitKey()
    #
    # cv2.imshow('a', cb)
    # cv2.waitKey()
    # #
    # cv2.imshow('b', cr)
    # cv2.waitKey()
    #


if __name__ == '__main__':

    zip_dir = '../data/round2_train_label'  # youku_00250_00299_h_GT.zip
    out_base_dir = './data/round2_label'
    # out_base_dir = '../data/round2/'

    num_workers = 8
    thread_pool = ThreadPoolExecutor(max_workers=num_workers)
    futures = []

    train_start1 = 0
    train_end1 = 199
    train_start2 = 250
    train_end2 = 849

    test_start1 = 200
    test_end1 = 249
    test_start2 = 850
    test_end2 = 999

    train_set = [i for i in range(train_start1, train_end1+1)]
    train_set.extend([i for i in range(train_start2, train_end2+1)])
    test_set = [i for i in range(test_start1, test_end1+1)]
    test_set.extend([i for i in range(test_start2, test_end2+1)])

    for zip_fn in sorted(glob.glob(os.path.join(zip_dir, '*.zip'))):
        if (not '_l' in zip_fn) and (not '_h_GT' in zip_fn): continue

        zip_start_vid = int(os.path.basename(zip_fn).split('_')[1])
        if zip_start_vid in train_set:
            save_dir = os.path.join(out_base_dir, 'train_np')
        else:
            save_dir = os.path.join(out_base_dir, 'test_np')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        zip_fr = zipfile.ZipFile(zip_fn)

        for y4m_fn in zip_fr.namelist():
            f = zip_fr.open(y4m_fn)

            parser = y4m.Reader(process_frame, verbose=False)  # 如果用一个对象一直运作会出错
            save_prefix = os.path.splitext(y4m_fn)[0]
            save_prefix = os.path.join(save_dir, save_prefix)
            #
            futures.append(thread_pool.submit(parser.decode, f.read(), save_prefix))

            if len(futures) % num_workers == 0:
                done, pending = wait(futures, timeout=None, return_when='FIRST_COMPLETED')
                for future in done:
                    futures.remove(future)

            # parser.decode(f.read(), save_prefix)  # single processing
        wait(futures)
        futures.clear()

