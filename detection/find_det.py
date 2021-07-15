import os
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
from . import unet
from . import segm_proc
import time
import numpy as np
import math
import shutil
import itertools
import multiprocessing
from utils.paths import DATA_DIR, IMG_DIR, POS_DIR, TMP_DIR, CHECKPOINT_DIR
from utils.func import DS, GPU_NAME, NUM_LAYERS, NUM_FILTERS, CLASSES
from utils import func

N_PROC = 3
BATCH_SIZE = 4

# 控制日志输出级别，info、warning和error都不输出
tf.logging.set_verbosity(tf.logging.ERROR)

# 用于多进程间通信，put：放数据，get：取数据
# get从队列里面取值并且把队列面的取出来的值删掉，没有参数的情况下就是是默认一直等着取值
to_save = multiprocessing.Queue()

def read_all_files():
    drs = [""]
    fls = []
    for dr in drs:
        dr_fls = os.listdir(IMG_DIR)
        dr_fls.sort()
        # 按时间排序后，文件地址放入fls中
        fls.extend(map(lambda fl: os.path.join(dr, fl), dr_fls)) # 将列表合并到列表末尾
    print("%i files" % len(fls), flush=True)
    return fls

#生成窗口网格坐标
def generate_offsets_for_frame():
    xs = range(0, func.FR_D, DS)  # FR_D=512图像大小, DS=256（像素窗口大小）
    ys = range(0, func.FR_D, DS)
    #product((1,2),('a','b'))=((1,'a'),(1,'b'),(2,'a'),(2,'b'))
    return list(itertools.product(xs, ys))

######## POSTPROCESSING AND SAVING SEGMENTATION RESULTS ############

#保存res的txt
def save_output_worker():
    output = np.zeros((BATCH_SIZE, 2, DS, DS)) # (shape:(b, 2, 256, 256))
    while True:
        output_i, offs, cur_fr = to_save.get()  # 输出的标号、偏差、正在处理的帧数
        if output_i < 0:
            break  # 用于停止进程
        fl = os.path.join(TMP_DIR, "segm_outputs_%i.npy" % output_i)  # 第i张图的分割结果（暂存）
        output[:,:,:,:] = np.load(fl)  #（batch，(0:class,1:angel),x,y)
        os.remove(fl)

        res = np.zeros((0, 4))
        for batch_i in range(BATCH_SIZE):
            (off_x, off_y) = offs[batch_i]
            if (off_x >= 0) and (off_y >= 0):  #截取窗口的起点在图片内
                # 中心位置、预测类、预测角、主轴
                prs = segm_proc.extract_positions(output[batch_i, 0, :, :], output[batch_i, 1, :, :]) 
                res_batch = np.zeros((len(prs), 4))
                for i in range(len(prs)):
                    (x, y, cl, a, ax) = prs[i]
                    ax_d = math.degrees(ax)   # 数值转角度
                    a_d = math.degrees(a)
                    ax_d = ax_d + 180 if (segm_proc.angle_diff(a_d, ax_d) > 90) else ax_d
                    res_batch[i, :] = [x, y, cl, ax_d]   
                res_batch[:, 0] += off_x      #x + off_x
                res_batch[:, 1] += off_y      #y + off_y
                res = np.append(res, res_batch, axis=0) #像数组一样相加
        print("processed frame %i, %i bees" % (cur_fr, res.shape[0]), flush=True)
        with open(os.path.join(POS_DIR, "%06d.txt" % cur_fr), 'a') as f:
            np.savetxt(f, res, fmt='%i', delimiter=',', newline='\n')  #保存POD_DIR 文件名是当前帧数.内容：res（x,y,cl,ax_d）

############# INFERENCE MODEL #####################

class DetectionInference:   #推断

    def __init__(self):
        self.batch_data = np.zeros((BATCH_SIZE, DS, DS, 1), dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        None
    
    #默认使用GPU
    def build_model(self, checkpoint_dir):
        cpu, gpu = func.find_devices()
        tf_dev = gpu if gpu != "" else cpu
        with tf.Graph().as_default(), tf.device(cpu):

            # batch size = 进程数*gpu数
            # update_ops = []
            self.is_train = False

            with tf.device(tf_dev), tf.name_scope('%s_%d' % (GPU_NAME, 0)) as scope:
                self.placeholder_img = tf.placeholder(tf.float32, shape=(BATCH_SIZE, DS, DS, 1), name="images")   # 灰度图
                self.placeholder_prior = tf.placeholder(tf.float32, shape=(BATCH_SIZE, DS, DS, NUM_FILTERS), name="prior") 
                # 进行预测（得到class、UNet最后一层、角度）
                logits, last_relu, angle_pred = unet.create_unet2(NUM_LAYERS, NUM_FILTERS, self.placeholder_img, self.is_train, prev=self.placeholder_prior, classes=CLASSES)
                self.outputs= (logits, angle_pred)
                self.priors = last_relu  # 将倒数第二层（UNet最后一层）保留
                tf.get_variable_scope().reuse_variables()  #获取计算图内部的参数
                #update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            
            # 保存模型 再调用模型权重
            #self.batchnorm_updates_op = tf.group(*update_ops)
            self.saver = tf.train.Saver(tf.global_variables())
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            checkpoint_nb = func.find_last_checkpoint(checkpoint_dir)
            checkpoint_file = os.path.join(checkpoint_dir, "model_%06d.ckpt" % checkpoint_nb)
            print("Restoring checkpoint %i.." % checkpoint_nb, flush=True)
            self.saver.restore(self.sess, checkpoint_file)
            init = tf.local_variables_initializer()
            self.sess.run(init)

    # 导入下一张图（再看看）
    def _feed_dict(self, offs, cur_fr, priors):
        img = func.read_img(cur_fr, IMG_DIR)
        for batch_i in range(BATCH_SIZE):
            (off_x, off_y) = offs[batch_i]
            if (off_x >= 0) and (off_y >= 0):
                self.batch_data[batch_i,:,:,0] = img[off_y:(off_y + DS), off_x:(off_x + DS)]  #添加这个窗口的图
            else:
                self.batch_data[batch_i, :, :, :] = 0
        res = [(self.placeholder_prior, priors), (self.placeholder_img, self.batch_data)]   #上一张图的权重和本次要预测的图打包
        return dict(res)

    
    # 加载窗口网格
    def _load_offs_for_run(self, offsets, start_i):
        res = []
        for batch_i in range(BATCH_SIZE):
            (off_x, off_y) = (-1, -1) if start_i >= len(offsets) else offsets[start_i]  # 把网格过一遍，超出长度赋（-1，-1）
            res.append((off_x, off_y))
            start_i = start_i + 1   # 开始的序号（把每个batch统合）
        return res, start_i   # 输出[(off_x,off_y),...]


    def start_workers(self):
        # 创建N_PROC个进程
        self.workers = [multiprocessing.Process(target=save_output_worker) for _ in range(N_PROC)]
        for p in self.workers:
            p.start()

    def stop_workers(self):
        for i in range(N_PROC):
            to_save.put((-1, [], -1))
        for p in self.workers:
            p.join()  # 主进程要等该子进程执行完后才能继续向下执行
    
    # outs：(logits, angle_pred)  output_i:第i轮训练
    def _save_output(self, outs, output_i):
        log_res = np.argmax(outs[0], axis=3)  # logit网格：取logit_pre通道数=3的（不明所以）
        angle_res = outs[1][:, :, :, 0]       # 角网格：直接就是angle_pre？
        res = np.append(np.expand_dims(log_res, axis=1), np.expand_dims(angle_res, axis=1), axis=1) # 加了一维，就可以区分logit（0）和angel（1）了
        np.save(os.path.join(TMP_DIR, "segm_outputs_%i.npy" % output_i), res)

    def run_inference(self, fls, offsets, start_off_i=0):  
        global to_save
        t1 = time.time()
        output_i = 0
        n_runs = math.ceil(len(offsets) / BATCH_SIZE)  # 运行轮数（跑完整张图）
        print("STARTING INFERENCE")
        for i in range(n_runs):
            run_offs, start_off_i = self._load_offs_for_run(offsets, start_off_i) # 得到一批次（off_x, off_y）的列表
            last_priors = np.zeros((BATCH_SIZE, DS, DS, NUM_FILTERS), dtype=np.float32) # 初始化‘UNet最后一层’
            for cur_fr in range(len(fls)):
                feed_dict = self._feed_dict(run_offs, cur_fr, last_priors)  # 将图分割为窗口
                # 运行model
                outs, last_priors = self.sess.run([self.outputs, self.priors], feed_dict=feed_dict) # feed_dict替换place_holder，得到[self.outputs, self.priors]
                self._save_output(outs, output_i) 
                to_save.put((output_i, run_offs, cur_fr))  # 存入（off_x,off_y）
                output_i += 1

        print("ALL FINISHED - time: %.3f min" % ((time.time() - t1)/60))


######## MAIN FUNCTION ##############

def find_detections(checkpoint_dir=os.path.join(CHECKPOINT_DIR, "unet2")):
    print(DATA_DIR)  
    if os.path.exists(POS_DIR):
        shutil.rmtree(POS_DIR) #递归删除文件夹下的所有子文件夹和子文件
    os.mkdir(POS_DIR)   #存放detection位置的path
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)

    fls = read_all_files()

    offsets = generate_offsets_for_frame()
    with DetectionInference() as model_obj:
        model_obj.build_model(checkpoint_dir)
        model_obj.start_workers()
        try:
            model_obj.run_inference(fls, offsets)
        finally:
            model_obj.stop_workers()
