import argparse
import math
import os
import random
import time

import torch
from torch import optim
from torchsummary import summary
from tqdm import tqdm

from module.detector import Detector
from module.loss import DetectorLoss
from utils.datasets import *
from utils.evaluation import CocoDetectionEvaluator
from utils.tool import *
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#import pdb;pdb.set_trace()
device = torch.device("cpu")
class FastestDet:
    def __init__(self):
        # 指定训练配置文件
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--yaml", type=str, default="./configs/coco.yaml", help=".yaml config"
        )
        parser.add_argument("--weight", type=str, default=None, help=".weight config")

        opt = parser.parse_args()
        assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"

        # 解析yaml配置文件
        self.cfg = LoadYaml(opt.yaml)
        print(self.cfg)

        # 初始化模型结构
        if opt.weight is not None:
            print("load weight from:%s" % opt.weight)
            self.model = Detector(self.cfg.category_num, True).to(device)
            self.model.load_state_dict(torch.load(opt.weight))
        else:
            self.model = Detector(self.cfg.category_num, False).to(device)

        # # 打印网络各层的张量维度
        summary(self.model, input_size=(3, self.cfg.input_height, self.cfg.input_width))

        # 构建优化器
        print("use SGD optimizer")
        self.optimizer = optim.SGD(
            params=self.model.parameters(),
            lr=self.cfg.learn_rate,
            momentum=0.949,
            weight_decay=0.0005,
        )
        # 学习率衰减策略
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.cfg.milestones, gamma=0.1
        )

        # 定义损失函数
        self.loss_function = DetectorLoss(device)

        # 定义验证函数
        self.evaluation = CocoDetectionEvaluator(self.cfg.names, device)

        # 数据集加载
        val_dataset = TensorDataset(
            self.cfg.val_txt, self.cfg.input_width, self.cfg.input_height, False
        )
        train_dataset = TensorDataset(
            self.cfg.train_txt, self.cfg.input_width, self.cfg.input_height, True
        )

        # 验证集
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=False,
            persistent_workers=True,
        )
        # 训练集
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=True,
            persistent_workers=True,
        )
        random.seed(time.time())
        self.train_id = random.randint(100000, 999999)
    def train(self):
        # 迭代训练
        batch_num = 0
        print("Starting training for %g epochs..." % self.cfg.end_epoch)
        #import pdb;pdb.set_trace()
        for epoch in range(self.cfg.end_epoch + 1):
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            #import pdb;pdb.set_trace()
            for imgs, targets in pbar:
                # 数据预处理
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)
                # 模型推理
                preds = self.model(imgs)

                # loss计算
                iou, obj, cls, total = self.loss_function(preds, targets)
                # 反向传播求解梯度
                total.backward()
                # 更新模型参数
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 学习率预热
                for g in self.optimizer.param_groups:
                    warmup_num = 5 * len(self.train_dataloader)
                    if batch_num <= warmup_num:
                        scale = math.pow(batch_num / warmup_num, 4)
                        g["lr"] = self.cfg.learn_rate * scale
                    lr = g["lr"]

                # 打印相关训练信息
                info = "Epoch:%d LR:%f IOU:%f Obj:%f Cls:%f Total:%f" % (
                    epoch,
                    lr,
                    iou,
                    obj,
                    cls,
                    total,
                )
                pbar.set_description(info)
                batch_num += 1

            # 模型验证及保存
            if epoch % 10 == 0 and epoch > 0:
                # 模型评估
                self.model.eval()
                print("computer mAP...")
                mAP05 = self.evaluation.compute_map(self.val_dataloader, self.model)
                torch.save(
                    self.model.state_dict(),
                    "checkpoint/T%d_weight_%f_%d-epoch.pth"
                    % (self.train_id, mAP05, epoch),
                )

            # 学习率调整
            self.scheduler.step()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--yaml", type=str, default="./configs/coco.yaml", help=".yaml config"
)
parser.add_argument("--weight", type=str, default=None, help=".weight config")
parser.add_argument("--img", type=str, default="./test.jpg", help="The path of test image")
parser.add_argument(
    "--thresh", type=float, default=0.65, help="The threshold of detection"
)
parser.add_argument(
    "--onnx", action="store_true", default=False, help="Export onnx file"
)
parser.add_argument(
    "--torchscript",
    action="store_true",
    default=False,
    help="Export torchscript file",
)
parser.add_argument("--cpu", action="store_true", default=False, help="Run on cpu")

opt = parser.parse_args()
#assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
#assert os.path.exists(opt.weight), "请指定正确的模型路径"
#assert os.path.exists(opt.img), "请指定正确的测试图像路径"

# 选择推理后端
'''if opt.cpu:
    print("run on cpu...")
    device = torch.device("cpu")
else:
    if torch.cuda.is_available():
        print("run on gpu...")
        device = torch.device("cuda")
    else:
        print("run on cpu...")
        device = torch.device("cpu")'''



# 解析yaml配置文件
cfg = LoadYaml(opt.yaml)

def gen_model():
    model = Detector(cfg.category_num, True)

    model.load_state_dict(torch.load("T669259_weight_0.915248_300-epoch.pth"),False)
    #ori_img = cv2.imread("test.jpg")
    #res_img = cv2.resize(
    #    ori_img, (cfg.input_width, cfg.input_height), interpolation=cv2.INTER_LINEAR
    #)
    #img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
    #img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    #img = img.to(device).float() / 255.0
    #start = time.perf_counter()
    #import pdb;pdb.set_trace()
    #model.cuda()
   # preds = model(img)
    #end = time.perf_counter()
    #time = (end - start) * 1000.0
    #print("forward time:%fms" % time)

    # 特征图后处理
    #output = handle_preds(preds, device, opt.thresh)
    
    # 加载label names
    LABEL_NAMES = []
    with open(cfg.names, "r") as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
    '''
    H, W, _ = ori_img.shape
    scale_h, scale_w = H / cfg.input_height, W / cfg.input_width

    # 绘制预测框
    for box in output[0]:
        print(box)
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * W), int(box[1] * H)
        x2, y2 = int(box[2] * W), int(box[3] * H)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, "%.2f" % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    cv2.imwrite("result.png", ori_img)
    '''
    return model,LABEL_NAMES

def img_tran(img):
    ori_img = img
    res_img = cv2.resize(
        ori_img, (cfg.input_width, cfg.input_height), interpolation=cv2.INTER_LINEAR
    )
    img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0
    start = time.perf_counter()
    return img

def outputhandling(preds,ori_img,draw=False):
    output = handle_preds(preds, device, opt.thresh)
    

    # 加载label names
    LABEL_NAMES = []
    with open(cfg.names, "r") as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    H, W, _ = ori_img.shape
    scale_h, scale_w = H / cfg.input_height, W / cfg.input_width
    # 绘制预测框
    for box in output[0]:
        print(box)
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * W), int(box[1] * H)
        x2, y2 = int(box[2] * W), int(box[3] * H)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, "%.2f" % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
    return output,ori_img