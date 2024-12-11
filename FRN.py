import gc

import numpy as np
from PIL import Image
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import classes

from torch.utils.data import DataLoader, Dataset
import cv2
import torchvision
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import matplotlib.pyplot as plt

from test import MultiClassObjectDetectionMetrics

name_to_idx={
    'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4, 'bus':5, 'car':6,
    'cat':7, 'chair':8, 'cow':9, 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
    'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19
}
label_name=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


# 定义VOC2007数据导入类
class VOCDataset(Dataset):
    def __init__(self,root, year, image_set, download=False, transforms = None):
        self.voc_dataset = VOCDetection(root, year = year, image_set=image_set, download=download)
        self.transforms = transforms
    def __getitem__(self, item):
        img, target = self.voc_dataset[item]
        # 提取边界框和标签
        boxes = []
        labels = []
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])

            label = name_to_idx[obj['name']]
            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.tensor(labels, dtype = torch.long)
        target = {
            'boxes': boxes,
            'labels':labels
        }
        # 图像预处理
        if self.transforms is not None:
            img= self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.voc_dataset)

# 可视化损失
def draw_loss(epochs, train_loss):
    img = plt.figure()
    plt.plot(range(1, epochs+1),train_loss, label = 'Train loss',color = 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # plt.savefig('loss.jpg',img)

def get_transform(train):
    transforms = []
    transforms.append(transforms.ToTensor())
    transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.3))
    return torchvision.transforms.Compose(transforms)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model(epochs = 10):
    # 导入训练数据
    data_path = './datasets/VOCdevkit'
    train_dataset = VOCDataset(data_path, year='2007', image_set='trainval', transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=1,shuffle=True,
                              collate_fn = lambda x:tuple(zip(*x)))
    # 导入模型
    model = fasterrcnn_resnet50_fpn(pretrained = True)
    num_classes = 21 # 包含背景类
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)
    # 将模型加载到GPU上
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # 定义模型损失
    train_loss = []
    # 定义SGD优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # 训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # 计算损失
            loss_dict = model(images, targets)
            losses = loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] + loss_dict['loss_objectness'] + \
                     loss_dict['loss_rpn_box_reg']
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            losses.backward()
            # 梯度清零
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.step()
            total_loss += losses.item()
        train_loss.append(total_loss/len(train_loader))
        print('Epoch {} : Loss {:.5f}'.format(epoch, total_loss/len(train_loader)))

    # 保存模型
    torch.save(model.state_dict(), 'faster_rcnn_voc2007.pth')

    draw_loss(epochs, train_loss)

    return model

def draw_result():
    test_image = Image.open('test.jpg')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 21  # 包含背景类
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)
    model.load_state_dict(torch.load("./faster_rcnn_voc2007.pth"), True)

    model.eval()
    tensor_image = transform(test_image)
    predictions = model(tensor_image.unsqueeze(0))

    for pred in predictions:
        scores = pred["scores"]
        mask = scores > 0.7

        boxes = pred['boxes'][mask].int().detach().numpy()
        labels = pred['labels'][mask]
        scores = scores[mask]

        plt.imshow(test_image)
        for boxe, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = boxe
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
            plt.text(x1, y1, '{} {:.2f}'.format(label_name[label], score), color='white')
        plt.show()

def test_model():
    # 导入数据
    data_path = './datasets/VOCdevkit'
    test_dataset = VOCDataset(data_path, year='2007', image_set='test', transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)))
    # 导入模型
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 21  # 包含背景类
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)
    model.load_state_dict(torch.load("./faster_rcnn_voc2007.pth"), True)
    model.eval()
    model.to(device)
    # 预测
    ground_truth = []
    predictions = []
    with torch.no_grad():
        i = 0
        for imgs, targets in test_loader:
            boxes = targets[0]['boxes']
            labels = targets[0]['labels'].unsqueeze(1)
            result = torch.cat((boxes, labels), dim =1)

            ground_truth.append(result.tolist())

            imgs = list(image.to(device) for image in imgs)
            pred = model(imgs)[0]
            bboxs = pred['boxes']
            labels = pred['labels'].unsqueeze(1)
            scores = pred['scores'].unsqueeze(1)
            result = torch.cat((bboxs, labels, scores), dim=1)
            predictions.append(result.tolist())

            # i+=1
            # if i>99:
            #     break

    metrics = MultiClassObjectDetectionMetrics(iou_threshold=0.5)
    ground_truth = [ box for bbox in ground_truth for box in bbox ]
    predictions = [box for bbox in predictions for box in bbox]
    map_results = metrics.calculate_map(ground_truth, predictions)

    # 打印结果
    print("Mean Average Precision (mAP):", map_results['mAP'])
    print("class\t\t AP")
    # print(map_results['class_aps'])
    for key in map_results['class_aps']:
        print('{:10}\t {:.3f}'.format(key, map_results['class_aps'][key]))
    # for class_ap in map_results['class_aps']:
    #     ap_pre = map_results['class_aps'][class_ap]


    # 绘制PR曲线
    metrics.plot_pr_curves()



if __name__ == '__main__':
    train_model()
    test_model()
    draw_result()