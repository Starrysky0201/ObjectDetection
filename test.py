import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
import torchvision
import torchvision.transforms as transforms


label_name=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class MultiClassObjectDetectionMetrics:
    def __init__(self, iou_threshold: float = 0.5):
        """
        多类别目标检测评价指标计算器

        Args:
            iou_threshold: 用于匹配检测框的IoU阈值
        """
        self.iou_threshold = iou_threshold
        self.class_aps = {}  # 存储每个类别的平均精度

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框的交并比(IoU)

        Args:
            box1: 第一个边界框 [x_min, y_min, x_max, y_max]
            box2: 第二个边界框 [x_min, y_min, x_max, y_max]

        Returns:
            IoU值
        """
        # 计算交集坐标
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # 计算交集面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算并集面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def calculate_ap_for_class(
            self,
            ground_truth: List[List[float]],
            predictions: List[List[float]],
            class_id: int,
            confidence_threshold: float = 0.5
    ) -> Tuple[float, List[float], List[float]]:
        """
        计算单个类别的平均精度(Average Precision)

        Args:
            ground_truth: 真实目标框 [[x_min, y_min, x_max, y_max, class_id], ...]
            predictions: 预测目标框 [[x_min, y_min, x_max, y_max, class_id, confidence], ...]
            class_id: 当前计算的类别ID
            confidence_threshold: 置信度阈值

        Returns:
            平均精度、精确率列表、召回率列表
        """
        # 过滤特定类别的真实目标和预测结果
        gt_class = [gt for gt in ground_truth if gt[4] == class_id]
        pred_class = [pred for pred in predictions if pred[4] == class_id]

        # 按置信度降序排序预测结果
        pred_class = sorted(pred_class, key=lambda x: x[5], reverse=True)

        # 初始化精确率和召回率列表
        precisions = []
        recalls = []

        # 统计真实目标数量
        total_gt = len(gt_class)

        # 记录已匹配的真实目标
        matched_gt = set()

        # 存储正例和负例
        true_positives = 0
        false_positives = 0

        for pred in pred_class:
            if pred[5] < confidence_threshold:
                continue

            # 查找最佳匹配
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_class):
                if gt_idx in matched_gt:
                    continue

                iou = self.calculate_iou(pred[:4], gt[:4])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx != -1:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1

            # 计算精确率和召回率
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / total_gt if total_gt > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        # 计算AP（使用11点插值法）
        ap = self._interpolate_ap(precisions, recalls)

        return ap, precisions, recalls

    def _interpolate_ap(self, precisions: List[float], recalls: List[float]) -> float:
        """
        使用11点插值法计算AP

        Args:
            precisions: 精确率列表
            recalls: 召回率列表

        Returns:
            平均精度(AP)
        """
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            max_precision = max([p for p, r in zip(precisions, recalls) if r >= t], default=0)
            ap += max_precision
        return ap / 11

    def calculate_map(
            self,
            ground_truth: List[List[float]],
            predictions: List[List[float]],
            confidence_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        计算多类别平均精度(mAP)

        Args:
            ground_truth: 真实目标框 [[x_min, y_min, x_max, y_max, class_id], ...]
            predictions: 预测目标框 [[x_min, y_min, x_max, y_max, class_id, confidence], ...]
            confidence_threshold: 置信度阈值

        Returns:
            评价指标字典
        """
        # 获取所有唯一的类别ID
        class_ids = set(gt[4] for gt in ground_truth)

        # 计算每个类别的AP
        for class_id in class_ids:
            ap, precisions, recalls = self.calculate_ap_for_class(
                ground_truth,
                predictions,
                class_id,
                confidence_threshold
            )
            self.class_aps[class_id] = {
                'AP': ap,
                'precisions': precisions,
                'recalls': recalls
            }

        # 计算mAP
        mean_ap = np.mean(list(self.class_aps[cid]['AP'] for cid in class_ids))

        # 评价指标
        metrics = {
            'mAP': mean_ap,
            'class_aps': {label_name[int(cid)]: self.class_aps[cid]['AP'] for cid in class_ids}
        }

        return metrics

    def plot_pr_curves(self):
        """
        绘制每个类别的精确率-召回率曲线
        """
        plt.figure(figsize=(12, 8))

        for class_id, class_data in self.class_aps.items():
            plt.plot(
                class_data['recalls'],
                class_data['precisions'],
                label=f'{label_name[int(class_id)]}'
            )

        plt.title('Precision-Recall Curves')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def test_FPN(test_image):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 21  # 包含背景类
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)
    model.load_state_dict(torch.load("./faster_rcnn_voc2007.pth"), False)

    model.eval()
    tensor_image = transform(test_image)
    predictions = model(tensor_image.unsqueeze(0))

    gt_predictions =[]

    for pred in predictions:
        bboxs = pred['boxes']
        labels = pred['labels'].unsqueeze(1)
        scores = pred['scores'].unsqueeze(1)
        result = torch.cat((bboxs,labels,scores), dim=1)
        gt_predictions.append(result)
    return gt_predictions
# 示例使用
def main():
    # 模拟多类别目标检测数据
    ground_truth = [
        # [x_min, y_min, x_max, y_max, class_id]
        [10, 10, 50, 50, 0],  # 类别0的目标1
        [100, 100, 150, 150, 1],  # 类别1的目标1
        [200, 200, 250, 250, 2],  # 类别2的目标1
        [300, 300, 350, 350, 0]  # 类别0的目标2
    ]

    predictions = [
        # [x_min, y_min, x_max, y_max, class_id, confidence]
        [12, 12, 48, 48, 0, 0.9],  # 类别0的预测1
        [110, 110, 140, 140, 1, 0.7],  # 类别1的预测1
        [210, 210, 240, 240, 2, 0.8],  # 类别2的预测1
        [305, 305, 345, 345, 0, 0.6]  # 类别0的预测2
    ]
    test_image = Image.open('000005.jpg')
    test_FPN(test_image)

    # 创建评价指标实例
    metrics = MultiClassObjectDetectionMetrics(iou_threshold=0.5)

    # 计算多类别mAP
    map_results = metrics.calculate_map(ground_truth, predictions)

    # 打印结果
    print("Mean Average Precision (mAP):", map_results['mAP'])
    print("Class-wise Average Precision:", map_results['class_aps'])

    # 绘制PR曲线
    metrics.plot_pr_curves()



if __name__ == "__main__":
    # a=torch.tensor([1,2]).unsqueeze(1)
    # b= torch.tensor([[3,4],[5,6]])
    # c = torch.cat((a,b),dim=1)
    # print(c)
    main()
