import os
import xml.etree.ElementTree as ET

from numpy.compat import os_PathLike
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from sklearn.model_selection import train_test_split

# VOC数据集路径配置
VOC_PATH = 'datasets/VOCdevkit/VOC2007'
IMAGES_DIR = os.path.join(VOC_PATH, 'JPEGImages')
ANNOTATIONS_DIR = os.path.join(VOC_PATH, 'Annotations')

# VOC 2007类别定义
CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def parse_voc_annotation(annotation_path):
    """
    解析VOC格式的XML标注文件
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    image_name = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    boxes = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in CLASSES:
            continue

        class_id = CLASSES.index(class_name)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # 转换为YOLOv8归一化格式 [class_id, x_center, y_center, width, height]
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        boxes.append([class_id, x_center, y_center, bbox_width, bbox_height])

    return image_name, boxes


def create_yolo_labels(output_dir):
    """
    将VOC标注转换为YOLO格式
    """
    os.makedirs(output_dir, exist_ok=True)

    for annotation_file in os.listdir(ANNOTATIONS_DIR):
        annotation_path = os.path.join(ANNOTATIONS_DIR, annotation_file)
        image_name, boxes = parse_voc_annotation(annotation_path)

        # 创建对应的txt标注文件
        label_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.txt')
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(' '.join(map(str, box)) + '\n')


def prepare_dataset():
    """
    准备数据集并分割训练/验证集
    """
    images = os.listdir(IMAGES_DIR)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    # 创建数据配置文件
    with open('datasets/voc2007.yaml', 'w') as f:
        f.write(f"""
            path: {VOC_PATH}  # 数据集根目录
            train: images/train  # 训练图像目录
            val: images/val    # 验证图像目录
            
            nc: {len(CLASSES)}  # 类别数量
            names: {CLASSES}   # 类别名称
            """)

    # 创建训练和验证目录
    os.makedirs(os.path.join(VOC_PATH, 'images\\train'), exist_ok=True)
    os.makedirs(os.path.join(VOC_PATH, 'images\\val'), exist_ok=True)
    os.makedirs(os.path.join(VOC_PATH, 'labels\\train'), exist_ok=True)
    os.makedirs(os.path.join(VOC_PATH, 'labels\\val'), exist_ok=True)

    # 创建标签并复制图像到对应目录
    create_yolo_labels(os.path.join(VOC_PATH, 'labels'))

    # 加载训练，测试集的图像和标签
    for img in train_images:
        os.symlink(
            os.path.join('..\\..\\JPEGImages', img),
            os.path.join(VOC_PATH, 'images\\train',img)
        )
        label_path = os.path.splitext(img)[0]+'.txt'
        os.symlink(
            os.path.join('..\\', label_path),
            os.path.join(VOC_PATH, 'labels\\train',label_path)
        )
    for img in val_images:
        os.symlink(
            os.path.join('..\\..\\JPEGImages', img),
            os.path.join(VOC_PATH, 'images\\val',img)
        )
        label_path = os.path.splitext(img)[0]+'.txt'
        os.symlink(
            os.path.join('..\\', label_path),
            os.path.join(VOC_PATH, 'labels\\val',label_path)
        )


def train_yolov8():
    """
    使用YOLOv8训练目标检测模型
    """
    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    # 训练
    results = model.train(
        data='F:\\AI-learn\\obiect\datasets\\voc2007.yaml',
        epochs=10,
        batch=16,
        imgsz=640,
        device='0',  # GPU设备
        plots=True  # 生成训练过程的性能图
    )
    model.save('yolov8_voc2007.pt')


def main():
    prepare_dataset()
    train_yolov8()


if __name__ == '__main__':
    main()