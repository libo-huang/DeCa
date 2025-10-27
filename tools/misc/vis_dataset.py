import fiftyone as fo

# 1. 定义您的 KITTI 训练集数据和标签的 *确切* 路径
image_dir = "/data4/maihn/Project/Data/ICDA/KITTI/clear/training/image_2"
label_dir = "/data4/maihn/Project/Data/ICDA/KITTI/clear/kitti_val_split.json"

# 2. 加载数据集
#    我们使用 data_path 和 labels_path 显式指定路径
#    而不是使用 dataset_dir 和 split

dataset = fo.Dataset.from_dir(
    data_path=image_dir,        # <--- 显式指向 image_2 文件夹
    labels_path=label_dir,      # <--- 显式指向 label_2 文件夹
    dataset_type=fo.types.COCODetectionDataset,
    name="kitti-coco-dataset2",
)

# (可选) 如果您想加载 'testing' 部分（只有图像，没有标签）
# test_dataset = fo.Dataset.from_dir(
#     dataset_dir=kitti_data_dir,
#     dataset_type=fo.types.KITTIDetectionDataset,
#     split="testing",
#     name="kitti-testing-dataset",
# )

# (可选) 计算图像的元数据（宽度、高度等），这有助于在 App 中筛选
dataset.compute_metadata()


fo.config.server_address = "0.0.0.0"
# 3. 启动 FiftyOne App 进行可视化
session = fo.launch_app(dataset)

# （可选）在脚本中保持 App 运行，直到您手动关闭它
# 如果您在 Jupyter Notebook 中运行，则不需要这行
session.wait()

session = fo.launch_app(dataset)

session.wait()