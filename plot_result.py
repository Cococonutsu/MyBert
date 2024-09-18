import matplotlib.pyplot as plt

# 从文件中读取数据
def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [float(line.strip()) for line in data]

# 读取数据
cls_acc = read_file('./result/cls_acc.txt')
mask_acc = read_file('./result/mask_acc.txt')
train_loss = read_file('./result/train_loss.txt')

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制分类准确率图
plt.subplot(3, 1, 1)
plt.plot(cls_acc, label='Classification Accuracy')
plt.title('Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# 绘制掩码准确率图
plt.subplot(3, 1, 2)
plt.plot(mask_acc, label='Mask Accuracy', color='orange')
plt.title('Mask Accuracy')
plt.xlabel('')
plt.ylabel('Accuracy (%)')
plt.legend()

# 绘制训练损失图
plt.subplot(3, 1, 3)
plt.plot(train_loss, label='Training Loss', color='green')
plt.title('Training Loss')
plt.xlabel('')
plt.ylabel('Loss')
plt.legend()

# 调整布局并显示图形
plt.tight_layout()
plt.show()
