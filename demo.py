import numpy as np
from PIL import Image

# 这里假设你已有 generate_arc_mask
# 如果已在其它文件中定义，记得自行导入
def generate_arc_mask(direction_angle=0, size=400, move=180, radius=340):
    """
    生成一个 size×size 的黑白掩膜，圆外白(255)，圆内黑(0)
    """
    # img = np.zeros((size, size), dtype=np.uint8)
    cx1 = size // 2 + int(move * np.cos(np.radians(direction_angle)))
    cy1 = size // 2 + int(move * np.sin(np.radians(direction_angle)))

    y, x = np.ogrid[:size, :size]
    dist = (x - cx1) ** 2 + (y - cy1) ** 2
    mask = np.ones((size, size), dtype=np.uint8) * 255
    mask[dist <= radius ** 2] = 0
    mask[dist <= (radius - 100) ** 2] = 255

    return mask


# ---------------------------
# 1. 生成 400×400 全黑图像
# ---------------------------
img = np.zeros((400, 400), dtype=np.uint8)

# ---------------------------
# 2. 生成掩膜
# ---------------------------
mask = generate_arc_mask()

# ---------------------------
# 3. 掩膜区域涂白
#    掩膜值为 255 的地方变成白色
# ---------------------------
result = img.copy()
result[mask == 255] = 255

# ---------------------------
# 4. 保存结果图像
# ---------------------------
Image.fromarray(result).save("result.png")

print("图像已保存为 result.png")
