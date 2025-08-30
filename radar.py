import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Arial'

DATASET_LABEL_SIZE = 15      # 数据集（轴名称）字体
RADIAL_TICK_SIZE = 16        # 环形刻度与y轴刻度字体
LEGEND_FONT_SIZE = 15        # 图例字体

def normalize_data_for_radar(df_original):
    df_normalized = df_original.copy()
    
    for col in df_original.columns:
        if col == 'group':
            continue
    
        original_values = df_original[col].values
    
        min_val = np.min(original_values)
        max_val = np.max(original_values)
        
        if min_val < max_val * 0.1:
            axis_min = 0
        else:
            axis_min = max(0, min_val - (max_val - min_val) * 0.1)
        
        normalized_values = []
        for val in original_values:
            if max_val == min_val:
                normalized_val = 100 
            else:
                normalized_val = ((val - axis_min) / (max_val - axis_min)) * 100
            normalized_values.append(normalized_val)
        
        # 更新归一化后的数据
        df_normalized[col] = normalized_values
    
    return df_normalized


# 自定义数据 - Original performance values
# 医学
# df_mapping = pd.DataFrame({
#     'group': ['SICAP-MIL', 'SKINCANCER', 'LC-LUNG', 'NCT-CRC', 'WSSS4LUAD'],
#     'CLIP':	            [29.84, 3.63,  31.27, 24.43, 64.89], 
#     'CONCH':            [27.38, 53.73, 90.01, 61.31, 82.20], 
#     'PLIP':	            [47.30, 22.35, 84.98, 62.99, 73.08], 
#     'MUSK':	            [36.64, 59.58, 92.89, 66.91, 75.43], 
#     'TransCLIP_Max' :   [53.30, 64.84, 97.21, 90.56, 88.24],
#     'SOTA' :            [66.02, 78.45, 97.45, 87.66, 90.98]
    
# })
# groups_name = ['CLIP', 'CONCH', 'PLIP', 'MUSK', 'TransCLIP_Max', 'SOTA']
# colors = ["#B22222", "#9467BC", "#DA70D5", "#D1B48C", "#228B22", "#86CDEA"]
# title = 'medicine'

# df_mapping = pd.DataFrame({
#     'group': ['ImageNet', 'SUN397', 'Aircraft', 'Eurosat', 'Stanfordcars', 'Food101', 'Pets', 'Flowers102', 'Caltech101', 'DTD', 'UCF101'],
#     'CLIP' : [66.60 , 62.50 , 24.70 , 48.30 , 65.60 , 85.90 , 89.10 , 70.70 , 93.20 , 43.50 , 67.50],
#     'Zlap' : [68.8, 67.2, 26.9, 49.1, 67.1, 86.9, 87.8, 68.7, 90.8, 45.5, 71.4],
#     'TransCLIP' : [70.3, 68.9, 26.9, 65.1, 69.4, 87.1, 92.6, 76.7, 92.7, 49.5, 74.4],
#     'SOTA' : [76.67, 72.04, 31.56, 84.8, 72.4, 88.36, 94.96, 83.03, 96.15, 55.08, 79.01]
# })
# groups_name = ['CLIP', 'Zlap', 'TransCLIP',  'SOTA']
# colors = ["#B22222", "#9467BC", "#2CA02C", "#86CDEA"]
# title = 'nature'


df_mapping = pd.DataFrame({
    'group': ['AID',	'EuroSAT',	'MLRSNet',	'OPTIMAL31',	'PatternNet',	'RESISC45',	'RSC11',	'RSICB128',	'RSICB256',	'WHURS19'],
    'CLIP'	: [66.40,  	45.30, 	51.20, 	73.00, 	59.60, 	60.70, 	55.50, 	27.70, 40.30, 	81.10], 
'GeoRSCLIP'	: [70.30,  	53.40, 	65.00, 	79.60, 	75.80, 	68.80, 	68.30, 	29.00, 46.50, 	88.80], 
'RemoteCLIP': [91.70,  	35.50, 	56.30, 	77.60, 	55.90, 	68.10, 	61.80, 	26.00, 41.50, 	95.20], 
'SkyCLIP50'	: [70.30,  	52.60, 	63.20, 	79.50, 	73.80, 	66.70, 	61.20, 	39.00, 47.10, 	91.00], 
'TransCLIP_Max'	: [95.60,  	69.00, 	73.20, 	87.80, 	94.50, 	79.50, 	79.70, 	49.40, 61.80, 	98.70],
'SOTA'	    : [96.61,  	71.68, 	81.12, 	92.90, 	96.44, 	88.79, 	78.73, 	47.66, 61.09, 	99.30],
})
groups_name = ['CLIP', 'GeoRSCLIP', 'RemoteCLIP', 'SkyCLIP50', 'TransCLIP_Max', 'SOTA']
colors = ["#B22222", "#9467BC", "#DA70D5", "#D1B48C", "#228B22", "#86CDEA"]
title = 'sensing'



df_mapping = pd.DataFrame(columns=df_mapping['group'], data=df_mapping.values.T[1:])
df = normalize_data_for_radar(df_mapping)

# # 自然数据
# df_mapping = pd.DataFrame({
#     'group': ['sicap_mil', 'skincancer', 'lc_lung', 'nct', 'pannuke', 'WSSS4LUAD'],
#     'CLIP' : [66.60 , 62.50 , 24.70 , 48.30 , 65.60 , 85.90 , 89.10 , 70.70 , 93.20 , 43.50 , 67.50],
#     'TransCLIP' : [70.3, 68.9, 26.9, 65.1, 69.4, 87.1, 92.6, 76.7, 92.7, 49.5, 74.4],
#     'SOTA' : [76.67, 72.04, 31.56, 84.8, 72.4, 88.36, 94.96, 83.03, 96.15, 55.08, 79.01]
# })

# 计算变量个数
categories = list(df)[:]
N = len(categories)

# 计算每个轴的角度
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# 初始化布局
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 偏移-将第一个轴位于顶部
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# 将每个变量绘制在极坐标上
# 移除默认的xticks标签，我们将用自定义的延伸轴线和标签
ax.set_xticks([])  # 隐藏默认的x轴刻度标签
ax.set_xticklabels([])  # 隐藏默认的x轴标签

# 添加轴线延伸和数据集标签
for i, angle in enumerate(angles[:-1]):
    # 计算延伸线的起点和终点
    start_radius = 100  # 从最外层开始延伸
    end_radius = 110    # 延伸到的位置
    
    # 绘制延伸线
    ax.plot([angle, angle], [start_radius, end_radius], 
            color='lightgray', linewidth=1, alpha=0.8)
    
    # 在延伸线末端添加数据集标签
    label_x = angle
    label_y = end_radius + 3  # 标签位置稍微向外一点
    
    # 根据角度调整标签位置，让两侧的标签向外挪
    angle_normalized = (angle + pi/2) % (2*pi)  # 将角度标准化到0-2π
    
    if angle_normalized < pi/6 or angle_normalized > 11*pi/6:  # 顶部附近
        label_x = angle
        label_y = end_radius + 15
    elif angle_normalized < pi/2:  # 右上
        label_x = angle + 0
        label_y = end_radius + 15
    elif angle_normalized < 5*pi/6:  # 右下
        label_x = angle + 0
        label_y = end_radius + 5
    elif angle_normalized < 7*pi/6:  # 左下
        label_x = angle - 0
        label_y = end_radius + 15
    elif angle_normalized < 3*pi/2:  # 右下
        label_x = angle - 0.1
        label_y = end_radius + 15
    else:  # 顶部偏左
        label_x = angle
        label_y = end_radius + 5
    
    ax.text(label_x, label_y, categories[i], 
            ha='center', va='center', fontsize=DATASET_LABEL_SIZE, fontweight='bold',
            color='black', rotation=0)

# y标签 - 根据每个轴线的原始数据范围显示对应的数值标签
ax.set_rlabel_position(0)

# 为每个轴线计算并显示对应的数值标签
for i, angle in enumerate(angles[:-1]):
    col = categories[i]
    if col == 'group':
        continue
        
    # 获取该轴线的原始数据范围
    original_values = df_mapping[col].values
    min_val = np.min(original_values)
    max_val = np.max(original_values)
    
    # 计算合适的轴线范围
    if min_val < max_val * 0.1:
        axis_min = 0
    else:
        axis_min = max(0, min_val - (max_val - min_val) * 0.1)
    
    axis_max = max_val * 1
    
    # 根据轴线数量动态调整刻度数量
    num_ticks = min(6, N)  # 最多6个刻度，避免过于密集
    
    # 在轴线上添加均匀分布的数值标签
    for j in range(num_ticks):
        if j==0:continue
        # 计算刻度位置（0到100均匀分布）
        tick_val = (j / (num_ticks - 1)) * 100 if num_ticks > 1 else 100
        
        # 将0-100的刻度值映射回原始数据范围
        if tick_val == 0:
            original_tick_val = axis_min
        else:
            original_tick_val = axis_min + (tick_val / 100) * (axis_max - axis_min)
        
        # 计算标签位置
        label_angle = angle
        label_radius = tick_val
        
        # 根据角度动态调整标签位置，避免重叠
        # 计算角度在0-2π中的位置
        angle_normalized = (angle + pi/2) % (2*pi)  # 将角度标准化到0-2π
        
        if angle_normalized < pi/6 or angle_normalized > 11*pi/6:  # 顶部附近
            text_x = label_angle + 0.05
            text_y = label_radius + 2
        elif angle_normalized < pi/2:  # 右上
            text_x = label_angle + 0.1
            text_y = label_radius + 2
        elif angle_normalized < 5*pi/6:  # 右下
            text_x = label_angle + 0.1
            text_y = label_radius - 2
        elif angle_normalized < 7*pi/6:  # 左下
            text_x = label_angle - 0.1
            text_y = label_radius - 2
        elif angle_normalized < 3*pi/2:  # 左上
            text_x = label_angle - 0.1
            text_y = label_radius + 2
        else:  # 顶部偏左
            text_x = label_angle - 0.05
            text_y = label_radius + 2
        
        # 添加数值标签（去掉小框，直接显示数字）
        ax.text(text_x, text_y, f"{original_tick_val:.1f}", 
                color='gray', ha='center', va='center', fontsize=RADIAL_TICK_SIZE-4)

# 隐藏默认的y轴刻度标签
plt.yticks([20, 40, 60, 80, 100], ["", "", "", "", ""], color="grey", size=RADIAL_TICK_SIZE)
plt.ylim(0, 120)  # 增加y轴范围以容纳延伸的轴线和标签

# 移除最外层的圆形轮廓，保留内部网格线
ax.spines['polar'].set_visible(False)

# 重新添加从中心到外层的轴线
for i, angle in enumerate(angles[:-1]):
    # 绘制从中心到最外层的轴线
    ax.plot([0, angle], [0, 100], 
            color='gray', linewidth=0.5, alpha=1)



# 添加多个极坐标图

line_styles = ['-', '-', '-', '-', '-', '-']  # MAX和AVERAGE用虚线，其他用实线

for i, group in enumerate(groups_name):
    values = df.loc[i].values.flatten().tolist()
    values += values[:1]
    
    # 设置线条样式：MAX和AVERAGE用虚线，其他用实线
    linestyle = line_styles[i]
    
    ax.plot(angles, values, linewidth=1.5, label=groups_name[i], 
            c=colors[i], linestyle=linestyle)
    ax.fill(angles, values, colors[i], alpha=0.1)
    
    # 在每个数据点添加圆点标记
    ax.scatter(angles[:-1], values[:-1], color=colors[i], s=15, zorder=3)
    
    # 只在最外围(FIRE)的数据点上显示数值（带偏移量）
    # if group == 'SOTA':
    #     values_2 = df_mapping.loc[i].values.flatten().tolist()
    #     values_2 += values_2[:1]
    #     for j in range(len(values)-1):
    #         angle_offset = 0.1
    #         radius_offset = 5
    #         text_x = angles[j] + angle_offset
    #         text_y = values[j] + radius_offset
    #         if j == 2:
    #             text_x -= 0.1
            
    #         ax.text(text_x, text_y, f"{values_2[j]:.2f}", 
    #                 color='gray', ha='center', va='center', fontsize=10,
    #                 bbox=dict(facecolor='none', alpha=0.7, edgecolor='none', pad=1))

# 设置图例样式
num_items = len(groups_name)
num_cols = num_items if num_items <= 5 else int(np.ceil(num_items / 2))  # ≤5 单行，否则均分为两行以便整体居中
legend = ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=num_cols,
    handlelength=3,
    handletextpad=0.5,
    borderaxespad=0.5,
    frameon=True,
    markerscale=1,
    fontsize=LEGEND_FONT_SIZE,
    columnspacing=1.0,  # 列间距
)

# 修改图例显示：保留原始线条样式，同时添加圆点标记
for i, handle in enumerate(legend.legend_handles):
    # 保持原始线条样式（虚线/实线）
    handle.set_linestyle(line_styles[i])  # 使用之前定义的线条样式
    
    # 添加圆点标记
    handle.set_marker('o')    # 圆点标记
    handle.set_markersize(5)  # 标记大小
    handle.set_markerfacecolor(colors[i])  # 标记填充色
    handle.set_markeredgecolor(colors[i])  # 标记边缘色
    
    # 调整线条和标记的相对位置
    handle.set_linestyle(line_styles[i])  # 重新应用以防被覆盖

plt.tight_layout()
plt.savefig(f'{title}.pdf', bbox_inches='tight')
plt.show()