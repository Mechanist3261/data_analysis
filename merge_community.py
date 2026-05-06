import pandas as pd
import numpy as np
import os

base_path = r"C:\Users\Cypress\Desktop\data1.0"

# =========================
# 1️⃣ 读取数据
# =========================
df = pd.read_csv(r"D:\大学\code\data_analysis\clean_data.csv")

sample_df = pd.read_stata(os.path.join(base_path, "Sample_Infor.dta"))
sample_df = sample_df[['ID', 'communityID']]

# ID类型统一
df['ID'] = df['ID'].astype(str)
sample_df['ID'] = sample_df['ID'].astype(str)

# 合并
df = df.merge(sample_df, on='ID', how='left')

print("合并后维度：", df.shape)
print("communityID缺失数：", df['communityID'].isna().sum())

# 删除没有社区ID的数据（必须）
df = df.dropna(subset=['communityID'])

# =========================
# 2️⃣ 构造 da002（关键修复）
# =========================
# 先定义函数
def extract_num(x):
    try:
        return float(str(x).split()[0])
    except:
        return np.nan

# 先把 da002_1_ 和 da002_2_ 转成数值
df['da002_1_'] = df['da002_1_'].apply(extract_num)
df['da002_2_'] = df['da002_2_'].apply(extract_num)

# 再求平均
df['da002'] = df[['da002_1_', 'da002_2_']].mean(axis=1)

# =========================
# 3️⃣ 选变量
# =========================
df_use = df[['communityID', 'ba001', 'ba002', 'da001', 'da002', 'ca001', 'gb001']].copy()

# =========================
# 4️⃣ 字符 → 数值
# =========================
def extract_num(x):
    try:
        return float(str(x).split()[0])
    except:
        return np.nan

cols = ['ba001', 'ba002', 'da001', 'da002', 'ca001', 'gb001']
for col in cols:
    df_use[col] = df_use[col].apply(extract_num)

# =========================
# 5️⃣ 填补缺失（不要drop！）
# =========================
df_use[cols] = df_use[cols].fillna(df_use[cols].mean())

print("清洗后数据：", df_use.shape)

# =========================
# 6️⃣ 聚合到社区
# =========================
community_df = df_use.groupby('communityID').mean()

print("社区数据维度：", community_df.shape)

# =========================
# 7️⃣ 构造需求强度（升级版）
# =========================
community_df['need_score'] = (
    0.25 * community_df['da001'] +              # 慢病
    0.25 * (5 - community_df['ba002']) +        # 健康（反向）
    0.2 * community_df['da002'] +               # 健康变化
    0.15 * community_df['ca001'] +              # 家庭支持
    0.15 * community_df['gb001']                # 社会角色
)

# =========================
# 8️⃣ 五级分类
# =========================
bins = np.percentile(community_df['need_score'], [20, 40, 60, 80])

community_df['need_level'] = np.digitize(
    community_df['need_score'],
    bins=bins
)

# =========================
# 9️⃣ 输出结果
# =========================
print("\n需求分数示例：")
print(community_df[['need_score']].head())

print("\n分类分布：")
print(community_df['need_level'].value_counts().sort_index())

print("\n最终数据维度：", community_df.shape)

# （可选）保存
community_df.to_csv(r"D:\大学\code\data_analysis\community_data.csv")