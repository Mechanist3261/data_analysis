import pandas as pd
from sklearn.preprocessing import StandardScaler

# =====================
# 1. 读取数据
# =====================
base_path = r"C:\Users\Cypress\Desktop\data1.0"

demo = pd.read_stata(base_path + r"\Demographic_Background.dta")
health = pd.read_stata(base_path + r"\Health_Status_and_Functioning.dta")
family = pd.read_stata(base_path + r"\Family_Information.dta")
income = pd.read_stata(base_path + r"\Household_Income.dta")

print("原始数据维度：")
print(demo.shape, health.shape, family.shape, income.shape)

# =====================
# 2. 变量筛选
# =====================

demo = demo[[
    "ID", "householdID",
    "ba001",   # 性别
    "ba002"    # 年龄
]]

health = health[[
    "ID",
    "da001",       # 自评健康
    "da002_1_",    # 慢病1
    "da002_2_"     # 慢病2
]]

family = family[[
    "householdID",
    "ca001"    # 家庭人数
]]

income = income[[
    "householdID",
    "gb001"    # 家庭总收入
]]

# =====================
# 3. 合并数据
# =====================
df = demo.merge(health, on="ID", how="left") \
         .merge(family, on="householdID", how="left") \
         .merge(income, on="householdID", how="left")

print("\n合并后维度：", df.shape)

# =====================
# 4. 构建特征,保留缺失值.
# =====================

X = df.drop(columns=["ID", "householdID"])

# 转数值（无法转换的变NaN）
X = X.apply(pd.to_numeric, errors='coerce')

# 生成“缺失标记变量”
for col in X.columns:
    X[col + "_missing"] = X[col].isna().astype(int)

# 用0填充（只是占位，不影响缺失信息）
X = X.fillna(0)

print("\n加缺失标记后：", X.shape)

# =====================
# 5. 标准化
# =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("标准化后：", X_scaled.shape)

# =====================
# 6. 保存数据
# =====================
df.to_csv("clean_data.csv", index=False)

print("\n✅ 数据处理完成！")
print("原始df：", df.shape)
print("建模数据：", X_scaled.shape)