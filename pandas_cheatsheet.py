"""
================================================================================
                    PANDAS 完全入门指南 - 实用代码集合
================================================================================
包含 Pandas 最常用的 16 个主题的所有操作示例
================================================================================
"""

import pandas as pd
import numpy as np

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', None)


# ================================================================================
# 1. 创建 DataFrame
# ================================================================================

# 方法1: 从字典创建
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
    'salary': [70000, 80000, 90000, 75000, 85000]
})

# 方法2: 从列表创建
data = [
    ['Alice', 25, 'NYC', 70000],
    ['Bob', 30, 'LA', 80000],
    ['Charlie', 35, 'Chicago', 90000]
]
df = pd.DataFrame(data, columns=['name', 'age', 'city', 'salary'])

# 方法3: 从 HuggingFace Dataset
from datasets import load_dataset
dataset = load_dataset("imdb", split="train[:100]")
df = dataset.to_pandas()


# ================================================================================
# 2. 查看数据（最常用）
# ================================================================================

# 查看前几行
df.head()        # 默认前5行
df.head(10)      # 前10行

# 查看后几行
df.tail()        # 默认后5行
df.tail(3)       # 后3行

# 查看随机样本
df.sample(5)     # 随机5行
df.sample(frac=0.1)  # 随机10%的数据

# 查看基本信息
df.info()        # 列类型、非空值数量
df.describe()    # 数值列的统计信息
df.shape         # (行数, 列数)
df.columns       # 列名列表
df.index         # 索引
df.dtypes        # 每列的数据类型

# 查看数据维度
len(df)          # 行数
df.size          # 总元素数（行×列）


# ================================================================================
# 3. 索引和选择数据 ⭐⭐⭐
# ================================================================================

# --- 3.1 选择列 ---

# 单列（返回 Series）
df['name']
df.name          # 也可以，但不推荐（如果列名有空格会出错）

# 多列（返回 DataFrame）
df[['name', 'age']]
df[['name', 'age', 'salary']]

# 选择所有列除了某些
df.drop(['name'], axis=1)  # 删除 name 列
df.drop(columns=['name', 'age'])  # 删除多列


# --- 3.2 选择行 ---

# 按位置选择（iloc - integer location）
df.iloc[0]           # 第1行（返回 Series）
df.iloc[0:3]         # 前3行（0, 1, 2）
df.iloc[[0, 2, 4]]   # 第1, 3, 5行
df.iloc[-1]          # 最后一行
df.iloc[-5:]         # 最后5行

# 按标签选择（loc - label location）
df.loc[0]            # 索引为0的行
df.loc[0:2]          # 索引0到2（包含2！）
df.loc[[0, 2, 4]]    # 索引为0, 2, 4的行


# --- 3.3 同时选择行和列 ⭐ ---

# iloc: 按位置
df.iloc[0, 1]              # 第1行，第2列的值
df.iloc[0:3, 0:2]          # 前3行，前2列
df.iloc[:, 0]              # 所有行，第1列
df.iloc[0, :]              # 第1行，所有列
df.iloc[[0,2], [1,3]]      # 第1,3行 × 第2,4列

# loc: 按标签
df.loc[0, 'name']          # 索引0，name列
df.loc[0:2, 'name']        # 索引0-2，name列
df.loc[:, 'name']          # 所有行，name列
df.loc[0, ['name', 'age']] # 索引0，name和age列
df.loc[0:2, ['name', 'age']]  # 索引0-2，name和age列

# 混合使用（不推荐，但要知道）
df['name'][0]              # 先选列，再选行
df[['name', 'age']][0:3]   # 先选列，再选行


# --- 3.4 条件筛选 ⭐⭐⭐ ---

# 单条件
df[df['age'] > 30]                    # age > 30 的行
df[df['city'] == 'NYC']               # city 是 NYC 的行
df[df['name'].str.contains('a')]      # name 包含 'a' 的行

# 多条件（AND）
df[(df['age'] > 25) & (df['city'] == 'NYC')]

# 多条件（OR）
df[(df['age'] > 30) | (df['city'] == 'LA')]

# 多条件（NOT）
df[~(df['age'] > 30)]                 # age <= 30
df[df['city'] != 'NYC']               # city 不是 NYC

# isin - 在列表中
df[df['city'].isin(['NYC', 'LA'])]    # city 是 NYC 或 LA

# between - 在范围内
df[df['age'].between(25, 30)]         # 25 <= age <= 30

# 字符串条件
df[df['name'].str.startswith('A')]    # name 以 A 开头
df[df['name'].str.endswith('e')]      # name 以 e 结尾
df[df['name'].str.len() > 5]          # name 长度 > 5

# 空值筛选
df[df['age'].isna()]                  # age 是空值
df[df['age'].notna()]                 # age 不是空值


# ================================================================================
# 4. 修改数据
# ================================================================================

# 修改单个值
df.loc[0, 'age'] = 26
df.iloc[0, 1] = 26
df.at[0, 'age'] = 26      # 更快，用于单个值

# 修改整列
df['age'] = df['age'] + 1              # 所有年龄+1
df['age'] = df['age'] * 1.1            # 所有年龄×1.1
df['bonus'] = df['salary'] * 0.1       # 新增列

# 修改多个值
df.loc[df['city'] == 'NYC', 'salary'] = 100000  # NYC的salary改为100000

# 条件赋值
df['senior'] = df['age'] > 30          # 新增布尔列
df['level'] = df['age'].apply(lambda x: 'Senior' if x > 30 else 'Junior')

# 重命名列
df.rename(columns={'name': 'employee_name'}, inplace=True)
df.columns = ['col1', 'col2', 'col3', 'col4']  # 重命名所有列

# 删除列
df.drop('age', axis=1, inplace=True)
df.drop(columns=['age', 'city'], inplace=True)
del df['age']  # 也可以

# 删除行
df.drop(0, axis=0, inplace=True)       # 删除索引0的行
df.drop([0, 1, 2], inplace=True)       # 删除多行
df = df[df['age'] > 25]                # 保留age>25的行


# ================================================================================
# 5. 排序
# ================================================================================

# 按单列排序
df.sort_values('age')                          # 升序
df.sort_values('age', ascending=False)         # 降序

# 按多列排序
df.sort_values(['city', 'age'])                # 先按city，再按age
df.sort_values(['city', 'age'], ascending=[True, False])  # city升序，age降序

# 按索引排序
df.sort_index()                                # 索引升序
df.sort_index(ascending=False)                 # 索引降序

# inplace 参数
df.sort_values('age', inplace=True)            # 直接修改原DataFrame


# ================================================================================
# 6. 分组和聚合 ⭐⭐⭐
# ================================================================================

# 基本分组
df.groupby('city')['salary'].mean()            # 每个城市的平均工资
df.groupby('city')['salary'].sum()             # 每个城市的总工资
df.groupby('city')['salary'].count()           # 每个城市的人数

# 多个聚合函数
df.groupby('city')['salary'].agg(['mean', 'sum', 'count'])

# 对多列聚合
df.groupby('city').agg({
    'salary': ['mean', 'sum'],
    'age': ['mean', 'max', 'min']
})

# 多列分组
df.groupby(['city', 'age'])['salary'].mean()

# 自定义聚合函数
df.groupby('city')['salary'].agg(lambda x: x.max() - x.min())

# 转换为 DataFrame
df.groupby('city')['salary'].mean().reset_index()


# ================================================================================
# 7. 缺失值处理
# ================================================================================

# 检查缺失值
df.isna()                    # 返回布尔DataFrame
df.isna().sum()              # 每列的缺失值数量
df.isna().any()              # 每列是否有缺失值

# 删除缺失值
df.dropna()                  # 删除任何有缺失值的行
df.dropna(axis=1)            # 删除任何有缺失值的列
df.dropna(subset=['age'])    # 删除age列有缺失值的行
df.dropna(how='all')         # 删除全部为缺失值的行

# 填充缺失值
df.fillna(0)                 # 用0填充
df.fillna({'age': 0, 'salary': 50000})  # 不同列用不同值
df['age'].fillna(df['age'].mean())      # 用平均值填充
df.fillna(method='ffill')    # 用前一个值填充
df.fillna(method='bfill')    # 用后一个值填充


# ================================================================================
# 8. 数据类型转换
# ================================================================================

# 查看类型
df.dtypes

# 转换类型
df['age'] = df['age'].astype(int)
df['salary'] = df['salary'].astype(float)
df['city'] = df['city'].astype('category')

# 转换为日期时间
df['date'] = pd.to_datetime(df['date'])

# 转换为字符串
df['age'] = df['age'].astype(str)


# ================================================================================
# 9. 字符串操作
# ================================================================================

# 常用字符串方法（需要用 .str）
df['name'].str.lower()           # 转小写
df['name'].str.upper()           # 转大写
df['name'].str.title()           # 首字母大写
df['name'].str.strip()           # 去除空格
df['name'].str.replace('a', 'A') # 替换
df['name'].str.split(' ')        # 分割
df['name'].str.len()             # 长度
df['name'].str.contains('ali')   # 是否包含
df['name'].str.startswith('A')   # 是否以...开头
df['name'].str.endswith('e')     # 是否以...结尾
df['name'].str[:3]               # 切片（前3个字符）

# 提取
df['name'].str.extract(r'(\w+)')  # 正则提取


# ================================================================================
# 10. 合并和连接
# ================================================================================

# 创建示例数据
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})

# merge（类似SQL JOIN）
pd.merge(df1, df2, on='key')                    # 内连接（默认）
pd.merge(df1, df2, on='key', how='left')        # 左连接
pd.merge(df1, df2, on='key', how='right')       # 右连接
pd.merge(df1, df2, on='key', how='outer')       # 外连接

# concat（拼接）
pd.concat([df1, df2])                           # 垂直拼接（行）
pd.concat([df1, df2], axis=1)                   # 水平拼接（列）
pd.concat([df1, df2], ignore_index=True)        # 重置索引

# join（按索引连接）
df1.join(df2, lsuffix='_left', rsuffix='_right')


# ================================================================================
# 11. 应用函数
# ================================================================================

# apply - 对列或行应用函数
df['age'].apply(lambda x: x * 2)                # 对单列
df.apply(lambda x: x.max() - x.min())           # 对每列
df.apply(lambda x: x['age'] + x['salary'], axis=1)  # 对每行

# map - 映射（仅Series）
df['city'].map({'NYC': 'New York', 'LA': 'Los Angeles'})

# applymap - 对每个元素（已弃用，用 map）
df.map(lambda x: str(x).upper())                # DataFrame的每个元素

# 自定义函数
def categorize_age(age):
    if age < 30:
        return 'Young'
    elif age < 40:
        return 'Middle'
    else:
        return 'Senior'

df['age_group'] = df['age'].apply(categorize_age)


# ================================================================================
# 12. 统计和数学运算
# ================================================================================

# 基本统计
df['age'].mean()         # 平均值
df['age'].median()       # 中位数
df['age'].mode()         # 众数
df['age'].std()          # 标准差
df['age'].var()          # 方差
df['age'].min()          # 最小值
df['age'].max()          # 最大值
df['age'].sum()          # 求和
df['age'].count()        # 计数（非空）
df['age'].nunique()      # 唯一值数量
df['age'].value_counts() # 值计数

# 分位数
df['age'].quantile(0.25)  # 25%分位数
df['age'].quantile([0.25, 0.5, 0.75])  # 多个分位数

# 相关性
df.corr()                # 相关系数矩阵
df['age'].corr(df['salary'])  # 两列相关性

# 累计运算
df['age'].cumsum()       # 累计和
df['age'].cumprod()      # 累计积
df['age'].cummax()       # 累计最大值
df['age'].cummin()       # 累计最小值


# ================================================================================
# 13. 重置和设置索引
# ================================================================================

# 重置索引
df.reset_index()                    # 重置索引，旧索引变成列
df.reset_index(drop=True)           # 重置索引，丢弃旧索引

# 设置索引
df.set_index('name')                # 将name列设为索引
df.set_index(['city', 'name'])      # 多级索引

# 重命名索引
df.index = ['row1', 'row2', 'row3']
df.rename(index={0: 'first', 1: 'second'})


# ================================================================================
# 14. 导入导出
# ================================================================================

# CSV
df.to_csv('output.csv', index=False)           # 保存
df = pd.read_csv('input.csv')                  # 读取

# Excel
df.to_excel('output.xlsx', index=False)
df = pd.read_excel('input.xlsx')

# JSON
df.to_json('output.json')
df = pd.read_json('input.json')

# Parquet
df.to_parquet('output.parquet')
df = pd.read_parquet('input.parquet')

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df.to_sql('table_name', conn, if_exists='replace')
df = pd.read_sql('SELECT * FROM table_name', conn)


# ================================================================================
# 15. 实用技巧 ⭐
# ================================================================================

# 链式操作
result = (df
    .query('age > 25')                    # 筛选
    .sort_values('salary', ascending=False)  # 排序
    .head(10)                             # 取前10
    .reset_index(drop=True)               # 重置索引
)

# 查看内存使用
df.memory_usage(deep=True)

# 复制DataFrame
df_copy = df.copy()                       # 深拷贝
df_view = df                              # 浅拷贝（引用）

# 去重
df.drop_duplicates()                      # 删除重复行
df.drop_duplicates(subset=['name'])       # 基于特定列去重
df['city'].unique()                       # 唯一值数组

# 透视表
df.pivot_table(values='salary', index='city', columns='age', aggfunc='mean')

# 交叉表
pd.crosstab(df['city'], df['age'])


# ================================================================================
# 16. 完整示例：HuggingFace Dataset 处理
# ================================================================================

from datasets import load_dataset
import pandas as pd

# 加载数据
dataset = load_dataset("imdb", split="train[:1000]")
df = dataset.to_pandas()

# 数据探索
print(df.head())
print(df.info())
print(df['label'].value_counts())

# 数据处理
df['text_length'] = df['text'].str.len()           # 文本长度
df['sentiment'] = df['label'].map({0: 'negative', 1: 'positive'})  # 标签映射

# 筛选
long_reviews = df[df['text_length'] > 500]         # 长评论
positive = df[df['label'] == 1]                    # 正面评论

# 分组统计
stats = df.groupby('label').agg({
    'text_length': ['mean', 'median', 'max'],
    'text': 'count'
}).reset_index()

# 采样
sample_df = df.sample(100)                         # 随机100条

# 保存
df.to_csv('processed_imdb.csv', index=False)


# ================================================================================
# 结束 - 掌握这些基本能应对 90% 的数据处理任务！
# ================================================================================
