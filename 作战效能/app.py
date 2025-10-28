import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 页面设置
st.set_page_config(page_title="足球运动员价值预测器", layout="wide")
st.title("足球运动员市场价值预测")

# 特征名称（与训练时一致）
FEATURE_NAMES = [
    'height_cm', "weight_kg", 'potential', 'pac', 'sho', 'pas', 'dri',
    'def', 'phy', 'international_reputation', 'skill_moves', 'weak_foot',
    'preferred_foot', "crossing", "finishing", "heading_accuracy"
]


# 加载模型和归一化参数
@st.cache_resource
def load_resources():
    model = load("xgb_model.pkl")
    min_vals = load("min_vals.pkl")
    max_vals = load("max_vals.pkl")
    ranges = load("ranges.pkl")
    return model, min_vals, max_vals, ranges


model, min_vals, max_vals, ranges = load_resources()


# 归一化函数
def normalize_data(df):
    return (df - min_vals[FEATURE_NAMES]) / ranges[FEATURE_NAMES]


# 反归一化函数（用于预测结果）
def denormalize_y(y_norm):
    return y_norm * ranges["y"] + min_vals["y"]


# 侧边栏 - 参数设置
st.sidebar.header("模型参数")
st.sidebar.write(f"**学习率**: {model.learning_rate}")
st.sidebar.write(f"**最大深度**: {model.max_depth}")
st.sidebar.write(f"**树数量**: {model.n_estimators}")
st.sidebar.write(f"**特征数量**: {len(FEATURE_NAMES)}")

# 主界面
tab1, tab2 = st.tabs(["单球员预测", "批量预测"])

with tab1:
    st.header("单球员市场价值预测")
    st.write("输入球员特征进行预测:")

    # 创建4列布局
    col1, col2, col3, col4 = st.columns(4)

    # 分组特征
    features = {}
    with col1:
        features['height_cm'] = st.number_input("身高 (cm)", min_value=150, max_value=220, value=180)
        features['weight_kg'] = st.number_input("体重 (kg)", min_value=50, max_value=100, value=75)
        features['potential'] = st.number_input("潜力值", min_value=0, max_value=100, value=80)
        features['pac'] = st.number_input("速度 (PAC)", min_value=0, max_value=100, value=75)

    with col2:
        features['sho'] = st.number_input("射门 (SHO)", min_value=0, max_value=100, value=75)
        features['pas'] = st.number_input("传球 (PAS)", min_value=0, max_value=100, value=75)
        features['dri'] = st.number_input("盘带 (DRI)", min_value=0, max_value=100, value=75)
        features['def'] = st.number_input("防守 (DEF)", min_value=0, max_value=100, value=65)

    with col3:
        features['phy'] = st.number_input("身体 (PHY)", min_value=0, max_value=100, value=75)
        features['international_reputation'] = st.number_input("国际声誉", min_value=1, max_value=5, value=3)
        features['skill_moves'] = st.number_input("技巧动作", min_value=1, max_value=5, value=3)
        features['weak_foot'] = st.number_input("弱脚能力", min_value=1, max_value=5, value=3)

    with col4:
        features['preferred_foot'] = st.selectbox("惯用脚", ["左脚", "右脚"])
        features['crossing'] = st.number_input("传中能力", min_value=0, max_value=100, value=70)
        features['finishing'] = st.number_input("终结能力", min_value=0, max_value=100, value=75)
        features['heading_accuracy'] = st.number_input("头球精度", min_value=0, max_value=100, value=70)

    # 转换惯用脚为数值
    preferred_foot_map = {"左脚": 0, "右脚": 1}
    features['preferred_foot'] = preferred_foot_map[features['preferred_foot']]

    # 组合输入特征（按特征名称顺序）
    input_data = pd.DataFrame([features], columns=FEATURE_NAMES)

    if st.button("预测球员价值"):
        # 应用归一化处理
        input_data_norm = normalize_data(input_data)

        # 预测归一化值
        prediction_norm = model.predict(input_data_norm)

        # 反归一化得到原始值
        prediction = denormalize_y(prediction_norm[0])

        # 显示结果
        st.success(f"预测球员市场价值: €{prediction:.2f}百万")

with tab2:
    st.header("批量球员预测")
    st.info("上传包含球员数据的CSV文件进行批量预测")

    # 显示所需特征格式
    with st.expander("查看所需数据格式"):
        st.write("CSV文件应包含以下列（顺序不限）:")
        st.code(", ".join(FEATURE_NAMES))
        st.write("示例数据:")
        example_data = {name: [min_vals[name]] for name in FEATURE_NAMES}
        st.dataframe(pd.DataFrame(example_data))

    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])

    if uploaded_file is not None:
        # 读取数据
        df = pd.read_csv(uploaded_file)

        # 检查特征是否完整
        missing_features = set(FEATURE_NAMES) - set(df.columns)
        if missing_features:
            st.error(f"文件缺少以下特征列: {', '.join(missing_features)}")
        else:
            st.write("数据预览:", df.head())

            # 确保特征顺序正确
            df_features = df[FEATURE_NAMES]

            # 预测
            if st.button("执行批量预测"):
                # 应用归一化处理
                df_norm = normalize_data(df_features)

                # 预测归一化值
                predictions_norm = model.predict(df_norm)

                # 反归一化得到原始值
                predictions = denormalize_y(predictions_norm)

                result_df = df.copy()
                result_df['预测价值(百万欧元)'] = predictions

                # 显示结果
                st.subheader("预测结果")
                st.dataframe(result_df)

                # 下载结果
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="下载预测结果",
                    data=csv,
                    file_name='球员价值预测结果.csv',
                    mime='text/csv'
                )

                # 如果有真实值列（假设列名为'actual_value'）
                if 'actual_value' in df.columns:
                    y_true = df['actual_value']
                    y_pred = predictions

                    st.subheader("模型评估指标")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("均方误差 (MSE)", f"{mean_squared_error(y_true, y_pred):.4f}")
                    with col2:
                        st.metric("平均绝对误差 (MAE)", f"{mean_absolute_error(y_true, y_pred):.4f}")
                    with col3:
                        st.metric("决定系数 (R^2)", f"{r2_score(y_true, y_pred):.4f}")

# 添加说明
st.sidebar.header("使用说明")
st.sidebar.info("""
1. **单球员预测**：输入球员特征值进行实时预测
2. **批量预测**：上传CSV文件进行批量预测
3. 模型预测结果为球员市场价值（单位：百万欧元）
4. 所有输入数据会自动进行归一化处理
5. 预测结果已反归一化为原始值
""")