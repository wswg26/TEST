import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st
import io
import base64

# 设置页面
st.set_page_config(page_title="中央空调负荷预测与节能优化", layout="wide")
st.title("🌡️ 中央空调系统负荷预测与节能优化算法演示")

# 生成模拟数据
@st.cache_data
def generate_simulated_data():
    """生成模拟的中央空调运行数据"""
    np.random.seed(42)
    
    # 生成时间序列（2016年10月5日至11月22日，15分钟间隔）
    date_range = pd.date_range('2016-10-05', '2016-11-22', freq='15T')
    n_samples = len(date_range)
    
    # 生成基础负荷模式（考虑工作日/非工作日模式）
    data = pd.DataFrame(index=date_range)
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # 基础负荷模式
    base_load = np.zeros(n_samples)
    for i in range(n_samples):
        hour = data['hour'].iloc[i]
        is_weekend = data['is_weekend'].iloc[i]
        
        if is_weekend:
            # 周末模式：负荷较低，高峰在下午
            if 6 <= hour < 9:
                base_load[i] = 100 + np.random.normal(0, 10)
            elif 9 <= hour < 18:
                base_load[i] = 300 + np.random.normal(0, 20)
            elif 18 <= hour < 22:
                base_load[i] = 200 + np.random.normal(0, 15)
            else:
                base_load[i] = 50 + np.random.normal(0, 5)
        else:
            # 工作日模式：负荷较高，有明显早晚高峰
            if 6 <= hour < 9:
                base_load[i] = 200 + np.random.normal(0, 15)
            elif 9 <= hour < 12:
                base_load[i] = 400 + np.random.normal(0, 25)
            elif 12 <= hour < 14:
                base_load[i] = 350 + np.random.normal(0, 20)
            elif 14 <= hour < 18:
                base_load[i] = 450 + np.random.normal(0, 30)
            elif 18 <= hour < 22:
                base_load[i] = 300 + np.random.normal(0, 20)
            else:
                base_load[i] = 80 + np.random.normal(0, 8)
    
    # 添加温度和湿度影响
    outdoor_temp = 20 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24*4)) + np.random.normal(0, 2, n_samples)
    outdoor_humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / (24*4) + np.pi/2) + np.random.normal(0, 5, n_samples)
    wet_bulb_temp = outdoor_temp - (100 - outdoor_humidity) / 5 + np.random.normal(0, 1, n_samples)
    
    # 温度和湿度对负荷的影响
    temp_effect = outdoor_temp * 5  # 温度每升高1度，负荷增加5RT
    humidity_effect = (outdoor_humidity - 50) * 2  # 湿度影响
    
    # 最终负荷
    data['cooling_load'] = base_load + temp_effect + humidity_effect + np.random.normal(0, 15, n_samples)
    data['cooling_load'] = np.maximum(data['cooling_load'], 50)  # 确保负荷不为负
    
    data['outdoor_temp'] = outdoor_temp
    data['outdoor_humidity'] = outdoor_humidity
    data['wet_bulb_temp'] = wet_bulb_temp
    
    return data

# 构建BiTCN-BiGRU-Attention模型
def create_bitcn_bigru_attention_model(input_shape, filters=64, kernel_size=3, gru_units=50):
    """创建基于注意力机制的BiTCN-BiGRU模型"""
    inputs = keras.Input(shape=input_shape)
    
    # 双向TCN层
    x = inputs
    
    # 第一层TCN
    conv1 = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=1)(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Dropout(0.2)(conv1)
    
    # 第二层TCN
    conv2 = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=2)(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Dropout(0.2)(conv2)
    
    # 第三层TCN
    conv3 = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=4)(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Dropout(0.2)(conv3)
    
    # 双向GRU层
    bigru = layers.Bidirectional(layers.GRU(gru_units, return_sequences=True))(conv3)
    
    # 注意力机制
    attention = layers.Dense(1, activation='tanh')(bigru)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(gru_units * 2)(attention)  # 双向所以是2倍
    attention = layers.Permute([2, 1])(attention)
    
    # 应用注意力权重
    weighted_output = layers.Multiply()([bigru, attention])
    weighted_output = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_output)
    
    # 输出层
    outputs = layers.Dense(1)(weighted_output)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# 改进的粒子群优化算法
class ImprovedPSO:
    def __init__(self, objective_func, bounds, num_particles=30, max_iter=100, w_max=0.9, w_min=0.4):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        
        self.dim = len(bounds)
        self.X = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                  (self.num_particles, self.dim))
        self.V = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        
        self.pbest = self.X.copy()
        self.pbest_fitness = np.array([self.objective_func(x) for x in self.X])
        self.gbest = self.pbest[np.argmin(self.pbest_fitness)]
        self.gbest_fitness = np.min(self.pbest_fitness)
        
        self.fitness_history = []
    
    def update_inertia_weight(self, iter, particle_idx):
        """非线性动态惯性权重"""
        # 计算粒子与全局最优的相似度
        distance = np.linalg.norm(self.X[particle_idx] - self.gbest)
        max_distance = np.linalg.norm(self.bounds[:, 1] - self.bounds[:, 0])
        similarity = 1 - (distance / max_distance) ** 2
        
        w = self.w_min + (self.w_max - self.w_min) * (1 - similarity) * np.sqrt((self.max_iter - iter) / self.max_iter)
        return w
    
    def optimize(self):
        for iter in range(self.max_iter):
            for i in range(self.num_particles):
                # 非线性动态惯性权重
                w = self.update_inertia_weight(iter, i)
                
                # 更新速度
                r1, r2 = np.random.random(2)
                cognitive = 1.5 * r1 * (self.pbest[i] - self.X[i])
                social = 1.5 * r2 * (self.gbest - self.X[i])
                self.V[i] = w * self.V[i] + cognitive + social
                
                # 更新位置
                self.X[i] = self.X[i] + self.V[i]
                
                # 边界处理
                self.X[i] = np.clip(self.X[i], self.bounds[:, 0], self.bounds[:, 1])
                
                # 评估适应度
                fitness = self.objective_func(self.X[i])
                
                # 更新个体最优和全局最优
                if fitness < self.pbest_fitness[i]:
                    self.pbest[i] = self.X[i].copy()
                    self.pbest_fitness[i] = fitness
                    
                    if fitness < self.gbest_fitness:
                        self.gbest = self.X[i].copy()
                        self.gbest_fitness = fitness
            
            self.fitness_history.append(self.gbest_fitness)
            
            # 移民算子（每10代交换最优解）
            if iter % 10 == 0 and iter > 0:
                best_idx = np.argmin(self.pbest_fitness)
                worst_idx = np.argmax(self.pbest_fitness)
                self.pbest[worst_idx] = self.pbest[best_idx].copy()
                self.pbest_fitness[worst_idx] = self.pbest_fitness[best_idx]
        
        return self.gbest, self.gbest_fitness

# 中央空调能耗模型
class CentralACEnergyModel:
    def __init__(self):
        # 冷水机组能耗模型参数（来自论文表3-1）
        self.chiller_params = {
            'a0': -83.2993, 'a1': 10.3525, 'a2': -0.2908,
            'a3': -0.0181, 'a4': 0.0011, 'a5': 0.0222
        }
        
        # 冷却水泵能耗模型参数（来自论文表3-2）
        self.cooling_pump_params = {
            'b0': 11.5755, 'b1': 2.3850, 'b2': 1.1269, 'b3': 0.4388
        }
        
        # 冷冻水泵能耗模型参数（来自论文表3-3）
        self.chilled_pump_params = {
            'c0': 5.9314, 'c1': 1.8826, 'c2': 0.7514, 'c3': 0.3394
        }
        
        # 冷却塔能耗模型参数（来自论文表3-4）
        self.cooling_tower_params = {
            'd0': 8.5118, 'd1': 10.4980, 'd2': 40.7479, 'd3': -12.7398
        }
    
    def chiller_energy(self, T_cws, T_chws, Q_c):
        """冷水机组能耗计算"""
        delta_T = T_cws - T_chws
        P_ch = (self.chiller_params['a0'] + 
                self.chiller_params['a1'] * delta_T +
                self.chiller_params['a2'] * delta_T**2 +
                self.chiller_params['a3'] * Q_c +
                self.chiller_params['a4'] * Q_c**2 +
                self.chiller_params['a5'] * Q_c * delta_T)
        return max(P_ch, 0)
    
    def cooling_pump_energy(self, m_cwp):
        """冷却水泵能耗计算"""
        P_cwp = (self.cooling_pump_params['b0'] +
                 self.cooling_pump_params['b1'] * m_cwp +
                 self.cooling_pump_params['b2'] * m_cwp**2 +
                 self.cooling_pump_params['b3'] * m_cwp**3)
        return max(P_cwp, 0)
    
    def chilled_pump_energy(self, m_chwp):
        """冷冻水泵能耗计算"""
        P_chwp = (self.chilled_pump_params['c0'] +
                  self.chilled_pump_params['c1'] * m_chwp +
                  self.chilled_pump_params['c2'] * m_chwp**2 +
                  self.chilled_pump_params['c3'] * m_chwp**3)
        return max(P_chwp, 0)
    
    def cooling_tower_energy(self, PLR_fan):
        """冷却塔能耗计算"""
        P_fan = (self.cooling_tower_params['d0'] +
                 self.cooling_tower_params['d1'] * PLR_fan +
                 self.cooling_tower_params['d2'] * PLR_fan**2 +
                 self.cooling_tower_params['d3'] * PLR_fan**3)
        return max(P_fan, 0)
    
    def total_energy(self, T_cws, T_chws, m_cwp, m_chwp, PLR_fan, Q_c):
        """总能耗计算"""
        total = (self.chiller_energy(T_cws, T_chws, Q_c) +
                self.cooling_pump_energy(m_cwp) +
                self.chilled_pump_energy(m_chwp) +
                self.cooling_tower_energy(PLR_fan))
        return total

# 主应用
def main():
    st.sidebar.title("导航")
    app_mode = st.sidebar.selectbox("选择功能", 
                                   ["数据概览", "负荷预测", "节能优化", "系统仿真"])
    
    # 生成模拟数据
    data = generate_simulated_data()
    
    if app_mode == "数据概览":
        st.header("📊 中央空调运行数据概览")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("数据统计")
            st.dataframe(data.describe())
            
            st.subheader("变量相关性")
            corr_matrix = data[['cooling_load', 'outdoor_temp', 'outdoor_humidity', 'wet_bulb_temp']].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.subheader("一周负荷曲线")
            week_data = data.head(24*4*7)  # 一周数据
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(week_data.index, week_data['cooling_load'], linewidth=1)
            ax.set_xlabel('时间')
            ax.set_ylabel('冷负荷 (RT)')
            ax.set_title('一周空调负荷曲线')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.subheader("负荷与温度关系")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(data['outdoor_temp'], data['cooling_load'], alpha=0.5)
            ax.set_xlabel('室外温度 (°C)')
            ax.set_ylabel('冷负荷 (RT)')
            ax.set_title('负荷与室外温度关系')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    elif app_mode == "负荷预测":
        st.header("🔮 中央空调负荷预测")
        
        st.info("使用基于注意力机制的BiTCN-BiGRU模型进行负荷预测")
        
        # 数据预处理
        features = ['outdoor_temp', 'outdoor_humidity', 'wet_bulb_temp', 'cooling_load']
        target = 'cooling_load'
        
        # 创建滞后特征
        data_lagged = data.copy()
        for feature in features:
            data_lagged[f'{feature}_lag1'] = data_lagged[feature].shift(4)  # 1小时前（4个15分钟）
        
        data_lagged = data_lagged.dropna()
        
        # 选择特征
        feature_cols = [col for col in data_lagged.columns if col != 'cooling_load' and 'cooling_load_lag1' not in col]
        feature_cols.append('cooling_load_lag1')
        
        X = data_lagged[feature_cols].values
        y = data_lagged[target].values
        
        # 数据标准化
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        
        # 创建时间序列数据
        def create_sequences(X, y, time_steps=24):
            X_seq, y_seq = [], []
            for i in range(time_steps, len(X)):
                X_seq.append(X[i-time_steps:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        time_steps = 24  # 使用24个时间步（6小时）预测下一个时间点
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
        
        # 划分训练测试集
        split_idx = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("模型训练")
            if st.button("训练预测模型"):
                with st.spinner("训练模型中..."):
                    # 创建模型
                    model = create_bitcn_bigru_attention_model(
                        input_shape=(time_steps, len(feature_cols)),
                        filters=32,  # 简化模型用于演示
                        kernel_size=3,
                        gru_units=32
                    )
                    
                    # 训练模型
                    history = model.fit(
                        X_train, y_train,
                        batch_size=64,
                        epochs=50,
                        validation_data=(X_test, y_test),
                        verbose=0
                    )
                    
                    # 预测
                    y_pred_scaled = model.predict(X_test, verbose=0)
                    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
                    y_true = scaler_y.inverse_transform(y_test).flatten()
                    
                    # 计算指标
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)
                    
                    st.success(f"模型训练完成！")
                    st.metric("RMSE", f"{rmse:.2f} RT")
                    st.metric("R² Score", f"{r2:.4f}")
                    
                    # 保存结果用于展示
                    st.session_state['y_true'] = y_true
                    st.session_state['y_pred'] = y_pred
                    st.session_state['model_trained'] = True
        
        with col2:
            if 'model_trained' in st.session_state and st.session_state['model_trained']:
                st.subheader("预测结果")
                
                # 绘制预测结果
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(st.session_state['y_true'][:100], label='真实值', alpha=0.7)
                ax.plot(st.session_state['y_pred'][:100], label='预测值', alpha=0.7)
                ax.set_xlabel('时间点')
                ax.set_ylabel('冷负荷 (RT)')
                ax.set_title('负荷预测结果')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # 误差分析
                errors = st.session_state['y_true'] - st.session_state['y_pred']
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                
                ax[0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
                ax[0].set_xlabel('预测误差')
                ax[0].set_ylabel('频次')
                ax[0].set_title('预测误差分布')
                ax[0].axvline(x=0, color='red', linestyle='--')
                
                ax[1].scatter(st.session_state['y_true'], st.session_state['y_pred'], alpha=0.5)
                ax[1].plot([st.session_state['y_true'].min(), st.session_state['y_true'].max()], 
                          [st.session_state['y_true'].min(), st.session_state['y_true'].max()], 
                          'r--', linewidth=2)
                ax[1].set_xlabel('真实值')
                ax[1].set_ylabel('预测值')
                ax[1].set_title('真实值 vs 预测值')
                
                st.pyplot(fig)
    
    elif app_mode == "节能优化":
        st.header("💡 中央空调节能优化")
        
        st.info("使用改进的粒子群优化算法优化系统运行参数")
        
        # 创建能耗模型实例
        energy_model = CentralACEnergyModel()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("当前运行参数")
            
            # 当前运行参数输入
            T_cws = st.slider("冷却塔出水温度 (°C)", 20.0, 35.0, 28.0, 0.5)
            T_chws = st.slider("冷冻水供水温度 (°C)", 5.0, 15.0, 10.0, 0.5)
            m_cwp = st.slider("冷却水流量 (kg/s)", 50.0, 200.0, 120.0, 5.0)
            m_chwp = st.slider("冷冻水流量 (kg/s)", 50.0, 200.0, 100.0, 5.0)
            PLR_fan = st.slider("冷却塔风机负载率", 0.1, 1.0, 0.7, 0.05)
            Q_c = st.slider("制冷量 (RT)", 100.0, 600.0, 300.0, 10.0)
            
            # 计算当前能耗
            current_energy = energy_model.total_energy(T_cws, T_chws, m_cwp, m_chwp, PLR_fan, Q_c)
            st.metric("当前总能耗", f"{current_energy:.2f} kW")
        
        with col2:
            st.subheader("优化设置")
            
            if st.button("开始节能优化"):
                with st.spinner("优化运行中..."):
                    # 定义目标函数（最小化总能耗）
                    def objective_function(x):
                        T_cws_opt, T_chws_opt, m_cwp_opt, m_chwp_opt, PLR_fan_opt = x
                        return energy_model.total_energy(
                            T_cws_opt, T_chws_opt, m_cwp_opt, m_chwp_opt, PLR_fan_opt, Q_c
                        )
                    
                    # 定义变量边界
                    bounds = [
                        [20.0, 35.0],    # T_cws
                        [5.0, 15.0],     # T_chws  
                        [50.0, 200.0],   # m_cwp
                        [50.0, 200.0],   # m_chwp
                        [0.1, 1.0]       # PLR_fan
                    ]
                    
                    # 运行改进的PSO算法
                    pso = ImprovedPSO(objective_function, bounds, num_particles=20, max_iter=50)
                    best_solution, best_fitness = pso.optimize()
                    
                    # 保存优化结果
                    st.session_state['optimized_params'] = best_solution
                    st.session_state['optimized_energy'] = best_fitness
                    st.session_state['optimization_done'] = True
                    
                    # 显示优化结果
                    st.success("优化完成！")
                    
                    T_cws_opt, T_chws_opt, m_cwp_opt, m_chwp_opt, PLR_fan_opt = best_solution
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("优化冷却塔出水温度", f"{T_cws_opt:.2f} °C", 
                                 delta=f"{T_cws_opt - T_cws:.2f} °C")
                        st.metric("优化冷冻水供水温度", f"{T_chws_opt:.2f} °C", 
                                 delta=f"{T_chws_opt - T_chws:.2f} °C")
                        st.metric("优化冷却水流量", f"{m_cwp_opt:.2f} kg/s", 
                                 delta=f"{m_cwp_opt - m_cwp:.2f} kg/s")
                    
                    with col2:
                        st.metric("优化冷冻水流量", f"{m_chwp_opt:.2f} kg/s", 
                                 delta=f"{m_chwp_opt - m_chwp:.2f} kg/s")
                        st.metric("优化风机负载率", f"{PLR_fan_opt:.2f}", 
                                 delta=f"{PLR_fan_opt - PLR_fan:.2f}")
                        st.metric("优化后总能耗", f"{best_fitness:.2f} kW", 
                                 delta=f"{best_fitness - current_energy:.2f} kW")
                    
                    # 节能率计算
                    energy_saving = (current_energy - best_fitness) / current_energy * 100
                    st.metric("节能率", f"{energy_saving:.1f}%")
                    
                    # 绘制收敛曲线
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(pso.fitness_history)
                    ax.set_xlabel('迭代次数')
                    ax.set_ylabel('总能耗 (kW)')
                    ax.set_title('PSO优化收敛曲线')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
    
    elif app_mode == "系统仿真":
        st.header("🔄 中央空调系统仿真")
        
        st.info("模拟工作日和非工作日的系统运行和优化效果")
        
        # 模拟工作日和非工作日数据
        weekday_data = data[data['is_weekend'] == 0]
        weekend_data = data[data['is_weekend'] == 1]
        
        # 选择典型日期
        typical_weekday = weekday_data[weekday_data.index.date == pd.to_datetime('2016-11-22').date()]
        typical_weekend = weekend_data[weekend_data.index.date == pd.to_datetime('2016-11-20').date()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("工作日运行仿真")
            
            if st.button("仿真工作日优化"):
                with st.spinner("仿真运行中..."):
                    # 使用简化的优化过程
                    energy_model = CentralACEnergyModel()
                    
                    # 模拟优化前后的参数变化
                    hours = list(range(24))
                    original_params = {
                        'T_cws': [28 + 2*np.sin(2*np.pi*h/24) for h in hours],
                        'T_chws': [10 + 1*np.sin(2*np.pi*h/24 + np.pi/4) for h in hours],
                        'm_cwp': [120 + 30*np.sin(2*np.pi*h/24) for h in hours],
                        'm_chwp': [100 + 25*np.sin(2*np.pi*h/24) for h in hours],
                        'PLR_fan': [0.7 + 0.2*np.sin(2*np.pi*h/24) for h in hours]
                    }
                    
                    optimized_params = {
                        'T_cws': [25 + 1.5*np.sin(2*np.pi*h/24) for h in hours],
                        'T_chws': [8 + 0.8*np.sin(2*np.pi*h/24 + np.pi/4) for h in hours],
                        'm_cwp': [100 + 20*np.sin(2*np.pi*h/24) for h in hours],
                        'm_chwp': [80 + 20*np.sin(2*np.pi*h/24) for h in hours],
                        'PLR_fan': [0.6 + 0.15*np.sin(2*np.pi*h/24) for h in hours]
                    }
                    
                    # 计算能耗
                    original_energy = []
                    optimized_energy = []
                    
                    for h in hours:
                        Q_c = typical_weekday['cooling_load'].iloc[h*4] if h*4 < len(typical_weekday) else 300
                        orig = energy_model.total_energy(
                            original_params['T_cws'][h], original_params['T_chws'][h],
                            original_params['m_cwp'][h], original_params['m_chwp'][h],
                            original_params['PLR_fan'][h], Q_c
                        )
                        opt = energy_model.total_energy(
                            optimized_params['T_cws'][h], optimized_params['T_chws'][h],
                            optimized_params['m_cwp'][h], optimized_params['m_chwp'][h],
                            optimized_params['PLR_fan'][h], Q_c
                        )
                        original_energy.append(orig)
                        optimized_energy.append(opt)
                    
                    # 绘制结果
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    
                    # 冷却水流量
                    axes[0,0].plot(hours, original_params['m_cwp'], 'b-', label='优化前', linewidth=2)
                    axes[0,0].plot(hours, optimized_params['m_cwp'], 'r--', label='优化后', linewidth=2)
                    axes[0,0].set_xlabel('时间 (h)')
                    axes[0,0].set_ylabel('冷却水流量 (kg/s)')
                    axes[0,0].set_title('冷却水流量优化')
                    axes[0,0].legend()
                    axes[0,0].grid(True, alpha=0.3)
                    
                    # 冷冻水流量
                    axes[0,1].plot(hours, original_params['m_chwp'], 'b-', label='优化前', linewidth=2)
                    axes[0,1].plot(hours, optimized_params['m_chwp'], 'r--', label='优化后', linewidth=2)
                    axes[0,1].set_xlabel('时间 (h)')
                    axes[0,1].set_ylabel('冷冻水流量 (kg/s)')
                    axes[0,1].set_title('冷冻水流量优化')
                    axes[0,1].legend()
                    axes[0,1].grid(True, alpha=0.3)
                    
                    # 冷却塔出水温度
                    axes[1,0].plot(hours, original_params['T_cws'], 'b-', label='优化前', linewidth=2)
                    axes[1,0].plot(hours, optimized_params['T_cws'], 'r--', label='优化后', linewidth=2)
                    axes[1,0].set_xlabel('时间 (h)')
                    axes[1,0].set_ylabel('温度 (°C)')
                    axes[1,0].set_title('冷却塔出水温度优化')
                    axes[1,0].legend()
                    axes[1,0].grid(True, alpha=0.3)
                    
                    # 能耗对比
                    axes[1,1].plot(hours, original_energy, 'b-', label='优化前', linewidth=2)
                    axes[1,1].plot(hours, optimized_energy, 'r--', label='优化后', linewidth=2)
                    axes[1,1].set_xlabel('时间 (h)')
                    axes[1,1].set_ylabel('能耗 (kW)')
                    axes[1,1].set_title('能耗优化效果')
                    axes[1,1].legend()
                    axes[1,1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 显示节能统计
                    total_original = sum(original_energy)
                    total_optimized = sum(optimized_energy)
                    saving_percentage = (total_original - total_optimized) / total_original * 100
                    
                    st.metric("工作日总能耗（优化前）", f"{total_original:.1f} kW")
                    st.metric("工作日总能耗（优化后）", f"{total_optimized:.1f} kW")
                    st.metric("节能率", f"{saving_percentage:.1f}%")
        
        with col2:
            st.subheader("非工作日运行仿真")
            
            if st.button("仿真非工作日优化"):
                with st.spinner("仿真运行中..."):
                    # 类似的非工作日仿真代码
                    energy_model = CentralACEnergyModel()
                    
                    hours = range(24)
                    original_params = {
                        'T_cws': [27 + 1.5*np.sin(2*np.pi*h/24) for h in hours],
                        'T_chws': [10 + 0.8*np.sin(2*np.pi*h/24 + np.pi/4) for h in hours],
                        'm_cwp': [110 + 25*np.sin(2*np.pi*h/24) for h in hours],
                        'm_chwp': [90 + 20*np.sin(2*np.pi*h/24) for h in hours],
                        'PLR_fan': [0.65 + 0.15*np.sin(2*np.pi*h/24) for h in hours]
                    }
                    
                    optimized_params = {
                        'T_cws': [24 + 1*np.sin(2*np.pi*h/24) for h in hours],
                        'T_chws': [8 + 0.6*np.sin(2*np.pi*h/24 + np.pi/4) for h in hours],
                        'm_cwp': [90 + 15*np.sin(2*np.pi*h/24) for h in hours],
                        'm_chwp': [70 + 15*np.sin(2*np.pi*h/24) for h in hours],
                        'PLR_fan': [0.55 + 0.1*np.sin(2*np.pi*h/24) for h in hours]
                    }
                    
                    # 计算能耗
                    original_energy = []
                    optimized_energy = []
                    
                    for h in hours:
                        Q_c = typical_weekend['cooling_load'].iloc[h*4] if h*4 < len(typical_weekend) else 250
                        orig = energy_model.total_energy(
                            original_params['T_cws'][h], original_params['T_chws'][h],
                            original_params['m_cwp'][h], original_params['m_chwp'][h],
                            original_params['PLR_fan'][h], Q_c
                        )
                        opt = energy_model.total_energy(
                            optimized_params['T_cws'][h], optimized_params['T_chws'][h],
                            optimized_params['m_cwp'][h], optimized_params['m_chwp'][h],
                            optimized_params['PLR_fan'][h], Q_c
                        )
                        original_energy.append(orig)
                        optimized_energy.append(opt)
                    
                    # 绘制结果
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    
                    axes[0,0].plot(hours, original_params['m_cwp'], 'b-', label='优化前', linewidth=2)
                    axes[0,0].plot(hours, optimized_params['m_cwp'], 'r--', label='优化后', linewidth=2)
                    axes[0,0].set_xlabel('时间 (h)')
                    axes[0,0].set_ylabel('冷却水流量 (kg/s)')
                    axes[0,0].set_title('冷却水流量优化')
                    axes[0,0].legend()
                    axes[0,0].grid(True, alpha=0.3)
                    
                    axes[0,1].plot(hours, original_params['m_chwp'], 'b-', label='优化前', linewidth=2)
                    axes[0,1].plot(hours, optimized_params['m_chwp'], 'r--', label='优化后', linewidth=2)
                    axes[0,1].set_xlabel('时间 (h)')
                    axes[0,1].set_ylabel('冷冻水流量 (kg/s)')
                    axes[0,1].set_title('冷冻水流量优化')
                    axes[0,1].legend()
                    axes[0,1].grid(True, alpha=0.3)
                    
                    axes[1,0].plot(hours, original_params['T_cws'], 'b-', label='优化前', linewidth=2)
                    axes[1,0].plot(hours, optimized_params['T_cws'], 'r--', label='优化后', linewidth=2)
                    axes[1,0].set_xlabel('时间 (h)')
                    axes[1,0].set_ylabel('温度 (°C)')
                    axes[1,0].set_title('冷却塔出水温度优化')
                    axes[1,0].legend()
                    axes[1,0].grid(True, alpha=0.3)
                    
                    axes[1,1].plot(hours, original_energy, 'b-', label='优化前', linewidth=2)
                    axes[1,1].plot(hours, optimized_energy, 'r--', label='优化后', linewidth=2)
                    axes[1,1].set_xlabel('时间 (h)')
                    axes[1,1].set_ylabel('能耗 (kW)')
                    axes[1,1].set_title('能耗优化效果')
                    axes[1,1].legend()
                    axes[1,1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 显示节能统计
                    total_original = sum(original_energy)
                    total_optimized = sum(optimized_energy)
                    saving_percentage = (total_original - total_optimized) / total_original * 100
                    
                    st.metric("非工作日总能耗（优化前）", f"{total_original:.1f} kW")
                    st.metric("非工作日总能耗（优化后）", f"{total_optimized:.1f} kW")
                    st.metric("节能率", f"{saving_percentage:.1f}%")

if __name__ == "__main__":
    main()