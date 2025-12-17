import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from scipy.signal import spectrogram
import obspy
from obspy.core.trace import Trace
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import joblib
import psutil
import logging
import gc
import time
from contextlib import contextmanager


@tf.keras.utils.register_keras_serializable()
class QualityAwareAttentionLayer(layers.Layer):
    """质量感知注意力层 - 支持masking"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.content_dense1 = layers.Dense(64, activation='relu')
        self.content_dense2 = layers.Dense(1, activation='linear')
        self.supports_masking = True  # 🛠️ 修复：声明支持masking

    def build(self, input_shape):
        feature_shape, feat_input_shape = input_shape
        # 统一转 tuple，避免 tuple + list
        feature_shape = tf.TensorShape(feature_shape)
        self.content_dense1.build(feature_shape)
        # 计算 1 维输出形状
        dense1_output_shape = feature_shape[:-1] + (64,)
        self.content_dense2.build(dense1_output_shape)
        super().build(input_shape)

    def call(self, inputs):
        features, feat_inputs = inputs

        # 基础内容注意力
        content_attention = self.content_dense1(features)
        content_attention = self.content_dense2(content_attention)  # (batch_size, num_traces, 1)

        # 从特征输入中提取质量分数（第10维）
        quality_scores = feat_inputs[:, :, -1]  # (batch_size, num_traces)
        quality_scores = tf.expand_dims(quality_scores, axis=-1)  # (batch_size, num_traces, 1)

        # 用质量分数调整注意力
        quality_adjustment = tf.where(
            quality_scores >= 0,
            tf.exp(quality_scores * 2),  # 正质量：指数增强
            tf.sigmoid(quality_scores * 10)  # 负质量：sigmoid抑制到接近0
        )

        adjusted_attention = content_attention * quality_adjustment
        adjusted_attention = tf.squeeze(adjusted_attention, axis=-1)  # (batch_size, num_traces)

        # softmax归一化
        attention_weights = tf.nn.softmax(adjusted_attention, axis=1)  # (batch_size, num_traces)

        return tf.expand_dims(attention_weights, -1)  # (batch_size, num_traces, 1)

    def compute_mask(self, inputs, mask=None):
        # 🛠️ 修复：正确处理mask
        if mask is not None:
            return mask[0]  # 返回第一个输入的mask
        return None

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
def quality_function(y_true, y_pred):
    """
    质量函数: quality = (2 * y - 1) * (2 * p - 1)
    值域 [-1, 1]，正确→正，错误→负，确信→|q|→1
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return (2.0 * y_true - 1.0) * (2.0 * y_pred - 1.0)


@tf.keras.utils.register_keras_serializable()
def bulletproof_trace_loss(y_true, y_pred):
    """修复的Trace级损失函数"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    # 创建有效样本掩码（排除填充的-1）
    valid_mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)

    # 二元交叉熵
    bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

    # 只对有效样本计算损失
    bce = bce * valid_mask
    valid_count = tf.reduce_sum(valid_mask)

    # 防止除零
    loss = tf.cond(valid_count > 0,
                   lambda: tf.reduce_sum(bce) / valid_count,
                   lambda: tf.constant(0.0))

    return tf.maximum(loss, 0.0)


@tf.keras.utils.register_keras_serializable()
def bulletproof_event_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-3, 1 - 1e-3)
    eps = 1e-7
    loss = - (y_true * tf.math.log(y_pred + eps) +
              (1 - y_true) * tf.math.log(1 - y_pred + eps))
    return tf.maximum(tf.reduce_mean(loss), 0.0)
# ========== 新增：优化器封装 + 负 loss 早期停训 ==========
def build_safe_optimizer(lr):
    return keras.optimizers.Adam(
        learning_rate=lr,
        global_clipnorm=1.0,   # 全局范数裁剪
        epsilon=1e-6           # 防止除以 0
    )


class KillOnNegativeLoss(Callback):
    def __init__(self, patience=3):
        super().__init__()
        self.patience = patience
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        main_loss = logs.get('loss')
        if main_loss is not None and main_loss < 0:
            self.count += 1
            print(f"\n KillOnNegativeLoss: 第 {self.count} 次负 loss ({main_loss:.4f})")
            if self.count >= self.patience:
                print(" 停止训练，检查损失函数！")
                self.model.stop_training = True

@tf.keras.utils.register_keras_serializable()
def stable_weighted_binary_crossentropy(class_weights):
    """数值稳定的加权二元交叉熵"""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-5, 1.0 - 1e-5)

        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        weights = tf.where(
            tf.equal(y_true, 1),
            class_weights[1],
            class_weights[0]
        )

        loss = tf.reduce_mean(bce * weights)
        loss = tf.maximum(loss, 0.0)  # 确保不会出现负数

        return loss
    return loss_fn


# 设置GPU内存增长，避免OOM错误
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"已配置GPU内存自增长: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")

# 确保Eager Execution已启用
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()
    print("已启用Eager Execution模式")

# -------------------------- 配置参数 --------------------------
RESULT_OUTPUT_PATH = "/home/he/PycharmProjects/PythonProject/test_results_improved.txt"
TRAIN_PATH = "/home/he/PycharmProjects/PythonProject/dataset/processed_comcat/train"
VAL_PATH = "/home/he/PycharmProjects/PythonProject/dataset/processed_comcat/val"
TEST_SETS = [
    ("General Test Set", "/home/he/PycharmProjects/PythonProject/dataset/processed_comcat/test"),
]
WAVEFORM_PATH = "/home/he/PycharmProjects/PythonProject/dataset/PNW-ML/comcat_waveforms.hdf5"
SAVE_MODEL_PATH = "/home/he/PycharmProjects/PythonProject/improved_earthquake_classifier.keras"
TRACE_HISTORY_PLOT_PATH = "/home/he/PycharmProjects/PythonProject/trace_training_history.png"
EVENT_HISTORY_PLOT_PATH = "/home/he/PycharmProjects/PythonProject/event_training_history.png"
TRACE_ATTENTION_HEATMAP_PATH = "/home/he/PycharmProjects/PythonProject/trace_weight_.png"
TRACE_PERFORMANCE_PATH = "/home/he/PycharmProjects/PythonProject/trace_performance.png"
CONFUSION_MATRIX_PATH = "/home/he/PycharmProjects/PythonProject/confusion_matrix.png"
ERROR_LOG_PATH = "data_processing_errors.log"
CHECKPOINT_DIR = "/home/he/PycharmProjects/PythonProject/checkpoints"

# 信号处理参数
SAMPLE_RATE = 75
WAVEFORM_DURATION = 60
WAVEFORM_LENGTH = int(SAMPLE_RATE * WAVEFORM_DURATION)
HIGHPASS_FREQ = 2
VALID_COMPONENTS = ["Z"]

# 事件处理参数
MIN_TRACES_PER_EVENT = 1
MAX_TRACES_PER_EVENT = None

# 频谱图参数
SPECTROGRAM_NPERS = int(2 * SAMPLE_RATE)
SPECTROGRAM_NOVER = int(SPECTROGRAM_NPERS * 0.75)
SPECTROGRAM_FREQ_MIN = 0.5
SPECTROGRAM_FREQ_MAX = 50

# 预计算频谱图维度
_f, _t, _ = spectrogram(
    x=np.zeros(WAVEFORM_LENGTH),
    fs=SAMPLE_RATE,
    nperseg=SPECTROGRAM_NPERS,
    noverlap=SPECTROGRAM_NOVER,
    nfft=SPECTROGRAM_NPERS
)
_spec_freq_mask = (_f >= SPECTROGRAM_FREQ_MIN) & (_f <= SPECTROGRAM_FREQ_MAX)
SPEC_HEIGHT = int(np.sum(_spec_freq_mask))
SPEC_WIDTH = int(len(_t))
print(f"预计算频谱图维度: (高度={SPEC_HEIGHT}, 宽度={SPEC_WIDTH})")
del _f, _t, _spec_freq_mask

# 训练参数

BATCH_SIZE = 32
TRACE_BATCH_SIZE = 32
EPOCHS = 400
TRACE_PRETRAIN_EPOCHS = 250
EXPLOSION_WEIGHT_SCALE = 1
LEARNING_RATE = 1e-5
TRACE_LEARNING_RATE = 1e-5
MAX_ERRORS = 1000
ERROR_LOG_INTERVAL = 100
INTERMEDIATE_LOSS_WEIGHT = 0.05
NUM_ATTENTION_HEADS = 4

# 列名映射
COLUMN_MAPPING = {
    "event_id": "event_id",
    "event_type": "event_type",
    "trace_name": "trace_name",
    "trace_P_arrival_sample": "trace_P_arrival_sample",
    "trace_S_arrival_sample": "trace_S_arrival_sample",
    "trace_start_time": "trace_start_time",
    "trace_sampling_rate_hz": "trace_sampling_rate_hz",
    "mag": "mag",
    "source_depth_km": "source_depth_km",
    "origin_time": "origin_time",
    "source_latitude_deg": "source_latitude_deg",
    "source_longitude_deg": "source_longitude_deg",
    "station_latitude_deg": "station_latitude_deg",
    "station_longitude_deg": "station_longitude_deg"
}



# 初始化错误日志
logging.basicConfig(
    filename=ERROR_LOG_PATH,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
error_logger = logging.getLogger("data_processor")

# 创建检查点目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -------------------------- 自定义可序列化Keras层 --------------------------
@tf.keras.utils.register_keras_serializable()
class TraceAttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)                       # (B, T, 32)
        x = self.dense2(x)                            # (B, T, 1)
        # 显式指定 softmax 轴，并返回与输入相同的 shape
        return tf.nn.softmax(x, axis=1)

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class MultiHeadTraceAttention(layers.Layer):
    def __init__(self, num_heads=NUM_ATTENTION_HEADS, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        # 每个头独立打分
        self.attention_heads = [layers.Dense(1, activation='sigmoid') for _ in range(num_heads)]
        self.fuse = layers.Dense(1)  # 把 num_heads 个分数融为 1 个

    def call(self, inputs, training=None):
        # inputs: (B, T, D)
        head_scores = [head(inputs) for head in self.attention_heads]  # 列表，每个 (B, T, 1)
        stacked = tf.concat(head_scores, axis=-1)                      # (B, T, num_heads)
        raw_score = self.fuse(stacked)                                # (B, T, 1)
        raw_score = tf.squeeze(raw_score, axis=-1)                    # (B, T)

        # ── L1 归一化：保持“分数越高越好”且和为 1 ──
        weight = raw_score / (tf.reduce_sum(raw_score, axis=1, keepdims=True) + 1e-8)
        return tf.expand_dims(weight, -1)  # (B, T, 1)  与旧接口一致

    def get_config(self):
        return dict(list(super().get_config().items()) + [('num_heads', self.num_heads)])

@tf.keras.utils.register_keras_serializable()
class ExpandWeightsLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        weights, features = inputs
        # 用 Reshape 显式加轴，避免 Lambda 无 output_shape
        expanded_weights = tf.reshape(weights, tf.concat([tf.shape(weights), [1]], 0))
        return expanded_weights * tf.ones_like(features)

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class ExpandLastDim(layers.Layer):
    """显式替代 Lambda(lambda x: tf.expand_dims(x, axis=-1))"""

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape + (1,)

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class FusedVectorLayer(layers.Layer):
    """自定义层：将加权后的特征按trace维度求和"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        fused = tf.reduce_sum(inputs, axis=1)  # (batch_size, features)
        return fused

    def get_config(self):
        config = super().get_config()
        return config


CUSTOM_OBJECTS = {
    'TraceAttentionLayer': TraceAttentionLayer,
    'MultiHeadTraceAttention': MultiHeadTraceAttention,
    'ExpandWeightsLayer': ExpandWeightsLayer,
    'FusedVectorLayer': FusedVectorLayer,
    'ExpandLastDim': ExpandLastDim,
    'QualityAwareAttentionLayer': QualityAwareAttentionLayer,  # 确保包含
    'event_loss': bulletproof_event_loss,
    'trace_loss': bulletproof_trace_loss,
    'weighted_binary_crossentropy': stable_weighted_binary_crossentropy,
    'quality_function': quality_function,
}
# -------------------------- H5FileManager 类 --------------------------
class H5FileManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self._file = None
        self._is_open = False

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HDF5文件未找到: {file_path}")
        if not os.path.isfile(file_path):
            raise IsADirectoryError(f"{file_path} 是目录，不是文件")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"HDF5文件为空: {file_path}")

    def open(self):
        if self._is_open:
            return self._file

        self.close()
        try:
            self._file = h5py.File(self.file_path, "r", swmr=True)
            self._is_open = True
            if len(list(self._file.keys())) == 0:
                raise RuntimeError(f"HDF5文件损坏或为空: {self.file_path}")
            return self._file
        except Exception as e:
            raise RuntimeError(f"打开HDF5文件失败: {str(e)}") from e

    def close(self):
        if self._is_open and self._file is not None:
            try:
                self._file.close()
            except Exception as e:
                error_logger.error(f"HDF5关闭错误: {str(e)}")
            finally:
                self._file = None
                self._is_open = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


# -------------------------- 工具函数 --------------------------
def split_feat_vector(feat_vector):
    """
    把 8 维特征向量拆成三部分：
      norm_part : 要归一化的 6 维  [frac, mag, log(P/S), hour_sin, hour_cos, weekday]
      dist      : 震中距（km）—— 不再归一化
      depth     : 震源深度（km）—— 不再归一化
    """
    norm_part = np.array([feat_vector[0],   # 分形维
                          feat_vector[1],   # 震级
                          feat_vector[3],   # log(P/S)
                          feat_vector[5],   # hour_sin
                          feat_vector[6],   # hour_cos
                          feat_vector[7]], dtype=np.float32)  # weekday
    dist  = feat_vector[4]   # 震中距
    depth = feat_vector[2]   # 震源深度
    return norm_part, dist, depth


def merge_feat_vector(norm_part, dist, depth):
    """
    把 6 维归一化部分 + 原始震中距 + 原始深度 重新拼回 8 维
    顺序与 split_feat_vector 严格对应：
      [frac, mag, depth, log(P/S), dist, hour_sin, hour_cos, weekday]
    """
    return np.array([norm_part[0],  # 0  fractal
                     norm_part[1],  # 1  mag
                     depth,         # 2  depth（km）—— 未归一化
                     norm_part[2],  # 3  log(P/S)
                     dist,          # 4  distance（km）—— 未归一化
                     norm_part[3],  # 5  hour_sin
                     norm_part[4],  # 6  hour_cos
                     norm_part[5]], # 7  weekday
                    dtype=np.float32)

def print_memory_usage(prefix=""):
    try:
        process = psutil.Process(os.getpid())
        mem_used = process.memory_info().rss / (1024 ** 3)
        mem_percent = process.memory_percent()
        print(f"{prefix}内存使用: {mem_used:.2f} GB ({mem_percent:.1f}%)")
    except Exception:
        pass


@contextmanager
def timing_context(description):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{description} 耗时: {end_time - start_time:.2f} 秒")


def parse_trace_name(trace_name_str):
    if "$" not in trace_name_str:
        raise ValueError(f"无效的trace_name格式: {trace_name_str} (缺少$分隔符)")
    bucket_part = trace_name_str.split("$")[0]
    event_idx_part = trace_name_str.split("$")[1].split(",")[0]
    try:
        event_idx = int(event_idx_part.strip())
    except ValueError:
        raise ValueError(f"trace_name中的索引无效: {trace_name_str} (索引部分: {event_idx_part})")
    return f"data/{bucket_part}", event_idx


def get_component_order(h5_file):
    comp_key = "data_format/component_order"
    if comp_key not in h5_file:
        raise KeyError(f"HDF5缺少必要的分量键: {comp_key} (文件中所有键: {list(h5_file.keys())})")
    comp_str = h5_file[comp_key][()].decode("utf-8").strip().upper()
    raw_comp = comp_str.split(",") if "," in comp_str else [c for c in comp_str]
    comp_map = {"E": "X", "N": "Y", "Z": "Z"}
    try:
        comp_order = [comp_map[c.strip()] for c in raw_comp]
    except KeyError as e:
        raise ValueError(f"不支持的分量: {e} (HDF5中可用分量: {raw_comp})")
    if VALID_COMPONENTS[0] not in comp_order:
        raise ValueError(
            f"HDF5中未找到目标分量 {VALID_COMPONENTS[0]} (可用分量: {comp_order})")
    return comp_order


def calculate_arrival_time(start_dt, arrival_sample, sampling_rate):
    if arrival_sample < 0:
        raise ValueError(f"到达样本数为负: {arrival_sample}")
    if sampling_rate <= 0:
        raise ValueError(f"无效的采样率: {sampling_rate} (必须为正数)")
    arrival_offset = arrival_sample / sampling_rate
    return start_dt + timedelta(seconds=arrival_offset)


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点之间的地球表面距离（公里）"""
    R = 6371.0  # 地球半径（公里）
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def preprocess_waveform(raw_wave, raw_sr, p_arrival, s_arrival, start_dt, is_training=False):
    if len(raw_wave) == 0:
        raise ValueError("原始波形数据为空")
    if np.all(raw_wave == 0):
        raise ValueError("原始波形全为零 (无有效信号)")
    trace = Trace(
        data=raw_wave.copy(),
        header={"sampling_rate": raw_sr, "starttime": obspy.UTCDateTime(start_dt)}
    )
    trace.detrend("demean").detrend("linear")
    trace.taper(max_percentage=0.05, type="hann")
    nyquist_freq = raw_sr / 2.0
    actual_highpass = min(HIGHPASS_FREQ, nyquist_freq - 0.1)
    if actual_highpass > 0:
        trace.filter('highpass', freq=actual_highpass, corners=4, zerophase=True)
    if trace.stats.sampling_rate != SAMPLE_RATE:
        trace.resample(SAMPLE_RATE, no_filter=True)
    max_amp = np.max(np.abs(trace.data))
    if max_amp > 0:
        trace.data /= max_amp
    else:
        trace.data = np.zeros_like(trace.data)
    p_time_rel = (p_arrival - start_dt).total_seconds()
    start_idx = max(0, int((p_time_rel - 15) * SAMPLE_RATE))
    end_idx = start_idx + WAVEFORM_LENGTH
    if end_idx > len(trace.data):
        end_idx = len(trace.data)
        start_idx = max(0, end_idx - WAVEFORM_LENGTH)
    processed_wave = trace.data[start_idx:end_idx]
    if len(processed_wave) < WAVEFORM_LENGTH:
        pad_length = WAVEFORM_LENGTH - len(processed_wave)
        processed_wave = np.pad(processed_wave, (0, pad_length), mode="constant")

    # 🛠️ 修复：只在训练时添加随机噪声
    if is_training:
        noise = np.random.normal(0, 0.01, size=processed_wave.shape)
        processed_wave = processed_wave + noise

    sp_delay = (s_arrival - p_arrival).total_seconds()
    if sp_delay < 1.0:
        pg_window = 0.3
        sg_window = 0.6
    elif 1.0 <= sp_delay < 5.0:
        pg_window = 0.4
        sg_window = 0.8
    else:
        pg_window = 0.5
        sg_window = min(1.0, sp_delay * 0.5)
    pg_start_rel = p_time_rel - pg_window
    pg_end_rel = p_time_rel + pg_window
    pg_start_idx = max(0, int(pg_start_rel * SAMPLE_RATE))
    pg_end_idx = min(len(trace.data), int(pg_end_rel * SAMPLE_RATE))
    pg_amp = np.max(np.abs(trace.data[pg_start_idx:pg_end_idx])) if (pg_end_idx > pg_start_idx) else 1e-6
    pg_amp = max(pg_amp, 1e-6)
    sg_start_rel = (s_arrival - start_dt).total_seconds() - sg_window
    sg_end_rel = (s_arrival - start_dt).total_seconds() + sg_window
    sg_start_idx = max(0, int(sg_start_rel * SAMPLE_RATE))
    sg_end_idx = min(len(trace.data), int(sg_end_rel * SAMPLE_RATE))
    sg_amp = np.max(np.abs(trace.data[sg_start_idx:sg_end_idx])) if (sg_end_idx > sg_start_idx) else 1e-6
    sg_amp = max(sg_amp, 1e-6)
    pg_sg_ratio = pg_amp / sg_amp
    pg_sg_ratio = np.clip(pg_sg_ratio, np.exp(-5), np.exp(5))
    log_pg_sg = np.log(pg_sg_ratio)
    if np.isnan(log_pg_sg):
        log_pg_sg = 0.0
    return processed_wave, log_pg_sg


def calculate_spectrogram(waveform_data):
    if len(waveform_data) != WAVEFORM_LENGTH:
        raise RuntimeError(f"无效的波形长度: 预期 {WAVEFORM_LENGTH}, 实际 {len(waveform_data)}")
    f, t, Sxx = spectrogram(
        x=waveform_data,
        fs=SAMPLE_RATE,
        nperseg=SPECTROGRAM_NPERS,
        noverlap=SPECTROGRAM_NOVER,
        nfft=SPECTROGRAM_NPERS
    )
    freq_mask = (f >= SPECTROGRAM_FREQ_MIN) & (f <= SPECTROGRAM_FREQ_MAX)
    Sxx_filtered = Sxx[freq_mask, :]
    if Sxx_filtered.shape != (SPEC_HEIGHT, SPEC_WIDTH):
        raise RuntimeError(
            f"无效的频谱图维度! 预期 ({SPEC_HEIGHT},{SPEC_WIDTH}), 实际 {Sxx_filtered.shape} "
        )
    Sxx_max = np.max(Sxx_filtered)
    if Sxx_max > 0:
        Sxx_filtered /= Sxx_max
    return Sxx_filtered.astype(np.float32)


def calculate_fractal_dimension(waveform):
    n = len(waveform)
    if n < 100:
        return 1.0

    scales = np.logspace(1, np.log10(n // 4), 10, dtype=int)
    scales = scales[scales > 0]
    if len(scales) < 2:
        return 1.0

    Ns = []
    for scale in scales:
        reshaped = waveform[:n // scale * scale].reshape(-1, scale)
        ranges = np.ptp(reshaped, axis=1)
        Ns.append(np.sum(ranges) / scale)

    log_scales = np.log(1.0 / scales)
    log_Ns = np.log(np.maximum(Ns, 1e-6))
    valid_mask = np.isfinite(log_scales) & np.isfinite(log_Ns)
    if np.sum(valid_mask) < 2:
        return 1.0

    coeffs = np.polyfit(log_scales[valid_mask], log_Ns[valid_mask], 1)
    # ✅ 防止 NaN
    if not np.isfinite(coeffs[0]):
        return 1.0
    return np.clip(coeffs[0], 1.0, 2.0)


def parse_time_str(time_str):
    try:
        if "+00:00" in time_str:
            time_str = time_str.split("+00:00")[0].strip()
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f" if "." in time_str else "%Y-%m-%d %H:%M:%S")
        elif "T" in time_str:
            time_str = time_str.replace("Z", "").strip()
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f" if "." in time_str else "%Y-%m-%dT%H:%M:%S")
        else:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f" if "." in time_str else "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        raise ValueError(f"时间解析失败: {time_str} (错误: {e})")


def extract_time_features(event_time_str):
    dt = parse_time_str(event_time_str)
    return [
        np.sin(2 * np.pi * dt.hour / 24),
        np.cos(2 * np.pi * dt.hour / 24),
        1 if dt.weekday() < 5 else 0
    ]


def dynamic_padding(traces_list, max_traces):
    """
    0 向量填充 + 返回 mask
    返回: (padded_traces, mask)  mask: 1=真实 0=填充
    """
    if isinstance(traces_list, np.ndarray):
        traces_list = traces_list.tolist()

    current_count = len(traces_list)
    if current_count >= max_traces:
        return traces_list[:max_traces], np.ones(max_traces, dtype=np.float32)

    if current_count == 0:
        raise ValueError("dynamic_padding 输入为空")

    padded = traces_list.copy()
    mask   = [1.0] * current_count

    # ✅ 用 0 向量填充，避免分布漂移
    zero_sample = np.zeros_like(traces_list[0])
    for _ in range(max_traces - current_count):
        padded.append(zero_sample.copy())
        mask.append(0.0)

    return padded, np.array(mask, dtype=np.float32)

def validate_h5_paths(h5_manager, metadata):
    with h5_manager as h5_file:
        missing_paths = []
        for _, row in metadata.iterrows():
            trace_name = row["trace_name"]
            hdf5_path, _ = parse_trace_name(trace_name)
            if hdf5_path not in h5_file:
                missing_paths.append(hdf5_path)
        if missing_paths:
            unique_missing = set(missing_paths)
            print(f"警告: 发现 {len(unique_missing)} 个不存在的HDF5路径:")
            for path in unique_missing:
                print(f"  - {path}")
        return len(missing_paths) == 0


def check_h5_integrity(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            buckets = [f"data/bucket{i}" for i in range(1, 11)]
            for bucket in buckets:
                if bucket not in f:
                    print(f"缺少bucket: {bucket}")
                else:
                    print(f"找到bucket: {bucket}, 形状: {f[bucket].shape}")
            return True
    except Exception as e:
        print(f"HDF5完整性检查失败: {e}")
        return False


# -------------------------- 数据加载函数 --------------------------
def load_metadata_from_split(path):
    metadata_filename = "metadata.csv"
    metadata_path = os.path.join(path, metadata_filename)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件未找到: {metadata_path}")

    required_columns = list(COLUMN_MAPPING.values())
    if "event_id" not in required_columns:
        required_columns.append("event_id")

    metadata = pd.read_csv(metadata_path, usecols=required_columns)

    missing_cols = [col for col in required_columns if col not in metadata.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")

    valid_event_types = ["earthquake", "explosion"]
    valid_mask = (
            metadata["event_type"].isin(valid_event_types) &
            metadata["trace_P_arrival_sample"].notna() &
            metadata["trace_S_arrival_sample"].notna() &
            metadata["trace_start_time"].notna() &
            metadata["trace_sampling_rate_hz"].notna() &
            metadata["mag"].notna() &
            metadata["source_depth_km"].notna() &
            metadata["origin_time"].notna() &
            metadata["source_latitude_deg"].notna() &
            metadata["source_longitude_deg"].notna() &
            metadata["station_latitude_deg"].notna() &
            metadata["station_longitude_deg"].notna()
    )
    metadata = metadata[valid_mask].copy()
    if len(metadata) == 0:
        raise ValueError(f"{path} 中没有有效样本")

    event_dist = metadata["event_type"].value_counts().to_dict()
    print(
        f"已加载 {path}: 共 {len(metadata)} 个有效样本 (分布: 地震={event_dist.get('earthquake', 0)}, 爆炸={event_dist.get('explosion', 0)})")

    reverse_mapping = {v: k for k, v in COLUMN_MAPPING.items()}
    metadata_renamed = metadata.rename(columns=reverse_mapping)

    if "event_id" not in metadata_renamed.columns and "event_id" in metadata.columns:
        metadata_renamed["event_id"] = metadata["event_id"]

    return metadata_renamed


def group_metadata_by_event(metadata):
    """
    按事件ID分组元数据 - 保留单trace事件
    """
    global MAX_TRACES_PER_EVENT

    event_id_groups = metadata.groupby("event_id").groups
    all_events = []

    # 详细统计
    total_events = len(event_id_groups)
    events_discarded = 0
    trace_count_distribution = {}

    print(f"\n=== 事件分组详细统计 (保留单trace事件) ===")
    print(f"原始事件总数: {total_events}")

    for event_id, trace_indices in event_id_groups.items():
        traces = metadata.loc[trace_indices]
        event_type = traces["event_type"].iloc[0]
        trace_count = len(traces)

        # 记录trace数量分布
        if trace_count not in trace_count_distribution:
            trace_count_distribution[trace_count] = 0
        trace_count_distribution[trace_count] += 1

        # 关键修改：保留所有事件，包括单trace事件
        all_events.append((event_type, traces))

    # 动态设置 MAX_TRACES_PER_EVENT（基于所有事件）
    if all_events:
        trace_counts = [len(traces) for _, traces in all_events]
        proposed_max = int(np.percentile(trace_counts, 98))
        # 确保至少能处理单trace事件
        MAX_TRACES_PER_EVENT = max(1, proposed_max)
    else:
        MAX_TRACES_PER_EVENT = 1  # 最小值为1

    # 打印详细统计
    print(f"保留的事件总数: {len(all_events)}")
    print(f"被丢弃的事件数: {events_discarded}")
    print(f"利用率: {len(all_events) / total_events * 100:.1f}%")
    print(f"动态设置 MAX_TRACES_PER_EVENT = {MAX_TRACES_PER_EVENT}")

    # 打印trace数量分布
    print(f"\nTrace数量分布:")
    for count in sorted(trace_count_distribution.keys()):
        events_with_count = trace_count_distribution[count]
        percentage = events_with_count / total_events * 100
        print(f"  {count} traces: {events_with_count} 事件 ({percentage:.1f}%) [全部保留]")

    # 事件类型分布
    event_types = [event_type for event_type, _ in all_events]
    unique_types, counts = np.unique(event_types, return_counts=True)
    type_dist = dict(zip(unique_types, counts))
    print(f"\n最终事件类型分布: {type_dist}")

    np.random.shuffle(all_events)
    return all_events

def is_valid_trace(waveform):
    """
    检查波形数据是否有效
    """
    if len(waveform) == 0:
        return False
    if np.all(waveform == 0):
        return False
    if np.std(waveform) < 1e-8:  # 几乎恒定的信号
        return False
    if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
        return False
    return True


def trace_generator(metadata, h5_manager, scaler=None, shuffle=False, is_training=False):
    """生成单个 trace 的样本 - 修复特征维度问题"""
    if shuffle:
        metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)

    total_samples = len(metadata)
    error_count = 0
    success_count = 0
    empty_trace_count = 0

    with h5_manager as h5_file:
        comp_order_cache = None
        for idx, (_, row) in enumerate(metadata.iterrows()):
            if idx % 1000 == 0:
                print(
                    f"处理进度: {idx}/{total_samples} (成功: {success_count}, 失败: {error_count}, 空trace: {empty_trace_count})")

            trace_name = row["trace_name"]
            try:
                if comp_order_cache is None:
                    comp_order_cache = get_component_order(h5_file)
                target_comp_idx = comp_order_cache.index(VALID_COMPONENTS[0])
                hdf5_path, event_idx = parse_trace_name(trace_name)

                if hdf5_path not in h5_file:
                    error_count += 1
                    continue

                wave_group = h5_file[hdf5_path]
                if event_idx < 0 or event_idx >= wave_group.shape[0]:
                    error_count += 1
                    continue

                raw_wave = wave_group[event_idx, target_comp_idx, :].copy()
                if not is_valid_trace(raw_wave):
                    empty_trace_count += 1
                    continue

                start_dt = parse_time_str(row["trace_start_time"])
                raw_sr = float(row["trace_sampling_rate_hz"])
                p_arrival = calculate_arrival_time(start_dt, int(row["trace_P_arrival_sample"]), raw_sr)
                s_arrival = calculate_arrival_time(start_dt, int(row["trace_S_arrival_sample"]), raw_sr)
                #修复：传入 is_training 参数
                processed_wave, log_pg_sg = preprocess_waveform(raw_wave, raw_sr, p_arrival, s_arrival, start_dt, is_training)
                del raw_wave

                if len(processed_wave) != WAVEFORM_LENGTH:
                    error_count += 1
                    continue

                spec_data = calculate_spectrogram(processed_wave)
                processed_wave = np.expand_dims(processed_wave, axis=-1)
                spec_data = np.expand_dims(spec_data, axis=-1)

                fractal_dim = calculate_fractal_dimension(processed_wave[:, 0])
                epicentral_distance = haversine_distance(
                    row["source_latitude_deg"], row["source_longitude_deg"],
                    row["station_latitude_deg"], row["station_longitude_deg"]
                )
                time_features = extract_time_features(row["origin_time"])

                #  修复：构建8维特征向量（单Trace训练阶段）
                feat_vector = np.array([
                    fractal_dim,
                    float(row["mag"]),
                    float(row["source_depth_km"]),
                    log_pg_sg,
                    epicentral_distance,
                    time_features[0],
                    time_features[1],
                    time_features[2]
                ], dtype=np.float32)

                # 归一化处理
                if scaler is not None:
                    norm_part, dist, depth = split_feat_vector(feat_vector)
                    norm_part = scaler.transform(norm_part.reshape(1, -1)).flatten()
                    feat_vector = merge_feat_vector(norm_part, dist, depth)

                label = 1 if row["event_type"] == "earthquake" else 0

                success_count += 1
                #  修复：输出8维特征向量
                yield (processed_wave, spec_data, feat_vector), np.array(label, dtype=np.int8)

            except Exception as e:
                error_count += 1
                continue

    print(f"处理完成: 共 {success_count} 个成功, {error_count} 个失败, {empty_trace_count} 个空trace")


def event_generator(event_groups, h5_path, scaler=None, shuffle=False, trace_prob_cache=None, is_training=False):
    """事件生成器：修复特征维度问题"""
    if trace_prob_cache is None:
        raise RuntimeError("必须传入 trace_prob_cache！")
    if shuffle:
        event_groups = event_groups.copy()
        np.random.shuffle(event_groups)

    with h5py.File(h5_path, 'r', swmr=True) as h5_file:
        comp_order_cache = None
        for event_type, traces in event_groups:
            event_waves, event_specs, event_feats = [], [], []
            valid_traces_count = 0

            for _, row in traces.iterrows():
                try:
                    trace_name = row["trace_name"]
                    if comp_order_cache is None:
                        comp_order_cache = get_component_order(h5_file)
                    target_comp_idx = comp_order_cache.index(VALID_COMPONENTS[0])
                    hdf5_path, event_idx = parse_trace_name(trace_name)

                    if hdf5_path not in h5_file:
                        continue

                    wave_group = h5_file[hdf5_path]

                    if event_idx < 0 or event_idx >= wave_group.shape[0]:
                        continue

                    raw_wave = wave_group[event_idx, target_comp_idx, :].copy()

                    if not is_valid_trace(raw_wave):
                        continue

                    # 处理波形数据
                    start_dt = parse_time_str(row["trace_start_time"])
                    raw_sr = float(row["trace_sampling_rate_hz"])
                    p_arrival = calculate_arrival_time(start_dt, int(row["trace_P_arrival_sample"]), raw_sr)
                    s_arrival = calculate_arrival_time(start_dt, int(row["trace_S_arrival_sample"]), raw_sr)
                    #  修复：传入 is_training 参数
                    processed_wave, log_pg_sg = preprocess_waveform(raw_wave, raw_sr, p_arrival, s_arrival, start_dt, is_training)
                    spec_data = calculate_spectrogram(processed_wave)
                    processed_wave = np.expand_dims(processed_wave, axis=-1)
                    spec_data = np.expand_dims(spec_data, axis=-1)

                    fractal_dim = calculate_fractal_dimension(processed_wave[:, 0])
                    epicentral_distance = haversine_distance(
                        row["source_latitude_deg"], row["source_longitude_deg"],
                        row["station_latitude_deg"], row["station_longitude_deg"])
                    time_features = extract_time_features(row["origin_time"])

                    #  修复：构建8维特征向量
                    feat_vector = np.array([
                        fractal_dim,
                        float(row["mag"]),
                        float(row["source_depth_km"]),
                        log_pg_sg,
                        epicentral_distance,
                        time_features[0],
                        time_features[1],
                        time_features[2]
                    ], dtype=np.float32)

                    # 归一化处理
                    if scaler is not None:
                        norm_part, dist, depth = split_feat_vector(feat_vector)
                        norm_part = scaler.transform(norm_part.reshape(1, -1)).flatten()
                        feat_vector = merge_feat_vector(norm_part, dist, depth)

                    # 获取trace模型预测概率
                    if trace_name not in trace_prob_cache:
                        continue

                    trace_prob = trace_prob_cache[trace_name]

                    # 计算质量分数
                    true_label = 1 if row["event_type"] == "earthquake" else 0
                    quality_score = float((2 * true_label - 1) * (2 * trace_prob - 1))

                    # 🛠️ 修复：构建10维特征向量（8维原始特征 + trace概率 + 质量分数）
                    enhanced_feat_vector = np.append(feat_vector, [trace_prob, quality_score])

                    event_waves.append(processed_wave)
                    event_specs.append(spec_data)
                    event_feats.append(enhanced_feat_vector)
                    valid_traces_count += 1

                    del processed_wave, spec_data, raw_wave

                except Exception as e:
                    continue

            # 只要有有效trace就处理
            if valid_traces_count > 0:
                event_waves_np = np.stack(event_waves, axis=0)
                event_specs_np = np.stack(event_specs, axis=0)
                event_feats_np = np.stack(event_feats, axis=0)
                event_label = 1 if event_type == "earthquake" else 0
                trace_labels = np.full((valid_traces_count,), event_label, dtype=np.int8)

                yield (event_waves_np, event_specs_np, event_feats_np), \
                    (np.int8(event_label), trace_labels)


def build_trace_tf_dataset(metadata, h5_manager, scaler=None, shuffle=False, batch_size=TRACE_BATCH_SIZE, is_training=False):
    """构建单trace的TF数据集 - 修复特征维度问题"""

    #  修复：输出形状调整：特征维度改为8
    output_types = ((tf.float32, tf.float32, tf.float32), tf.int8)
    output_shapes = (
        (
            tf.TensorShape([WAVEFORM_LENGTH, 1]),
            tf.TensorShape([SPEC_HEIGHT, SPEC_WIDTH, 1]),
            tf.TensorShape([8])  # 改为8维!
        ),
        tf.TensorShape([])
    )

    def generator_factory():
        return trace_generator(metadata, h5_manager, scaler, shuffle, is_training)

    dataset = tf.data.Dataset.from_generator(
        generator_factory,
        output_types=output_types,
        output_shapes=output_shapes
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_event_tf_dataset(event_groups, h5_path, scaler=None, shuffle=False,
                           batch_size=BATCH_SIZE, trace_prob_cache=None, is_training=False):
    """构建事件级TF数据集：修复填充逻辑"""

    max_tr = MAX_TRACES_PER_EVENT or 10
    print(f"[DEBUG] 构建事件数据集: {len(event_groups)} 个事件, batch_size={batch_size}")

    if trace_prob_cache is None:
        raise RuntimeError("必须传入 trace_prob_cache！")

    def gen():
        for (waves, specs, feats), (y_event, y_trace) in event_generator(
                event_groups=event_groups,
                h5_path=h5_path,
                scaler=scaler,
                shuffle=shuffle,
                trace_prob_cache=trace_prob_cache,
                is_training=is_training  # 🛠️ 修复：传入 is_training 参数
        ):
            n_real = waves.shape[0]

            # 修复：确保所有数组都有正确的形状
            if n_real > max_tr:
                # 如果trace数量超过最大值，截断
                waves = waves[:max_tr]
                specs = specs[:max_tr]
                feats = feats[:max_tr]
                y_trace = y_trace[:max_tr]
            elif n_real < max_tr:
                # 如果trace数量不足，填充
                pad = max_tr - n_real

                # 获取正确的填充形状
                wave_pad_shape = ((0, pad), (0, 0), (0, 0))
                spec_pad_shape = ((0, pad), (0, 0), (0, 0), (0, 0))
                feat_pad_shape = ((0, pad), (0, 0))

                waves = np.pad(waves, wave_pad_shape, 'constant')
                specs = np.pad(specs, spec_pad_shape, 'constant')
                feats = np.pad(feats, feat_pad_shape, 'constant')
                y_trace = np.pad(y_trace, (0, pad), constant_values=-1)

            # 修复：确保返回正确的数据类型
            yield (waves.astype(np.float32),
                   specs.astype(np.float32),
                   feats.astype(np.float32)), \
                (np.int8(y_event), y_trace.astype(np.int8))

    # 输出类型和形状
    output_types = (
        (tf.float32, tf.float32, tf.float32),
        (tf.int8, tf.int8)
    )
    output_shapes = (
        (tf.TensorShape([None, WAVEFORM_LENGTH, 1]),
         tf.TensorShape([None, SPEC_HEIGHT, SPEC_WIDTH, 1]),
         tf.TensorShape([None, 10])),
        (tf.TensorShape([]),
         tf.TensorShape([None]))
    )

    dataset = tf.data.Dataset.from_generator(
        gen, output_types=output_types, output_shapes=output_shapes
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # 🛠️ 修复：添加repeat()避免数据耗尽
    dataset = dataset.repeat()

    return dataset

# -------------------------- 学习率调度器 --------------------------
def _create_trace_encoder(self):
    """共享 Trace 编码器：特征分支已适配 10 维输入"""
    # 波形分支
    wave_in = layers.Input(shape=(WAVEFORM_LENGTH, 1))
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(wave_in)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    wave_enc = keras.Model(wave_in, x)

    # 频谱图分支
    spec_in = layers.Input(shape=(SPEC_HEIGHT, SPEC_WIDTH, 1))
    y = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(spec_in)
    y = layers.MaxPooling2D((2, 2))(y)
    y = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(y)
    y = layers.GlobalAveragePooling2D()(y)
    spec_enc = keras.Model(spec_in, y)

    # 特征分支（10 维）
    feat_in = layers.Input(shape=(10,))
    z = layers.Dense(32, activation='relu')(feat_in)
    z = layers.Dense(32, activation='relu')(z)
    feat_enc = keras.Model(feat_in, z)

    return wave_enc, spec_enc, feat_enc


def create_trace_adaptive_scheduler(initial_lr):
    """为单trace模型创建自适应学习率调度器"""

    def lr_scheduler(epoch):
        if epoch < 15:
            return initial_lr
        elif epoch < 30:
            return initial_lr * 0.5
        else:
            return initial_lr * 0.2

    return lr_scheduler


# -------------------------- 回调函数 --------------------------
class ExplosionRecallLogger(Callback):
    def __init__(self, val_dataset, val_steps, is_trace_model=False):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_steps = val_steps
        self.is_trace_model = is_trace_model

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred_prob = []

        if self.is_trace_model:
            model_output = self.model.output
        else:
            model_output = self.model.output[0]  # 事件级输出是第一个

        try:
            if self.is_trace_model:
                for (x1, x2, x3), y_label in self.val_dataset.take(self.val_steps):
                    pred = self.model.predict([x1, x2, x3], verbose=0)
                    y_true.extend(y_label.numpy().tolist())
                    y_pred_prob.extend(pred.flatten().tolist())
            else:
                # 修复：现在只有两个输出
                for (x1, x2, x3), (y_event, y_trace) in self.val_dataset.take(self.val_steps):
                    pred = self.model.predict([x1, x2, x3], verbose=0)
                    pred = pred[0]  # 事件级输出是第一个
                    y_true.extend(y_event.numpy().tolist())
                    y_pred_prob.extend(pred.flatten().tolist())
        except ValueError as e:
            print(f"数据解包错误: {str(e)}")
            return

        if len(y_true) == 0:
            logs['val_explosion_recall'] = 0.0
            return

        y_pred = (np.array(y_pred_prob) > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            explosion_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            earthquake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            explosion_recall = 0.0
            earthquake_recall = 0.0

        if logs is not None:
            logs['val_explosion_recall'] = explosion_recall
            logs['val_earthquake_recall'] = earthquake_recall
            print(f" - Validation explosion recall: {explosion_recall:.4f}, earthquake recall: {earthquake_recall:.4f}")
# -------------------------- 修复负 loss 的回调 --------------------------
class FixTotalLossCallback(Callback):
    """
    把 logs['loss'] 修正为
        1.0 * event_output_loss + INTERMEDIATE_LOSS_WEIGHT * trace_classifier_loss
    覆盖 TensorFlow 日志里的负值 bug。
    """
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        ev_loss = logs.get('event_output_loss')
        tr_loss = logs.get('trace_classifier_loss')

        # 浮点容错 + 常量存在性检查
        if (ev_loss is not None and tr_loss is not None and
            np.isfinite(ev_loss) and np.isfinite(tr_loss)):
            logs['loss'] = ev_loss + INTERMEDIATE_LOSS_WEIGHT * tr_loss
            # 可选：打印确认（第一次 epoch 或每 10 次）
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f" - FixTotalLossCallback: 修正后 total loss = {logs['loss']:.4f}")
class MemoryCleaner(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        print_memory_usage("训练后 ")


class TracePerformanceLogger(Callback):
    """修复的Trace性能记录器"""

    def __init__(self, val_dataset, val_steps):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_steps = val_steps

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred_prob = []

        try:
            for (x1, x2, x3), (y_event, y_trace) in self.val_dataset.take(self.val_steps):
                preds = self.model.predict([x1, x2, x3], verbose=0)

                # 🛠️ 修复：正确解包输出
                if isinstance(preds, list) and len(preds) >= 2:
                    trace_pred = preds[1]  # trace级输出是第二个
                else:
                    trace_pred = preds  # 如果是单一输出

                batch_size = y_event.shape[0]
                for b in range(batch_size):
                    # 计算真实trace数量（排除填充的-1）
                    real_mask = y_trace[b] != -1
                    n_real = tf.reduce_sum(tf.cast(real_mask, tf.int32)).numpy()

                    if n_real == 0:
                        continue

                    # 提取真实trace的预测和标签
                    trace_pred_real = trace_pred[b, :n_real, 0] if len(trace_pred.shape) == 3 else trace_pred[
                        b, :n_real]
                    y_trace_real = y_trace[b, :n_real]

                    y_true.extend(y_trace_real.numpy().tolist())
                    y_pred_prob.extend(trace_pred_real.flatten().tolist())

            if len(y_true) == 0:
                print("警告: 没有有效的trace样本用于计算指标")
                logs['val_trace_accuracy'] = 0.0
                logs['val_trace_precision'] = 0.0
                logs['val_trace_recall'] = 0.0
                return

            y_pred = (np.array(y_pred_prob) > 0.5).astype(int)
            accuracy = np.mean(np.array(y_true) == y_pred)

            # 计算precision和recall（只在有正样本时）
            if np.sum(y_true) > 0 and np.sum(y_pred) > 0:
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                else:
                    precision = 0.0
                    recall = 0.0
            else:
                precision = 0.0
                recall = 0.0

            if logs is not None:
                logs['val_trace_accuracy'] = accuracy
                logs['val_trace_precision'] = precision
                logs['val_trace_recall'] = recall
                print(f" - Validation trace accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")

        except Exception as e:
            print(f"Trace性能记录器错误: {e}")
            if logs is not None:
                logs['val_trace_accuracy'] = 0.0
                logs['val_trace_precision'] = 0.0
                logs['val_trace_recall'] = 0.0


# -------------------------- 模型类 --------------------------
class EarthquakeClassifier:
    def __init__(self, column_mapping):
        self.model = None
        self.trace_model = None
        self.scaler = StandardScaler()
        self.column_mapping = column_mapping
        self.h5_manager = None
        self.event_class_weights = {0: 1.0, 1: 1.0}
        os.makedirs(os.path.dirname(RESULT_OUTPUT_PATH), exist_ok=True)
        self.trace_prob_cache = None

    def validate_shapes(self, dataset, steps=1):
        """验证输入输出形状"""
        print("验证数据集形状:")
        try:
            for i, (inputs, outputs) in enumerate(dataset.take(steps)):
                x1, x2, x3 = inputs
                y_event, y_trace = outputs

                print(f"批次 {i + 1}:")
                print(f"  输入形状: wave={x1.shape}, spec={x2.shape}, feat={x3.shape}")
                print(f"  输出形状: event_label={y_event.shape}, trace_label={y_trace.shape}")

                # 验证模型输出形状
                if self.model is not None:
                    try:
                        pred_event, pred_trace = self.model.predict([x1, x2, x3], verbose=0)
                        print(f"  模型输出形状: event_pred={pred_event.shape}, trace_pred={pred_trace.shape}")

                        # 验证损失计算
                        loss = self.model.test_on_batch([x1, x2, x3], [y_event, y_trace])
                        print(f"  测试损失: {loss}")
                    except Exception as e:
                        print(f"  模型预测失败: {e}")
                print("-" * 50)
        except Exception as e:
            print(f"形状验证失败: {e}")

    def build_trace_model(self):
        """高准确率版Trace模型 - 修复特征维度问题"""
        waveform_input = layers.Input(shape=(WAVEFORM_LENGTH, 1), name="waveform_input")
        x = layers.Conv1D(16, 5, activation="relu", padding="same")(waveform_input)
        x = layers.MaxPooling1D(2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(32, 5, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 5, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        waveform_branch = layers.Dense(32, activation="relu", name="wave_embed")(x)

        spectrogram_input = layers.Input(shape=(SPEC_HEIGHT, SPEC_WIDTH, 1), name="spectrogram_input")
        y = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(spectrogram_input)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(y)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2D(48, (3, 3), activation="relu", padding="same")(y)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.GlobalAveragePooling2D()(y)
        spectrogram_branch = layers.Dense(32, activation="relu", name="spec_embed")(y)

        #  修复：特征输入改为8维（预计算阶段只有8维特征）
        features_input = layers.Input(shape=(8,), name="features_input")  # 改为8维!
        z = layers.Dense(32, activation="relu")(features_input)
        z = layers.BatchNormalization()(z)
        features_branch = layers.Dense(32, activation="relu", name="feat_embed")(z)

        combined = layers.concatenate([waveform_branch, spectrogram_branch, features_branch])
        combined = layers.Dense(64, activation="relu")(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.5)(combined)
        output = layers.Dense(1, activation="sigmoid", name="trace_output")(combined)

        self.trace_model = keras.Model(
            inputs=[waveform_input, spectrogram_input, features_input],
            outputs=output
        )

        # 使用安全优化器和防弹损失函数
        optimizer = build_safe_optimizer(TRACE_LEARNING_RATE)
        self.trace_model.compile(
            optimizer=optimizer,
            loss=bulletproof_event_loss,
            metrics=["accuracy", "precision", "recall"]
        )
        return self.trace_model

    def precompute_trace_probs(self, metadata, h5_manager):
        """预计算trace概率 - 修复特征维度问题"""
        print("开始预计算 trace 概率...")
        probs = {}

        #  修复：逐条处理，确保特征维度正确
        total_traces = len(metadata)
        success_count = 0
        error_count = 0
        invalid_trace_count = 0

        with h5_manager as h5_file:
            comp_order_cache = None

            for idx, (_, row) in enumerate(metadata.iterrows()):
                if idx % 1000 == 0:
                    print(
                        f"预计算进度: {idx}/{total_traces} (成功: {success_count}, 失败: {error_count}, 无效: {invalid_trace_count})")

                trace_name = row["trace_name"]

                try:
                    if comp_order_cache is None:
                        comp_order_cache = get_component_order(h5_file)
                    target_comp_idx = comp_order_cache.index(VALID_COMPONENTS[0])
                    hdf5_path, event_idx = parse_trace_name(trace_name)

                    if hdf5_path not in h5_file:
                        error_count += 1
                        if error_count <= 10:
                            print(f"错误: HDF5路径不存在: {hdf5_path}, trace: {trace_name}")
                        continue

                    wave_group = h5_file[hdf5_path]
                    if event_idx < 0 or event_idx >= wave_group.shape[0]:
                        error_count += 1
                        if error_count <= 10:
                            print(
                                f"错误: 事件索引超出范围: {event_idx}, 最大索引: {wave_group.shape[0] - 1}, trace: {trace_name}")
                        continue

                    raw_wave = wave_group[event_idx, target_comp_idx, :].copy()

                    # 检查波形有效性
                    if not is_valid_trace(raw_wave):
                        invalid_trace_count += 1
                        continue

                    # 预处理波形
                    start_dt = parse_time_str(row["trace_start_time"])
                    raw_sr = float(row["trace_sampling_rate_hz"])
                    p_arrival = calculate_arrival_time(start_dt, int(row["trace_P_arrival_sample"]), raw_sr)
                    s_arrival = calculate_arrival_time(start_dt, int(row["trace_S_arrival_sample"]), raw_sr)
                    # 🛠️ 修复：预计算时不使用数据增强
                    processed_wave, log_pg_sg = preprocess_waveform(raw_wave, raw_sr, p_arrival, s_arrival, start_dt,
                                                                    is_training=False)

                    # 再次检查处理后的波形
                    if len(processed_wave) != WAVEFORM_LENGTH:
                        invalid_trace_count += 1
                        continue

                    spec_data = calculate_spectrogram(processed_wave)
                    processed_wave = np.expand_dims(processed_wave, axis=-1)
                    spec_data = np.expand_dims(spec_data, axis=-1)

                    # 计算特征
                    fractal_dim = calculate_fractal_dimension(processed_wave[:, 0])
                    epicentral_distance = haversine_distance(
                        row["source_latitude_deg"], row["source_longitude_deg"],
                        row["station_latitude_deg"], row["station_longitude_deg"])
                    time_features = extract_time_features(row["origin_time"])

                    # 🛠️ 修复：构建8维特征向量（预计算阶段）
                    feat_vector = np.array([
                        fractal_dim,
                        float(row["mag"]),
                        float(row["source_depth_km"]),
                        log_pg_sg,
                        epicentral_distance,
                        time_features[0],
                        time_features[1],
                        time_features[2]
                    ], dtype=np.float32)

                    # 归一化处理
                    if self.scaler is not None:
                        norm_part, dist, depth = split_feat_vector(feat_vector)
                        norm_part = self.scaler.transform(norm_part.reshape(1, -1)).flatten()
                        feat_vector = merge_feat_vector(norm_part, dist, depth)

                    #  关键：对每条有效trace进行预测
                    # 准备输入数据 - 使用8维特征
                    wave_input = np.expand_dims(processed_wave, axis=0)  # (1, 4500, 1)
                    spec_input = np.expand_dims(spec_data, axis=0)  # (1, 75, 115, 1)
                    feat_input = np.expand_dims(feat_vector, axis=0)  # (1, 8) - 8维特征

                    # 预测概率
                    pred_prob = self.trace_model.predict([wave_input, spec_input, feat_input], verbose=0)
                    pred_prob = float(np.clip(pred_prob[0, 0], 0, 1))

                    probs[trace_name] = pred_prob
                    success_count += 1

                    del processed_wave, spec_data, raw_wave

                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"预计算 trace {trace_name} 失败: {str(e)}")
                    continue

        # 详细的统计报告
        coverage = success_count / total_traces * 100
        print(f"\n=== 预计算完成统计 ===")
        print(f"总trace数: {total_traces}")
        print(f"成功预测: {success_count} ({coverage:.2f}%)")
        print(f"失败: {error_count}")
        print(f"无效波形: {invalid_trace_count}")
        print(f"缓存覆盖率: {len(probs)}/{total_traces} ({coverage:.2f}%)")

        if coverage < 98:
            print(f"⚠️  警告: 缓存覆盖率较低 ({coverage:.2f}%)，建议检查数据质量")

        return probs

    def build_event_model(self):
        """构建事件级模型：使用修复的注意力层"""

        # 输入维度调整：特征从9维改为10维
        event_wave_input = layers.Input(shape=(None, WAVEFORM_LENGTH, 1), name="event_waveform_input")
        event_spec_input = layers.Input(shape=(None, SPEC_HEIGHT, SPEC_WIDTH, 1), name="event_spectrogram_input")
        event_feat_input = layers.Input(shape=(None, 10), name="event_features_input")

        # Trace编码器
        wave_enc, spec_enc, feat_enc = self._create_trace_encoder()

        # 编码每个trace
        wave_emb = layers.TimeDistributed(wave_enc)(event_wave_input)
        spec_emb = layers.TimeDistributed(spec_enc)(event_spec_input)
        feat_emb = layers.TimeDistributed(feat_enc)(event_feat_input)

        # 合并特征
        combined = layers.Concatenate()([wave_emb, spec_emb, feat_emb])
        masked = layers.Masking(mask_value=0.0)(combined)

        # 🛠️ 修复：正确使用质量感知注意力层
        attention_weights = QualityAwareAttentionLayer(name="attention_weights")([masked, event_feat_input])

        # 加权特征聚合
        weighted = layers.Multiply()([masked, attention_weights])
        event_emb = layers.GlobalAveragePooling1D()(weighted)

        # 事件级分类输出
        event_out = layers.Dense(64, activation='relu')(event_emb)
        event_out = layers.Dropout(0.3)(event_out)
        event_out = layers.Dense(1, activation='sigmoid', name='event_output')(event_out)

        # Trace级辅助输出
        trace_out = layers.TimeDistributed(
            layers.Dense(1, activation='sigmoid'), name='trace_classifier')(masked)

        #  修复：创建模型时只包含需要损失函数的输出
        model = keras.Model(
            inputs=[event_wave_input, event_spec_input, event_feat_input],
            outputs=[event_out, trace_out]  # 只包含两个需要损失函数的输出
        )

        #  修复：编译配置只包含两个输出
        optimizer = build_safe_optimizer(LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss={
                'event_output': bulletproof_event_loss,
                'trace_classifier': bulletproof_trace_loss,
            },
            loss_weights={
                'event_output': 1.0,
                'trace_classifier': INTERMEDIATE_LOSS_WEIGHT,
            },
            metrics={
                'event_output': ['accuracy', 'precision', 'recall'],
                'trace_classifier': ['accuracy']
            }
        )

        #  新增：创建包含注意力权重的子模型用于可视化
        self.attention_model = keras.Model(
            inputs=[event_wave_input, event_spec_input, event_feat_input],
            outputs=attention_weights
        )

        self.model = model
        return model

    def _create_trace_encoder(self):
        """共享 Trace 编码器：特征分支已适配 10 维输入"""
        # 波形分支
        wave_in = layers.Input(shape=(WAVEFORM_LENGTH, 1))
        x = layers.Conv1D(32, 5, activation='relu', padding='same')(wave_in)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)
        wave_enc = keras.Model(wave_in, x)

        # 频谱图分支
        spec_in = layers.Input(shape=(SPEC_HEIGHT, SPEC_WIDTH, 1))
        y = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(spec_in)
        y = layers.MaxPooling2D((2, 2))(y)
        y = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(y)
        y = layers.GlobalAveragePooling2D()(y)
        spec_enc = keras.Model(spec_in, y)

        # 特征分支（10 维）
        feat_in = layers.Input(shape=(10,))
        z = layers.Dense(32, activation='relu')(feat_in)
        z = layers.Dense(32, activation='relu')(z)
        feat_enc = keras.Model(feat_in, z)

        return wave_enc, spec_enc, feat_enc

    def quality_aware_attention(self, features, feat_inputs):
        """质量感知注意力机制"""
        # 从10维特征中提取质量分数（第10维）
        batch_size = tf.shape(feat_inputs)[0]
        num_traces = tf.shape(feat_inputs)[1]
        quality_scores = feat_inputs[:, :, -1]  # 提取质量分数
        quality_scores = tf.reshape(quality_scores, (batch_size, num_traces, 1))

        # 基础内容注意力
        content_attention = layers.Dense(64, activation='relu')(features)
        content_attention = layers.Dense(1, activation='linear')(content_attention)

        # 用质量分数调整注意力
        # 正质量增强注意力，负质量抑制注意力
        quality_adjustment = tf.where(
            quality_scores >= 0,
            tf.exp(quality_scores * 2),  # 正质量：指数增强
            tf.sigmoid(quality_scores * 10)  # 负质量：sigmoid抑制到接近0
        )

        adjusted_attention = content_attention * quality_adjustment
        adjusted_attention = tf.squeeze(adjusted_attention, axis=-1)

        # softmax归一化
        attention_weights = tf.nn.softmax(adjusted_attention, axis=1)

        return tf.expand_dims(attention_weights, -1)

    def validate_model_output_shapes(self):
        """验证模型输出形状"""
        print("验证模型输出形状:")
        # 创建虚拟输入
        dummy_batch_size = 2
        dummy_wave = tf.random.normal((dummy_batch_size, MAX_TRACES_PER_EVENT, WAVEFORM_LENGTH, 1))
        dummy_spec = tf.random.normal((dummy_batch_size, MAX_TRACES_PER_EVENT, SPEC_HEIGHT, SPEC_WIDTH, 1))
        dummy_feat = tf.random.normal((dummy_batch_size, MAX_TRACES_PER_EVENT, 8))

        # 获取模型输出
        outputs = self.model([dummy_wave, dummy_spec, dummy_feat])

        print(f"事件级输出形状: {outputs[0].shape}")
        print(f"Trace级输出形状: {outputs[1].shape}")
        print(f"注意力权重形状: {outputs[2].shape}")  # 新增

        # 创建虚拟标签
        dummy_event_labels = tf.constant([1, 0], dtype=tf.int8)  # 形状 (2,)
        dummy_trace_labels = tf.constant([[1] * MAX_TRACES_PER_EVENT, [0] * MAX_TRACES_PER_EVENT],
                                         dtype=tf.int8)  # 形状 (2, 10)

        # 测试损失计算
        try:
            loss = self.model.test_on_batch(
                [dummy_wave, dummy_spec, dummy_feat],
                [dummy_event_labels, dummy_trace_labels]
            )
            print(f"测试损失计算成功: {loss}")
        except Exception as e:
            print(f"测试损失计算失败: {e}")

    def pretrain_trace_model(self, train_metadata, val_metadata):
        """修复版：预训练单trace分类模型"""
        print("构建单trace训练数据集...")
        train_dataset = build_trace_tf_dataset(
            metadata=train_metadata,
            h5_manager=self.h5_manager,
            scaler=self.scaler,
            shuffle=True,
            batch_size=TRACE_BATCH_SIZE,
            is_training=True  # 🛠️ 修复：训练集使用数据增强
        )

        print("构建单trace验证数据集...")
        val_dataset = build_trace_tf_dataset(
            metadata=val_metadata,
            h5_manager=self.h5_manager,
            scaler=self.scaler,
            shuffle=False,
            batch_size=TRACE_BATCH_SIZE,
            is_training=False  # 🛠️ 修复：验证集不使用数据增强
        )

        train_steps = max(1, len(train_metadata) // TRACE_BATCH_SIZE)
        val_steps = max(1, len(val_metadata) // TRACE_BATCH_SIZE)
        print(f"单trace训练步数: {train_steps}, 验证步数: {val_steps}")

        print("构建单trace模型...")
        self.build_trace_model()

        #  关键修复：验证模型构建成功
        if self.trace_model is None:
            raise RuntimeError("单Trace模型构建失败")

        self.trace_model.summary()

        y_train_labels = train_metadata["event_type"].apply(lambda x: 1 if x == "earthquake" else 0).values
        unique_classes = np.unique(y_train_labels)
        if len(unique_classes) < 2:
            print("警告: 训练集中只包含一种事件类型，这会影响单trace模型性能")
            unique_classes = np.array([0, 1])
        base_class_weights = compute_class_weight("balanced", classes=unique_classes, y=y_train_labels)

        class_weight_dict = {}
        for i, cls in enumerate(unique_classes):
            if cls == 0:
                class_weight_dict[0] = base_class_weights[i] * EXPLOSION_WEIGHT_SCALE
            else:
                class_weight_dict[1] = base_class_weights[i]

        if 0 not in class_weight_dict:
            class_weight_dict[0] = 1.0
        if 1 not in class_weight_dict:
            class_weight_dict[1] = 1.0

        print(f"单trace模型类别权重: {class_weight_dict}")

        explosion_recall_cb = ExplosionRecallLogger(val_dataset=val_dataset, val_steps=val_steps, is_trace_model=True)
        memory_cleaner_cb = MemoryCleaner()
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, "trace_model_epoch_{epoch:02d}.keras"),
            save_freq='epoch',
            save_weights_only=False,
            verbose=1
        )

        # 使用更灵活的学习率调度器
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )

        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1, monitor="val_loss"),
            lr_scheduler,
            explosion_recall_cb,
            memory_cleaner_cb,
            checkpoint_cb
        ]

        print("开始单trace模型预训练...")
        print_memory_usage("预训练前 ")

        with timing_context("单trace模型预训练"):
            history = self.trace_model.fit(
                x=train_dataset,
                epochs=TRACE_PRETRAIN_EPOCHS,
                steps_per_epoch=train_steps,
                validation_data=val_dataset,
                validation_steps=val_steps,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )

        print("单trace模型预训练完成")
        return history

    def fit_scaler(self, train_metadata):
        """仅对 6 维可归一化部分拟合 scaler"""
        feat_list = []
        with self.h5_manager as h5_file:
            comp_order_cache = None
            for idx, (_, row) in enumerate(train_metadata.iterrows()):
                if idx % 1000 == 0:
                    print(f"Scaler进度: {idx}/{len(train_metadata)}")
                try:
                    if comp_order_cache is None:
                        comp_order_cache = get_component_order(h5_file)
                    target_comp_idx = comp_order_cache.index(VALID_COMPONENTS[0])
                    hdf5_path, event_idx = parse_trace_name(row["trace_name"])
                    raw_wave = h5_file[hdf5_path][event_idx, target_comp_idx, :].copy()
                    start_dt = parse_time_str(row["trace_start_time"])
                    raw_sr = float(row["trace_sampling_rate_hz"])
                    p_arrival = calculate_arrival_time(start_dt, int(row["trace_P_arrival_sample"]), raw_sr)
                    s_arrival = calculate_arrival_time(start_dt, int(row["trace_S_arrival_sample"]), raw_sr)
                    processed_wave, log_pg_sg = preprocess_waveform(raw_wave, raw_sr, p_arrival, s_arrival, start_dt)
                    fractal_dim = calculate_fractal_dimension(processed_wave)
                    epicentral_distance = haversine_distance(
                        row["source_latitude_deg"], row["source_longitude_deg"],
                        row["station_latitude_deg"], row["station_longitude_deg"])
                    time_features = extract_time_features(row["origin_time"])

                    # ✅ 与 split_feat_vector 顺序完全一致
                    feat_vector = np.array([
                        fractal_dim,  # 0
                        float(row["mag"]),  # 1
                        float(row["source_depth_km"]),  # 2
                        log_pg_sg,  # 3
                        epicentral_distance,  # 4
                        time_features[0],  # 5
                        time_features[1],  # 6
                        time_features[2]  # 7
                    ], dtype=np.float32)

                    norm_part, _, _ = split_feat_vector(feat_vector)
                    feat_list.append(norm_part)
                    del processed_wave, raw_wave
                except Exception:
                    continue

        if len(feat_list) == 0:
            raise ValueError("没有有效特征用于scaler")
        self.scaler.fit(np.array(feat_list))

    def evaluate_test_set(self, test_name, test_metadata):
        """
        评估单个测试集：修复输出解包问题
        """
        try:
            # 获取事件分组统计信息
            test_event_groups = group_metadata_by_event(test_metadata)

            # 详细统计信息
            total_events = len(test_event_groups)
            earthquake_events = sum(1 for event_type, _ in test_event_groups if event_type == "earthquake")
            explosion_events = sum(1 for event_type, _ in test_event_groups if event_type == "explosion")

            # 单trace事件详细统计
            single_trace_events = sum(1 for _, traces in test_event_groups if len(traces) == 1)
            multi_trace_events = total_events - single_trace_events

            print(f"{test_name} 包含 {total_events} 个事件")

            # 预计算 trace 概率
            all_traces = pd.concat([tr for _, tr in test_event_groups], ignore_index=True)
            if self.trace_prob_cache is None:
                self.trace_prob_cache = self.precompute_trace_probs(all_traces, self.h5_manager)
            trace_prob_cache = self.trace_prob_cache

            test_dataset = build_event_tf_dataset(
                event_groups=test_event_groups,
                h5_path=WAVEFORM_PATH,
                scaler=self.scaler,
                shuffle=False,
                batch_size=BATCH_SIZE,
                trace_prob_cache=trace_prob_cache,
                is_training=False  # 🛠️ 修复：测试集不使用数据增强
            )

            test_steps = max(1, len(test_event_groups) // BATCH_SIZE)

            # ------- 收集预测 & 标签 -------
            y_true, y_pred_prob = [], []
            trace_pred_flat, trace_true_flat = [], []

            #  修复：现在只有两个输出
            for (x1, x2, x3), (y_event, y_trace) in test_dataset.take(test_steps):
                # 预测
                preds = self.model.predict([x1, x2, x3], verbose=0)
                event_pred = preds[0].flatten()
                trace_pred = preds[1]

                # 事件级
                batch_y_event = np.asarray(y_event).ravel().tolist()
                y_true.extend(batch_y_event)
                y_pred_prob.extend(event_pred.tolist())

                # trace 级：去掉 padding
                batch_size = y_event.shape[0]
                for b in range(batch_size):
                    n_real = tf.reduce_sum(tf.cast(y_trace[b] != -1, tf.int32)).numpy()
                    if n_real == 0:
                        continue
                    trace_pred_flat.extend(trace_pred[b, :n_real, 0].tolist())
                    trace_true_flat.extend(y_trace[b, :n_real].numpy().tolist())

            # ------- 计算详细指标 -------
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            # 事件级指标
            flat_y_true = [int(item) for item in y_true]
            y_pred = (np.array(y_pred_prob) > 0.5).astype(int)

            event_accuracy = accuracy_score(flat_y_true, y_pred)
            event_precision = precision_score(flat_y_true, y_pred, zero_division=0)
            event_recall = recall_score(flat_y_true, y_pred, zero_division=0)
            event_f1 = f1_score(flat_y_true, y_pred, zero_division=0)

            # 事件级混淆矩阵
            cm = confusion_matrix(flat_y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                explosion_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                earthquake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                explosion_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                earthquake_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            else:
                explosion_recall = earthquake_recall = explosion_precision = earthquake_precision = 0.0
                tn, fp, fn, tp = 0, 0, 0, 0

            # Trace级指标
            if len(trace_pred_flat) > 0:
                trace_pred_binary = (np.array(trace_pred_flat) > 0.5).astype(int)
                trace_accuracy = accuracy_score(trace_true_flat, trace_pred_binary)
                trace_precision = precision_score(trace_true_flat, trace_pred_binary, zero_division=0)
                trace_recall = recall_score(trace_true_flat, trace_pred_binary, zero_division=0)
                trace_f1 = f1_score(trace_true_flat, trace_pred_binary, zero_division=0)

                # Trace级混淆矩阵
                trace_cm = confusion_matrix(trace_true_flat, trace_pred_binary)
                if trace_cm.shape == (2, 2):
                    t_tn, t_fp, t_fn, t_tp = trace_cm.ravel()
                    trace_explosion_recall = t_tn / (t_tn + t_fp) if (t_tn + t_fp) > 0 else 0.0
                    trace_earthquake_recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
                else:
                    trace_explosion_recall = trace_earthquake_recall = 0.0
                    t_tn, t_fp, t_fn, t_tp = 0, 0, 0, 0
            else:
                trace_accuracy = trace_precision = trace_recall = trace_f1 = 0.0
                trace_explosion_recall = trace_earthquake_recall = 0.0
                t_tn, t_fp, t_fn, t_tp = 0, 0, 0, 0

            # ------- 输出详细结果 -------
            output_lines = [
                f"\n[{test_name} - 详细测试结果]",
                "=" * 80,
                "\n[数据集统计信息]",
                f"- 原始事件总数: {len(test_metadata['event_id'].unique())}",
                f"- 最终事件数 (包含单trace): {total_events}",
                f"- 事件利用率: {total_events / len(test_metadata['event_id'].unique()) * 100:.1f}%",
                f"- 地震事件: {earthquake_events}, 爆炸事件: {explosion_events}",
                f"- 单trace事件: {single_trace_events} ({single_trace_events / total_events * 100:.1f}%)",
                f"- 多trace事件: {multi_trace_events} ({multi_trace_events / total_events * 100:.1f}%)",
                "\n[事件级性能指标]",
                f"- 准确率 (Accuracy): {event_accuracy:.4f}",
                f"- 精确率 (Precision): {event_precision:.4f}",
                f"- 召回率 (Recall): {event_recall:.4f}",
                f"- F1分数: {event_f1:.4f}",
                f"- 地震召回率: {earthquake_recall:.4f}",
                f"- 爆炸召回率: {explosion_recall:.4f}",
                f"- 地震精确率: {earthquake_precision:.4f}",
                f"- 爆炸精确率: {explosion_precision:.4f}",
                f"- 混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
                "\n[Trace级性能指标]",
                f"- 准确率 (Accuracy): {trace_accuracy:.4f}",
                f"- 精确率 (Precision): {trace_precision:.4f}",
                f"- 召回率 (Recall): {trace_recall:.4f}",
                f"- F1分数: {trace_f1:.4f}",
                f"- 地震召回率: {trace_earthquake_recall:.4f}",
                f"- 爆炸召回率: {trace_explosion_recall:.4f}",
                f"- 总Trace数: {len(trace_pred_flat)}",
                f"- 混淆矩阵: TN={t_tn}, FP={t_fp}, FN={t_fn}, TP={t_tp}",
                "\n" + "=" * 80 + "\n"
            ]

            out_str = "\n".join(output_lines)
            print(out_str)
            with open(RESULT_OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(out_str)

            # ------- 画图 -------
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 事件级混淆矩阵
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Explosion', 'Earthquake'],
                        yticklabels=['Explosion', 'Earthquake'], ax=ax1)
            ax1.set_title('Event-Level Confusion Matrix')

            # Trace级混淆矩阵
            if len(trace_pred_flat) > 0:
                sns.heatmap(trace_cm, annot=True, fmt='d', cmap='Oranges',
                            xticklabels=['Explosion', 'Earthquake'],
                            yticklabels=['Explosion', 'Earthquake'], ax=ax2)
            ax2.set_title('Trace-Level Confusion Matrix')

            plt.tight_layout()
            both_cm_path = CONFUSION_MATRIX_PATH.replace('.png', f'_{test_name.replace(" ", "_")}_both.png')
            plt.savefig(both_cm_path, dpi=200, bbox_inches='tight')
            plt.close()

        except Exception as e:
            import traceback
            err_msg = f"加载 {test_name} 失败: {str(e)}\n" + traceback.format_exc()
            error_logger.error(err_msg)
            print(err_msg)
            with open(RESULT_OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(f"错误: {err_msg}\n\n" + "=" * 80 + "\n\n")

    # ------------------------------------------------------------------
    # 新增：修复负 loss 的回调（放在 train 方法外面也行，这里直接内嵌）
    # ------------------------------------------------------------------
    class FixTotalLossCallback(Callback):
        """
        把 logs['loss'] 修正为
            1.0 * event_output_loss + INTERMEDIATE_LOSS_WEIGHT * trace_classifier_loss
        覆盖 TensorFlow 日志里的负值 bug。
        """

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                return
            ev_loss = logs.get('event_output_loss')
            tr_loss = logs.get('trace_classifier_loss')
            if ev_loss is not None and tr_loss is not None:
                logs['loss'] = ev_loss + INTERMEDIATE_LOSS_WEIGHT * tr_loss

    def train(self, train_path, val_path, test_sets, waveform_path, skip_training=False):
        """修复版：确保模型正确构建和训练"""
        self.h5_manager = H5FileManager(waveform_path)
        trace_history = None
        event_history = None

        # ============== 内嵌修复版回调 ===============
        class FixedTracePerformanceLogger(Callback):
            def __init__(self, val_dataset, val_steps):
                super().__init__()
                self.val_dataset = val_dataset
                self.val_steps = val_steps

            def on_epoch_end(self, epoch, logs=None):
                y_true, y_pred_prob = [], []
                try:
                    for (x1, x2, x3), (y_event, y_trace) in self.val_dataset.take(self.val_steps):
                        preds = self.model.predict([x1, x2, x3], verbose=0)
                        trace_pred = preds[1] if isinstance(preds, list) and len(preds) >= 2 else preds

                        batch_size = y_event.shape[0]
                        for b in range(batch_size):
                            real_mask = y_trace[b] != -1
                            n_real = tf.reduce_sum(tf.cast(real_mask, tf.int32)).numpy()
                            if n_real == 0:
                                continue
                            pred_real = trace_pred[b, :n_real, 0] if len(trace_pred.shape) == 3 else trace_pred[
                                b, :n_real]
                            y_true.extend(y_trace[b, :n_real].numpy().tolist())
                            y_pred_prob.extend(pred_real.numpy().flatten().tolist())

                    if not y_true:
                        logs.update({'val_trace_accuracy': 0.0, 'val_trace_precision': 0.0, 'val_trace_recall': 0.0})
                        return

                    y_pred = (np.array(y_pred_prob) > 0.5).astype(int)
                    acc = np.mean(np.array(y_true) == y_pred)
                    cm = confusion_matrix(y_true, y_pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    else:
                        prec = rec = 0.0

                    logs.update({'val_trace_accuracy': acc, 'val_trace_precision': prec, 'val_trace_recall': rec})
                    print(f" - Validation trace accuracy: {acc:.4f}, precision: {prec:.4f}, recall: {rec:.4f}")

                except Exception as e:
                    print(f"Trace性能记录器错误: {e}")
                    logs.update({'val_trace_accuracy': 0.0, 'val_trace_precision': 0.0, 'val_trace_recall': 0.0})

        try:
            print("检查HDF5文件完整性...")
            check_h5_integrity(waveform_path)

            print("加载训练元数据...")
            train_metadata = load_metadata_from_split(train_path)
            val_metadata = load_metadata_from_split(val_path)
            train_event_groups = group_metadata_by_event(train_metadata)
            val_event_groups = group_metadata_by_event(val_metadata)

            if not skip_training:
                print("拟合特征标准化器...")
                self.fit_scaler(train_metadata)

                # 🛠️ 关键修复1：先预训练单Trace模型
                print("开始预训练单Trace模型...")
                trace_history = self.pretrain_trace_model(train_metadata, val_metadata)

                # 🛠️ 关键修复2：构建事件级模型
                print("构建事件级模型...")
                self.build_event_model()

                # -------------- 预计算 trace 概率缓存 --------------
                print("预计算训练集trace概率...")
                train_trace_prob_cache = self.precompute_trace_probs(train_metadata, self.h5_manager)
                print("预计算验证集trace概率...")
                val_trace_prob_cache = self.precompute_trace_probs(val_metadata, self.h5_manager)

                # -------------- 构建数据集（带 repeat） --------------
                print("构建训练数据集...")
                train_dataset = build_event_tf_dataset(
                    train_event_groups, waveform_path, self.scaler, shuffle=True,
                    batch_size=BATCH_SIZE, trace_prob_cache=train_trace_prob_cache, is_training=True
                ).repeat()  # 🛠️ 防耗尽

                print("构建验证数据集...")
                val_dataset = build_event_tf_dataset(
                    val_event_groups, waveform_path, self.scaler, shuffle=False,
                    batch_size=BATCH_SIZE, trace_prob_cache=val_trace_prob_cache, is_training=False
                )

                # -------------- 正确计算 steps（向上取整） --------------
                train_steps = int(np.ceil(len(train_event_groups) / BATCH_SIZE))
                val_steps = int(np.ceil(len(val_event_groups) / BATCH_SIZE))
                print(f"训练步数: {train_steps} | 验证步数: {val_steps}")

                #  关键修复3：验证模型是否构建成功
                if self.model is None:
                    raise RuntimeError("事件级模型构建失败，无法训练")

                print("事件级模型构建成功，开始训练...")
                self.model.summary()

                # -------------- 回调 --------------
                lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
                explosion_recall_cb = ExplosionRecallLogger(val_dataset=val_dataset, val_steps=val_steps)
                memory_cleaner_cb = MemoryCleaner()
                checkpoint_cb = keras.callbacks.ModelCheckpoint(
                    os.path.join(CHECKPOINT_DIR, "event_model_epoch_{epoch:02d}.keras"),
                    save_freq='epoch', save_weights_only=False, verbose=1
                )

                callbacks = [
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1,
                                                  monitor="val_loss"),
                    lr_scheduler,
                    explosion_recall_cb,
                    FixedTracePerformanceLogger(val_dataset=val_dataset, val_steps=val_steps),  # 🛠️ 修复版
                    memory_cleaner_cb,
                    checkpoint_cb,
                    self.FixTotalLossCallback(),
                ]

                # -------------- 训练 --------------
                print("开始事件级模型训练...")
                print_memory_usage("训练前 ")

                with timing_context("事件级模型训练"):
                    event_history = self.model.fit(
                        train_dataset,
                        epochs=EPOCHS,
                        steps_per_epoch=train_steps,
                        validation_data=val_dataset,
                        validation_steps=val_steps,
                        callbacks=callbacks,
                        verbose=1
                    )

                self.save_model(SAVE_MODEL_PATH)
                print("模型已保存至:", SAVE_MODEL_PATH)
            else:
                print("跳过训练，直接使用预训练模型")
                self.load_model(SAVE_MODEL_PATH)

            # -------------- 测试评估 --------------
            for test_name, test_path in test_sets:
                print(f"评估测试集: {test_name}")
                test_metadata = load_metadata_from_split(test_path)
                self.evaluate_test_set(test_name, test_metadata)

            # -------------- 统一缓存 --------------
            print("构建统一trace概率缓存...")
            all_traces = pd.concat(
                [pd.concat([tr for _, tr in train_event_groups], ignore_index=True),
                 pd.concat([tr for _, tr in val_event_groups], ignore_index=True)] +
                [pd.concat([tr for _, tr in group_metadata_by_event(load_metadata_from_split(tp))]) for _, tp in
                 test_sets],
                ignore_index=True
            )
            self.trace_prob_cache = self.precompute_trace_probs(all_traces, self.h5_manager)

            # -------------- 绘图 --------------
            if not skip_training and event_history:
                self.plot_optimized_training_history(trace_history, event_history)
            self.plot_trace_performance(val_event_groups)
            self.plot_quality_vs_attention(val_event_groups)
            self.plot_trace_model_quality(val_event_groups)

            return (trace_history, event_history)

        except Exception as e:
            error_logger.error("训练中断: %s", e, exc_info=True)
            raise e
        finally:
            self.h5_manager.close()
            print("训练流程结束（HDF5 已关闭）")

    def plot_quality_vs_attention(self, val_event_groups, test_event_groups=None, save_suffix='quality'):
        """
        完整替换版：修复输出解包问题 & 复用 trace_prob_cache
        """
        import matplotlib.pyplot as plt
        from statsmodels.nonparametric.smoothers_lowess import lowess

        # ---------- 1. 数据收集 ----------
        all_groups = val_event_groups + (test_event_groups or [])
        if not all_groups:
            print("警告：无事件数据可处理")
            return

        all_traces = pd.concat([tr for _, tr in all_groups], ignore_index=True)

        # ✅ 复用缓存，不再重复预计算
        if self.trace_prob_cache is None:
            raise RuntimeError("trace_prob_cache 未初始化，请先运行 train() 或手动设置 cache")
        trace_prob_cache = self.trace_prob_cache

        dataset = build_event_tf_dataset(
            all_groups, WAVEFORM_PATH, self.scaler, False, 32, trace_prob_cache
        )
        steps = min(100, max(1, len(all_groups) // 32))

        distances, qualities, attention_weights = [], [], []
        event_types, depths, correctness_labels = [], [], []

        #  修复：现在只有两个输出 (y_event, y_trace)
        for (x1, x2, x3), (y_event, y_trace) in dataset.take(steps):
            preds = self.model.predict([x1, x2, x3], verbose=0)

            #  修复：使用注意力子模型获取注意力权重
            attn_weights = self.attention_model.predict([x1, x2, x3], verbose=0)
            trace_pred = preds[1]

            batch_size = y_event.shape[0]
            for b in range(batch_size):
                feats = x3.numpy()[b]
                labels = y_trace.numpy()[b]
                ev_type = 'Earthquake' if y_event.numpy()[b] == 1 else 'Explosion'
                n_real = tf.reduce_sum(tf.cast(labels != -1, tf.int32)).numpy()
                if n_real == 0:
                    continue
                for i in range(n_real):
                    attn = attn_weights[b, i] if attn_weights.ndim == 2 else attn_weights[b, i, 0]
                    if attn <= 1e-8:
                        continue
                    pred_lab = 1 if trace_pred[b, i, 0] > 0.5 else 0
                    true_lab = int(labels[i])
                    correct = (pred_lab == true_lab)
                    prob = np.clip(trace_pred[b, i, 0], 1e-5, 1 - 1e-5)
                    logit = np.log(prob / (1 - prob))
                    confidence = abs(np.clip(logit, -10, 10))
                    quality = confidence if correct else -confidence
                    distances.append(feats[i, 4])
                    depths.append(feats[i, 2])
                    qualities.append(quality)
                    attention_weights.append(attn)
                    event_types.append(ev_type)
                    correctness_labels.append(correct)

        qualities = np.array(qualities)
        correctness_labels = np.array(correctness_labels)
        distances = np.array(distances)
        depths = np.array(depths)
        event_types = np.array(event_types)

        n_correct = int(correctness_labels.sum())
        n_total = len(correctness_labels)
        n_error = n_total - n_correct  # ✅ 简单减法
        acc = n_correct / n_total

        print(f'\n===== Trace 模型四子图 Quality 诊断正确率统计 =====')
        print(f'总样本数: {n_total}')
        print(f'正确样本: {n_correct} | 错误样本: {n_error}')
        print(f'整体正确率: {acc:.4f}')
        print(f'Quality 为负的比例: {(qualities < 0).mean():.2%}')
        print('=============================================\n')

        # ---------- 3. 绘图 ----------
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Event Model: Signed Quality Diagnostics', fontsize=16)

        # 图1 距离 vs signed quality
        colors1 = np.where(correctness_labels, 'blue', 'red')
        ax1.scatter(distances, qualities, c=colors1, s=20, alpha=0.7, edgecolors='k', linewidths=0.5)
        if distances.size > 10:
            order = np.argsort(distances)
            trend = lowess(qualities[order], distances[order], frac=0.3, return_sorted=False)
            ax1.plot(distances[order], trend, color='black', lw=2, label='LOWESS (all)')
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Epicentral Distance (km)')
        ax1.set_ylabel('Signed Quality (wrong<0, right>0)')
        ax1.set_title('Distance vs Signed Quality')
        ax1.legend();
        ax1.grid(True, alpha=0.3)

        # 图2 直方图（带负轴）
        max_abs = max(abs(qualities)) if qualities.size else 1
        bins = np.linspace(-max_abs, max_abs, 41)
        ax2.hist(qualities, bins=bins, color='steelblue', alpha=0.7, edgecolor='k')
        ax2.axvline(0, color='k', linestyle='--')
        ax2.set_xlabel('Signed Quality')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Signed Quality Distribution')
        ax2.grid(True, alpha=0.3)

        # ---------- 3. 第3子图：震中距 vs Signed Quality（完整图例） ----------
        # 散点：地震=青色，爆破=红色
        eq_mask = event_types == 'Earthquake'
        ex_mask = ~eq_mask
        ax3.scatter(distances[eq_mask], qualities[eq_mask],
                    c='cyan', s=20, alpha=0.7, edgecolors='k', linewidths=0.5,
                    label='Earthquake')
        ax3.scatter(distances[ex_mask], qualities[ex_mask],
                    c='red', s=20, alpha=0.7, edgecolors='k', linewidths=0.5,
                    label='Explosion')

        # 趋势线：地震=深绿，爆破=深橙
        for et, col_trend, msk in [('Earthquake', 'darkgreen', eq_mask),
                                   ('Explosion', 'darkorange', ex_mask)]:
            if msk.sum() < 10:
                continue
            x_m = np.array(distances)[msk]
            y_m = np.array(qualities)[msk]
            order = np.argsort(x_m)
            trend = lowess(y_m[order], x_m[order], frac=0.3, return_sorted=False)
            ax3.plot(x_m[order], trend, color=col_trend, lw=2.5,
                     label=f'{et} trend')

        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Epicentral Distance (km)')
        ax3.set_ylabel('Signed Quality')
        ax3.set_title('Distance vs Signed Quality')
        ax3.legend()  # ← 现在四项全部出现
        ax3.grid(True, alpha=0.3)

        # 图4 深度 vs Signed Quality
        ax4.scatter(depths, qualities, c=colors1, s=20, alpha=0.7, edgecolors='k', linewidths=0.5)
        if depths.size > 10:
            order = np.argsort(depths)
            trend = lowess(qualities[order], depths[order], frac=0.3, return_sorted=False)
            ax4.plot(depths[order], trend, color='black', lw=2, label='LOWESS (all)')
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Source Depth (km)')
        ax4.set_ylabel('Signed Quality (wrong<0, right>0)')
        ax4.set_title('Depth vs Signed Quality')
        ax4.legend();
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = TRACE_ATTENTION_HEATMAP_PATH.replace(
            '.png', f'_{save_suffix}_signed_quality_complete.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'已保存完整替换版 quality 图：{save_path}')

    def plot_trace_model_quality(self, val_event_groups, test_event_groups=None, save_suffix='trace_model_quality'):
        """
        绘制单TRACE模型质量诊断图
        纵坐标：单TRACE模型对TRACE的判别质量 (2*y_true-1)*(2*y_pred-1)
        """
        import matplotlib.pyplot as plt
        from statsmodels.nonparametric.smoothers_lowess import lowess

        # ---------- 1. 数据收集 ----------
        all_groups = val_event_groups + (test_event_groups or [])
        if not all_groups:
            print("警告：无事件数据可处理")
            return

        all_traces = pd.concat([tr for _, tr in all_groups], ignore_index=True)

        # ✅ 复用缓存，不再重复预计算
        if self.trace_prob_cache is None:
            raise RuntimeError("trace_prob_cache 未初始化，请先运行 train() 或手动设置 cache")
        trace_prob_cache = self.trace_prob_cache

        dataset = build_event_tf_dataset(
            all_groups, WAVEFORM_PATH, self.scaler, False, 32, trace_prob_cache
        )
        steps = min(100, max(1, len(all_groups) // 32))

        distances, trace_qualities, attention_weights = [], [], []
        event_types, depths, correctness_labels = [], [], []

        #  修复：现在只有两个输出 (y_event, y_trace)
        for (x1, x2, x3), (y_event, y_trace) in dataset.take(steps):
            # 使用单TRACE模型的预测概率计算质量
            batch_size = y_event.shape[0]
            for b in range(batch_size):
                feats = x3.numpy()[b]
                labels = y_trace.numpy()[b]
                ev_type = 'Earthquake' if y_event.numpy()[b] == 1 else 'Explosion'
                n_real = tf.reduce_sum(tf.cast(labels != -1, tf.int32)).numpy()
                if n_real == 0:
                    continue

                for i in range(n_real):
                    # 从特征中提取单TRACE模型的预测概率（第9维，索引8）
                    trace_prob = feats[i, 8]  # 第9维是trace概率
                    true_label = int(labels[i])

                    # 🛠️ 使用与单TRACE模型相同的质量函数
                    trace_quality = float((2 * true_label - 1) * (2 * trace_prob - 1))

                    # 计算正确性
                    pred_label = 1 if trace_prob > 0.5 else 0
                    correct = (pred_label == true_label)

                    distances.append(feats[i, 4])  # 震中距
                    depths.append(feats[i, 2])  # 深度
                    trace_qualities.append(trace_quality)
                    event_types.append(ev_type)
                    correctness_labels.append(correct)

        trace_qualities = np.array(trace_qualities)
        correctness_labels = np.array(correctness_labels)
        distances = np.array(distances)
        depths = np.array(depths)
        event_types = np.array(event_types)

        n_correct = int(correctness_labels.sum())
        n_total = len(correctness_labels)
        n_error = n_total - n_correct
        acc = n_correct / n_total

        print(f'\n===== 单TRACE模型质量诊断统计 =====')
        print(f'总样本数: {n_total}')
        print(f'正确样本: {n_correct} | 错误样本: {n_error}')
        print(f'整体正确率: {acc:.4f}')
        print(f'质量为正的比例: {(trace_qualities > 0).mean():.2%}')
        print(f'质量平均值: {trace_qualities.mean():.4f}')
        print(f'质量标准差: {trace_qualities.std():.4f}')
        print('==================================\n')

        # ---------- 2. 绘图 ----------
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Trace Model Quality Diagnostics (Quality = (2*y_true-1)*(2*y_pred-1))', fontsize=16)

        # 图1 距离 vs 单TRACE模型质量
        colors1 = np.where(correctness_labels, 'blue', 'red')
        ax1.scatter(distances, trace_qualities, c=colors1, s=20, alpha=0.7, edgecolors='k', linewidths=0.5)
        if distances.size > 10:
            order = np.argsort(distances)
            trend = lowess(trace_qualities[order], distances[order], frac=0.3, return_sorted=False)
            ax1.plot(distances[order], trend, color='black', lw=2, label='LOWESS (all)')
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Epicentral Distance (km)')
        ax1.set_ylabel('Trace Model Quality')
        ax1.set_title('Distance vs Trace Model Quality')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2 直方图（带负轴）
        max_abs = max(abs(trace_qualities)) if trace_qualities.size else 1
        bins = np.linspace(-max_abs, max_abs, 41)
        ax2.hist(trace_qualities, bins=bins, color='steelblue', alpha=0.7, edgecolor='k')
        ax2.axvline(0, color='k', linestyle='--')
        ax2.set_xlabel('Trace Model Quality')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Trace Model Quality Distribution')
        ax2.grid(True, alpha=0.3)

        # 图3 距离 vs 单TRACE模型质量（按事件类型分类）
        eq_mask = event_types == 'Earthquake'
        ex_mask = ~eq_mask
        ax3.scatter(distances[eq_mask], trace_qualities[eq_mask],
                    c='cyan', s=20, alpha=0.7, edgecolors='k', linewidths=0.5,
                    label='Earthquake')
        ax3.scatter(distances[ex_mask], trace_qualities[ex_mask],
                    c='red', s=20, alpha=0.7, edgecolors='k', linewidths=0.5,
                    label='Explosion')

        # 趋势线：地震=深绿，爆破=深橙
        for et, col_trend, msk in [('Earthquake', 'darkgreen', eq_mask),
                                   ('Explosion', 'darkorange', ex_mask)]:
            if msk.sum() < 10:
                continue
            x_m = np.array(distances)[msk]
            y_m = np.array(trace_qualities)[msk]
            order = np.argsort(x_m)
            trend = lowess(y_m[order], x_m[order], frac=0.3, return_sorted=False)
            ax3.plot(x_m[order], trend, color=col_trend, lw=2.5,
                     label=f'{et} trend')

        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Epicentral Distance (km)')
        ax3.set_ylabel('Trace Model Quality')
        ax3.set_title('Distance vs Trace Model Quality (by Event Type)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 图4 深度 vs 单TRACE模型质量
        ax4.scatter(depths, trace_qualities, c=colors1, s=20, alpha=0.7, edgecolors='k', linewidths=0.5)
        if depths.size > 10:
            order = np.argsort(depths)
            trend = lowess(trace_qualities[order], depths[order], frac=0.3, return_sorted=False)
            ax4.plot(depths[order], trend, color='black', lw=2, label='LOWESS (all)')
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Source Depth (km)')
        ax4.set_ylabel('Trace Model Quality')
        ax4.set_title('Depth vs Trace Model Quality')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = TRACE_ATTENTION_HEATMAP_PATH.replace(
            '.png', f'_{save_suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'已保存单TRACE模型质量诊断图：{save_path}')

        # ---------- 3. 额外分析：质量与正确率的关系 ----------
        plt.figure(figsize=(12, 8))

        # 按质量分箱计算正确率
        quality_bins = np.linspace(-1, 1, 21)
        bin_centers = (quality_bins[:-1] + quality_bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []

        for i in range(len(quality_bins) - 1):
            mask = (trace_qualities >= quality_bins[i]) & (trace_qualities < quality_bins[i + 1])
            if mask.sum() > 0:
                bin_acc = correctness_labels[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)

        # 绘制质量-正确率关系图
        plt.subplot(2, 2, 1)
        plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Trace Model Quality')
        plt.ylabel('Accuracy')
        plt.title('Quality vs Accuracy')
        plt.grid(True, alpha=0.3)

        # 绘制样本数量分布
        plt.subplot(2, 2, 2)
        plt.bar(bin_centers, bin_counts, width=0.08, alpha=0.7)
        plt.xlabel('Trace Model Quality')
        plt.ylabel('Sample Count')
        plt.title('Quality Distribution')
        plt.grid(True, alpha=0.3)

        # 绘制质量与距离的关系（按正确性）
        plt.subplot(2, 2, 3)
        correct_mask = correctness_labels
        plt.scatter(distances[correct_mask], trace_qualities[correct_mask],
                    c='green', s=15, alpha=0.6, label='Correct')
        plt.scatter(distances[~correct_mask], trace_qualities[~correct_mask],
                    c='red', s=15, alpha=0.6, label='Incorrect')
        plt.xlabel('Epicentral Distance (km)')
        plt.ylabel('Trace Model Quality')
        plt.title('Distance vs Quality (by Correctness)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 绘制质量与深度的关系（按正确性）
        plt.subplot(2, 2, 4)
        plt.scatter(depths[correct_mask], trace_qualities[correct_mask],
                    c='green', s=15, alpha=0.6, label='Correct')
        plt.scatter(depths[~correct_mask], trace_qualities[~correct_mask],
                    c='red', s=15, alpha=0.6, label='Incorrect')
        plt.xlabel('Source Depth (km)')
        plt.ylabel('Trace Model Quality')
        plt.title('Depth vs Quality (by Correctness)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        analysis_path = TRACE_ATTENTION_HEATMAP_PATH.replace(
            '.png', f'_{save_suffix}_analysis.png')
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'已保存单TRACE模型质量分析图：{analysis_path}')

    def plot_trace_training_history(self, trace_history):
        """绘制单Trace模型训练历史"""
        if trace_history is None:
            print("没有单Trace训练历史数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 定义要绘制的指标
        metrics = [
            ('accuracy', 'Accuracy'),
            ('loss', 'Loss'),
            ('precision', 'Precision'),
            ('recall', 'Recall')
        ]

        for i, (metric_key, metric_name) in enumerate(metrics):
            ax = axes[i]

            # 训练集指标
            train_metric = trace_history.history.get(metric_key)
            if train_metric is not None:
                ax.plot(train_metric, label=f'Training {metric_name}',
                        linewidth=2, color='#1f77b4')

            # 验证集指标
            val_metric_key = f'val_{metric_key}'
            val_metric = trace_history.history.get(val_metric_key)
            if val_metric is not None:
                ax.plot(val_metric, label=f'Validation {metric_name}',
                        linewidth=2, color='#ff7f0e', linestyle='--')

            ax.set_title(f'Trace Model - {metric_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 如果是准确率、精确率、召回率，设置y轴范围为[0,1]
            if metric_key in ['accuracy', 'precision', 'recall']:
                ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(TRACE_HISTORY_PLOT_PATH, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"单Trace模型训练历史图已保存至: {TRACE_HISTORY_PLOT_PATH}")

    def plot_event_training_history(self, event_history):
        """event 训练历史图：只保留和 trace 完全一致的 4 个指标"""
        if event_history is None:
            print("没有事件级训练历史数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        keys = ['accuracy', 'loss', 'precision', 'recall']
        titles = ['Accuracy', 'Loss', 'Precision', 'Recall']

        for ax, k, t in zip(axes, keys, titles):
            tr_key = f'event_output_{k}'
            val_tr_key = f'val_{tr_key}'

            # 训练集
            if tr_key in event_history.history:
                ax.plot(event_history.history[tr_key],
                        label=f'Training {t}', linewidth=2, color='#1f77b4')

            # 验证集
            if val_tr_key in event_history.history:
                ax.plot(event_history.history[val_tr_key],
                        label=f'Validation {t}', linewidth=2, color='#ff7f0e', linestyle='--')

            ax.set_title(f'Event Model - {t}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(t)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(EVENT_HISTORY_PLOT_PATH, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"事件级模型训练历史图已保存至: {EVENT_HISTORY_PLOT_PATH}")

    def plot_optimized_training_history(self, trace_history, event_history):
        """优化的训练历史图 - 分为两个独立的图"""
        # 绘制单Trace模型训练历史
        self.plot_trace_training_history(trace_history)

        # 绘制事件级模型训练历史
        self.plot_event_training_history(event_history)

    def plot_trace_performance(self, val_event_groups):
        """修复版：确保数据能够正确收集"""
        if len(val_event_groups) == 0:
            print("警告: 验证事件组为空")
            return

        # 预计算 trace 概率
        if self.trace_prob_cache is None:
            print("警告: trace_prob_cache 为空，无法构建数据集")
            return

        trace_prob_cache = self.trace_prob_cache

        # 构建验证数据集
        try:
            print(f"构建验证数据集，事件数: {len(val_event_groups)}")
            val_dataset = build_event_tf_dataset(
                val_event_groups,
                WAVEFORM_PATH,
                self.scaler,
                shuffle=False,
                batch_size=BATCH_SIZE,
                trace_prob_cache=trace_prob_cache
            )

            # 计算合适的验证步数
            val_steps = min(50, max(1, len(val_event_groups) // BATCH_SIZE))
            print(f"验证数据集: {len(val_event_groups)} 事件, {val_steps} 步")

            if val_steps == 0:
                print("错误: 验证步数为0，无法进行评估")
                return

        except Exception as e:
            print(f"构建验证数据集失败: {e}")
            import traceback
            traceback.print_exc()
            return

        event_true, event_pred_prob = [], []
        trace_pred_flat, trace_true_flat = [], []

        try:
            # 🛠️ 关键修复：直接迭代数据集，不使用 take()
            print("开始数据收集...")
            batch_count = 0
            total_samples = 0

            for batch in val_dataset:
                if batch_count >= val_steps:  # 手动控制批次数量
                    break

                inputs, outputs = batch
                x1, x2, x3 = inputs
                y_event, y_trace = outputs

                #  修复：确保输入数据有效
                if (x1.shape[0] == 0 or x2.shape[0] == 0 or x3.shape[0] == 0):
                    print(f"批次 {batch_count}: 输入数据为空，跳过")
                    continue

                #  修复：模型现在返回两个输出
                try:
                    preds = self.model.predict([x1, x2, x3], verbose=0, steps=1)
                except Exception as e:
                    print(f"批次 {batch_count}: 预测失败: {e}")
                    continue

                # 事件级预测是第一个输出
                if isinstance(preds, list) and len(preds) >= 1:
                    event_pred = preds[0]
                else:
                    event_pred = preds

                event_pred_prob.extend(event_pred.flatten().tolist())
                event_true.extend(y_event.numpy().tolist())

                # trace级预测是第二个输出
                trace_pred = None
                if isinstance(preds, list) and len(preds) >= 2:
                    trace_pred = preds[1]
                elif hasattr(preds, 'shape') and len(preds.shape) > 2:
                    trace_pred = preds

                batch_size = y_event.shape[0]
                for b in range(batch_size):
                    # 计算真实trace数量（排除填充的-1）
                    try:
                        real_mask = y_trace[b] != -1
                        n_real = tf.reduce_sum(tf.cast(real_mask, tf.int32)).numpy()

                        if n_real == 0:
                            continue

                        #  修复：正确处理trace预测形状
                        if trace_pred is not None:
                            # trace_pred 形状应该是 (batch_size, max_traces, 1)
                            if len(trace_pred.shape) == 3:
                                batch_trace_pred = trace_pred[b, :n_real, 0]  # 取前n_real个，去掉最后一个维度
                            elif len(trace_pred.shape) == 2:
                                batch_trace_pred = trace_pred[b, :n_real]  # 形状可能是 (batch_size, max_traces)
                            else:
                                print(f"批次 {batch_count}: 未知的trace预测形状: {trace_pred.shape}")
                                continue

                            trace_pred_flat.extend(batch_trace_pred.flatten().tolist())

                        # 真实标签
                        trace_true_flat.extend(y_trace[b, :n_real].numpy().tolist())
                        total_samples += n_real

                    except Exception as e:
                        print(f"批次 {batch_count} 样本 {b} 处理失败: {e}")
                        continue

                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"处理了 {batch_count} 批次, 累计 {total_samples} 个trace样本")

            print(f"数据收集完成: {len(event_true)} 事件, {len(trace_true_flat)} traces, {batch_count} 批次")

        except Exception as e:
            print(f"数据收集过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return

        # 检查是否有数据
        if len(event_true) == 0:
            print("错误: 没有收集到事件级数据")
            print("调试信息:")
            print(f"- 批次处理数: {batch_count}")
            print(f"- 验证步数: {val_steps}")
            print(f"- 事件组数量: {len(val_event_groups)}")
            return

        if len(trace_true_flat) == 0:
            print("警告: 没有收集到trace级数据，只绘制事件级指标")

        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        # 事件级指标计算
        event_pred = [int(p > 0.5) for p in event_pred_prob]
        acc_e = accuracy_score(event_true, event_pred)
        pre_e = precision_score(event_true, event_pred, zero_division=0)
        rec_e = recall_score(event_true, event_pred, zero_division=0)
        f1_e = f1_score(event_true, event_pred, zero_division=0)

        print(f"事件级指标 - 准确率: {acc_e:.4f}, 精确率: {pre_e:.4f}, 召回率: {rec_e:.4f}, F1: {f1_e:.4f}")

        # Trace级指标计算
        if len(trace_true_flat) > 0:
            trace_pred_binary = (np.array(trace_pred_flat) > 0.5).astype(int)
            acc_t = accuracy_score(trace_true_flat, trace_pred_binary)
            pre_t = precision_score(trace_true_flat, trace_pred_binary, zero_division=0)
            rec_t = recall_score(trace_true_flat, trace_pred_binary, zero_division=0)
            f1_t = f1_score(trace_true_flat, trace_pred_binary, zero_division=0)

            print(f"Trace级指标 - 准确率: {acc_t:.4f}, 精确率: {pre_t:.4f}, 召回率: {rec_t:.4f}, F1: {f1_t:.4f}")
        else:
            acc_t = pre_t = rec_t = f1_t = 0.0
            print("Trace级指标: 无数据")

        # 绘制指标对比图
        metrics_name = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        event_vals = [acc_e, pre_e, rec_e, f1_e]
        trace_vals = [acc_t, pre_t, rec_t, f1_t]

        # 创建图形
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        colors = ['#1f77b4', '#ff7f0e']

        for i, (ax, m_name, e_val, t_val) in enumerate(zip(axes, metrics_name, event_vals, trace_vals)):
            labels = ['Event-Level', 'Trace-Level']
            values = [e_val, t_val]

            bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='k')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('Score')
            ax.set_title(f'{m_name}')

            # 在柱状图上显示数值
            for bar, v in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(labels, rotation=45)

        plt.suptitle('Model Performance: Event-Level vs Trace-Level', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存合并的指标图
        merged_path = TRACE_PERFORMANCE_PATH.replace('.png', '_merged.png')
        plt.savefig(merged_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f'指标对比图已保存：{merged_path}')

        # 绘制概率分布图
        plt.figure(figsize=(10, 6))

        if len(event_pred_prob) > 0:
            event_probs = np.array(event_pred_prob)
            plt.hist(event_probs, bins=30, alpha=0.6, label='Event-Level',
                     color='#1f77b4', density=True)
            print(f"事件级概率分布: 均值={event_probs.mean():.4f}, 标准差={event_probs.std():.4f}")

        if len(trace_pred_flat) > 0:
            trace_probs = np.array(trace_pred_flat)
            plt.hist(trace_probs, bins=30, alpha=0.6, label='Trace-Level',
                     color='#ff7f0e', density=True)
            print(f"Trace级概率分布: 均值={trace_probs.mean():.4f}, 标准差={trace_probs.std():.4f}")

        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        dist_path = TRACE_PERFORMANCE_PATH.replace('.png', '_prob_dist.png')
        plt.savefig(dist_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f'概率分布图已保存：{dist_path}')

    def _build_attention_model(self):
        if self.model is None:
            raise RuntimeError("必须先加载 self.model 才能构建 attention_model")
        # 提取输入
        inputs = self.model.input  # [wave, spec, feat]
        # 提取注意力层输出
        attention_layer = self.model.get_layer("attention_weights")
        attention_output = attention_layer.output
        # 构建子模型
        self.attention_model = keras.Model(inputs=inputs, outputs=attention_output)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if not path.endswith('.keras'):
            path += '.keras'

        self.model.save(path)
        trace_model_path = path.replace(".keras", "_trace_model.keras")
        if hasattr(self, 'trace_model') and self.trace_model is not None:
            self.trace_model.save(trace_model_path)

        scaler_path = path.replace(".keras", "_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        # ✅ 保存缓存
        if self.trace_prob_cache is not None:
            cache_path = path.replace(".keras", "_trace_probs.joblib")
            joblib.dump(self.trace_prob_cache, cache_path)
            print(f"trace 概率缓存已保存到: {cache_path}")

    def load_model(self, path):
        import keras
        self.model = keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
        self._build_attention_model()

        trace_model_path = path.replace(".keras", "_trace_model.keras")
        if os.path.exists(trace_model_path):
            self.trace_model = keras.models.load_model(trace_model_path, custom_objects=CUSTOM_OBJECTS)

        scaler_path = path.replace(".keras", "_scaler.joblib")
        self.scaler = joblib.load(scaler_path)

        # ✅ 加载缓存
        cache_path = path.replace(".keras", "_trace_probs.joblib")
        if os.path.exists(cache_path):
            self.trace_prob_cache = joblib.load(cache_path)
            print(f"trace 概率缓存已加载: {cache_path}")
        else:
            print("⚠️ 未找到 trace 概率缓存，画图时将重新计算（建议先 train()）")

    def load_latest_checkpoint(self):
        """加载最新的检查点"""
        if not os.path.exists(CHECKPOINT_DIR):
            return None

        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('event_model_') and f.endswith('.keras')]
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)), reverse=True)
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[0])
        print(f"加载最新事件级模型检查点: {latest_checkpoint}")

        self.model = keras.models.load_model(
            latest_checkpoint,
            custom_objects=CUSTOM_OBJECTS
        )

        trace_checkpoint = latest_checkpoint.replace('event_model_', 'trace_model_')
        if os.path.exists(trace_checkpoint):
            self.trace_model = keras.models.load_model(trace_checkpoint)

        scaler_path = latest_checkpoint.replace(".keras", "_scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        return latest_checkpoint


# -------------------------- 主函数 --------------------------
def main():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"使用GPU: {[gpu.name for gpu in gpus]}")
        else:
            print("使用CPU")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")

    model_exists = os.path.exists(SAVE_MODEL_PATH)
    trace_model_path = SAVE_MODEL_PATH.replace(".keras", "_trace_model.keras")
    trace_model_exists = os.path.exists(trace_model_path)
    scaler_path = SAVE_MODEL_PATH.replace(".keras", "_scaler.joblib")
    scaler_exists = os.path.exists(scaler_path)

    skip_training = model_exists and trace_model_exists and scaler_exists

    if skip_training:
        print(f"已找到完整的预训练模型: {SAVE_MODEL_PATH}")
        print("将跳过训练，直接进行测试和可视化")
    else:
        print("未找到完整的预训练模型")
        print("将开始训练过程")

    classifier = EarthquakeClassifier(COLUMN_MAPPING)

    try:
        print("=" * 60)
        if skip_training:
            print("改进的地震与爆炸分类 - 使用预训练模型")
        else:
            print("改进的地震与爆炸分类模型训练开始")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        if skip_training:
            print(f"从 {SAVE_MODEL_PATH} 加载预训练模型")
            try:
                classifier.load_model(SAVE_MODEL_PATH)
                classifier.h5_manager = H5FileManager(WAVEFORM_PATH)
                print("模型加载成功!")
            except Exception as e:
                print(f"模型加载失败: {e}")
                print("将重新训练模型...")
                skip_training = False

        if not skip_training:
            classifier.h5_manager = H5FileManager(WAVEFORM_PATH)

        # ========== 训练/测试 ==========
        print("开始训练流程...")
        hist = classifier.train(
            train_path=TRAIN_PATH,
            val_path=VAL_PATH,
            test_sets=TEST_SETS,
            waveform_path=WAVEFORM_PATH,
            skip_training=skip_training
        )

        #  修复：检查模型是否成功构建
        if classifier.model is None:
            print("错误: 模型训练失败，模型为None!")
            return

        # ========== 验证保存-加载 ==========
        print("保存模型...")
        classifier.save_model(SAVE_MODEL_PATH)

        # 验证模型可以重新加载
        print("验证模型重新加载...")
        try:
            loaded_model = keras.models.load_model(SAVE_MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
            print("✅ 模型已可正常重新加载！")
        except Exception as e:
            print(f"❌ 模型重新加载失败: {e}")

        # ========== 准备事件组 ==========
        print("准备验证和测试事件组...")
        val_metadata = load_metadata_from_split(VAL_PATH)
        val_event_groups = group_metadata_by_event(val_metadata)

        test_event_groups = []
        for _, test_path in TEST_SETS:
            test_meta = load_metadata_from_split(test_path)
            test_event_groups.extend(group_metadata_by_event(test_meta))

        # ========== 统一绘图 ==========
        print('\n====== 开始绘制性能与可视化图表 ======')

        # 1. 训练历史
        if hist is not None:
            trace_history, event_history = hist
            if trace_history is not None and event_history is not None:
                classifier.plot_optimized_training_history(trace_history, event_history)
                print("训练历史图已保存")
            else:
                print('（跳过训练历史图：无历史数据）')
        else:
            print('（跳过训练历史图：无训练历史）')

        # 2. 验证集性能
        print("绘制Trace性能对比图...")
        try:
            classifier.plot_trace_performance(val_event_groups)
            print("Trace性能对比图已保存")
        except Exception as e:
            print(f"绘制Trace性能对比图失败: {e}")

        # 3. 注意力权重相关
        print("绘制事件模型质量诊断图...")
        try:
            classifier.plot_quality_vs_attention(val_event_groups, test_event_groups)
            print("事件模型质量诊断图已保存")
        except Exception as e:
            print(f"绘制事件模型质量诊断图失败: {e}")

        # 4. 新增：单TRACE模型质量诊断图
        print("绘制单TRACE模型质量诊断图...")
        try:
            classifier.plot_trace_model_quality(val_event_groups, test_event_groups)
            print("单TRACE模型质量诊断图已保存")
        except Exception as e:
            print(f"绘制单TRACE模型质量诊断图失败: {e}")

        print('====== 所有图表已保存到项目目录 ======')

        # ========== 最终统计信息 ==========
        print("\n" + "=" * 60)
        print("最终统计信息:")
        print(f"- 验证集事件数: {len(val_event_groups)}")
        print(
            f"- 测试集事件数: {sum(len(group_metadata_by_event(load_metadata_from_split(test_path))) for _, test_path in TEST_SETS)}")

        # 计算总体trace数量
        total_traces = 0
        for _, traces in val_event_groups:
            total_traces += len(traces)
        for test_name, test_path in TEST_SETS:
            test_meta = load_metadata_from_split(test_path)
            test_groups = group_metadata_by_event(test_meta)
            for _, traces in test_groups:
                total_traces += len(traces)

        print(f"- 总trace数: {total_traces}")
        print(f"- 模型保存路径: {SAVE_MODEL_PATH}")
        print(f"- 结果输出路径: {RESULT_OUTPUT_PATH}")
        print("=" * 60)

    except Exception as e:
        import traceback
        print("\n" + "=" * 60)
        print("处理过程中发生错误:")
        traceback.print_exc()
        print("=" * 60)

        #  修复：使用正确的备份文件名
        if hasattr(classifier, 'model') and classifier.model is not None:
            print("尝试保存当前模型...")
            try:
                backup_path = SAVE_MODEL_PATH.replace('.keras', '_backup.keras')
                classifier.save_model(backup_path)
                print(f"模型已备份保存到: {backup_path}")
            except Exception as save_error:
                print(f"模型备份失败: {save_error}")
        exit(1)


if __name__ == "__main__":
    main()
