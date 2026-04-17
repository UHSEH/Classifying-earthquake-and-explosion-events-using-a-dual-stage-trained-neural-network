import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU, disable GPU

# =============================================================================
# Step 1: Import standard and third-party libraries
# =============================================================================
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from scipy.signal import spectrogram
import obspy
from obspy.core.trace import Trace
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import psutil
import logging
import gc
import time
from contextlib import contextmanager
from sklearn.metrics import accuracy_score

@tf.keras.utils.register_keras_serializable()
class QualityAwareAttentionLayer(layers.Layer):
    """
    Quality-aware attention layer - Fixed: completely removed ground truth dependency, using only prediction confidence
    """

    def __init__(self, confidence_temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.confidence_temperature = confidence_temperature
        self.content_dense1 = layers.Dense(64, activation='relu')
        self.content_dense2 = layers.Dense(1, activation='linear')
        self.supports_masking = True

    def build(self, input_shape):
        feature_shape, feat_input_shape = input_shape
        feature_shape = tf.TensorShape(feature_shape)
        self.content_dense1.build(feature_shape)
        dense1_output_shape = feature_shape[:-1] + (64,)
        self.content_dense2.build(dense1_output_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        features, feat_inputs = inputs

        # Base content attention
        content_attention = self.content_dense1(features)
        content_attention = self.content_dense2(content_attention)  # (batch_size, num_traces, 1)

        # Extract trace probabilities from feature inputs (dimension 9, index 8)
        trace_probs = feat_inputs[:, :, 8:9]  # (batch_size, num_traces, 1)

        # Fixed: completely removed ground truth dependency, using only prediction confidence to compute quality score
        confidence = tf.abs(trace_probs - 0.5) * 2.0 * self.confidence_temperature
        confidence = tf.clip_by_value(confidence, 0.0, 1.0)
        predicted_class = tf.cast(trace_probs > 0.5, tf.float32)
        effective_quality = (2.0 * predicted_class - 1.0) * confidence

        # Adjust attention with quality score - add numerical clipping to prevent overflow
        quality_adjustment = tf.where(
            effective_quality >= 0,
            tf.exp(tf.clip_by_value(effective_quality * 2, -10, 10)),
            tf.sigmoid(effective_quality * 10)
        )

        adjusted_attention = content_attention * quality_adjustment
        adjusted_attention = tf.squeeze(adjusted_attention, axis=-1)  # (batch_size, num_traces)

        # softmax normalization
        attention_weights = tf.nn.softmax(adjusted_attention, axis=1)  # (batch_size, num_traces)

        return layers.Reshape((-1, 1))(attention_weights)  # (batch_size, num_traces, 1)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]
        return None

    def get_config(self):
        config = super().get_config()
        config.update({'confidence_temperature': self.confidence_temperature})
        return config


@tf.keras.utils.register_keras_serializable()
def quality_function(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return (2.0 * y_true - 1.0) * (2.0 * y_pred - 1.0)


@tf.keras.utils.register_keras_serializable()
def bulletproof_trace_loss(y_true, y_pred):
    """
    Robust trace-level loss function, supports 2D [batch, traces] or 3D [batch, traces, 1] input.
    Automatically reshapes input to 2D and handles padding value -1.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    # Uniform reshape to 2D [batch, traces]
    batch_size = tf.shape(y_pred)[0]
    y_pred = tf.reshape(y_pred, [batch_size, -1])
    y_true = tf.reshape(y_true, [batch_size, -1])

    # Valid sample mask (exclude padding -1)
    valid_mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)

    # Replace padding -1 with 0 for loss calculation
    y_true_safe = tf.where(tf.equal(y_true, -1.0), 0.0, y_true)

    # Binary cross-entropy
    bce = - (y_true_safe * tf.math.log(y_pred) +
             (1 - y_true_safe) * tf.math.log(1 - y_pred))

    # Apply mask
    bce = bce * valid_mask

    # Total number of valid samples
    valid_count = tf.reduce_sum(valid_mask)

    # Avoid division by zero
    loss = tf.cond(
        tf.greater(valid_count, 0),
        lambda: tf.reduce_sum(bce) / valid_count,
        lambda: tf.constant(0.0)
    )

    return tf.maximum(loss, 0.0)


@tf.keras.utils.register_keras_serializable()
def bulletproof_event_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-3, 1 - 1e-3)
    eps = 1e-7
    loss = - (y_true * tf.math.log(y_pred + eps) +
              (1 - y_true) * tf.math.log(1 - y_pred + eps))
    return tf.maximum(tf.reduce_mean(loss), 0.0)


def build_safe_optimizer(lr):
    return keras.optimizers.Adam(
        learning_rate=lr,
        global_clipnorm=1.0,
        epsilon=1e-6
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
            print(f"\nKillOnNegativeLoss: {self.count}th negative loss ({main_loss:.4f})")
            if self.count >= self.patience:
                print("Stopping training, check loss function!")
                self.model.stop_training = True


@tf.keras.utils.register_keras_serializable()
def stable_weighted_binary_crossentropy(class_weights):
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
        loss = tf.maximum(loss, 0.0)

        return loss

    return loss_fn


# Set GPU memory growth to avoid OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth configured: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()
    print("Eager Execution mode enabled")

# -------------------------- Configuration Parameters --------------------------
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

# Signal processing parameters
SAMPLE_RATE = 75
WAVEFORM_DURATION = 60
WAVEFORM_LENGTH = int(SAMPLE_RATE * WAVEFORM_DURATION)
HIGHPASS_FREQ = 2
VALID_COMPONENTS = ["Z"]

# Event processing parameters
MIN_TRACES_PER_EVENT = 1
MAX_TRACES_PER_EVENT = None

# Spectrogram parameters
SPECTROGRAM_NPERS = int(2 * SAMPLE_RATE)
SPECTROGRAM_NOVER = int(SPECTROGRAM_NPERS * 0.75)
SPECTROGRAM_FREQ_MIN = 0.5
SPECTROGRAM_FREQ_MAX = 50

# Precompute spectrogram dimensions
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
print(f"Precomputed spectrogram dimensions: (height={SPEC_HEIGHT}, width={SPEC_WIDTH})")
del _f, _t, _spec_freq_mask

# Training parameters
BATCH_SIZE = 32
TRACE_BATCH_SIZE = 32
EPOCHS = 400
TRACE_PRETRAIN_EPOCHS = 400
EXPLOSION_WEIGHT_SCALE = 1
LEARNING_RATE = 1e-4
TRACE_LEARNING_RATE = 1e-5
MAX_ERRORS = 1000
ERROR_LOG_INTERVAL = 100
INTERMEDIATE_LOSS_WEIGHT = 0.05
NUM_ATTENTION_HEADS = 4

# Column name mapping
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

# Initialize error logging
logging.basicConfig(
    filename=ERROR_LOG_PATH,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
error_logger = logging.getLogger("data_processor")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


@tf.keras.utils.register_keras_serializable()
class TraceAttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return tf.nn.softmax(x, axis=1)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class MultiHeadTraceAttention(layers.Layer):
    def __init__(self, num_heads=NUM_ATTENTION_HEADS, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_heads = [layers.Dense(1, activation='sigmoid') for _ in range(num_heads)]
        self.fuse = layers.Dense(1)

    def call(self, inputs, training=None):
        head_scores = [head(inputs) for head in self.attention_heads]
        stacked = tf.concat(head_scores, axis=-1)
        raw_score = self.fuse(stacked)
        raw_score = tf.squeeze(raw_score, axis=-1)
        weight = raw_score / (tf.reduce_sum(raw_score, axis=1, keepdims=True) + 1e-8)
        return tf.expand_dims(weight, -1)

    def get_config(self):
        return dict(list(super().get_config().items()) + [('num_heads', self.num_heads)])


@tf.keras.utils.register_keras_serializable()
class ExpandLastDimLayer(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config


@tf.keras.utils.register_keras_serializable()
class AttentionNormalizeLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        attention_sum = tf.reduce_sum(inputs, axis=1, keepdims=True) + 1e-8
        return inputs / attention_sum

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class FusedVectorLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        fused = tf.reduce_sum(inputs, axis=1)
        return fused

    def get_config(self):
        config = super().get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class SqueezeLastDimLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True  # Key: declare masking support

    def call(self, inputs):
        return tf.squeeze(inputs, axis=-1)

    def compute_mask(self, inputs, mask=None):

        if mask is not None:
            # After squeeze, mask shape matches output, return directly
            return mask
        return None

    def get_config(self):
        config = super().get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class MaskedAccuracy(tf.keras.metrics.Metric):
    """
    Mask-aware accuracy metric
    Excludes padding value (default -1) when computing accuracy
    """

    def __init__(self, mask_value=-1, name='masked_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mask_value = float(mask_value)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert y_true to float32 for comparison
        y_true = tf.cast(y_true, tf.float32)

        # Create mask: exclude padding value
        mask = tf.not_equal(y_true, self.mask_value)
        mask = tf.cast(mask, tf.float32)

        # Convert predictions to binary (>0.5 is 1, else 0)
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)

        # Compute matches (only at mask positions)
        matches = tf.cast(tf.equal(y_pred_binary, y_true), tf.float32)
        matches = matches * mask

        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total / (self.count + 1e-7)

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)

    def get_config(self):
        config = super().get_config()
        config.update({'mask_value': self.mask_value})
        return config

CUSTOM_OBJECTS = {
    # Existing custom layers
    'TraceAttentionLayer': TraceAttentionLayer,
    'MultiHeadTraceAttention': MultiHeadTraceAttention,
    'FusedVectorLayer': FusedVectorLayer,
    'ExpandLastDimLayer': ExpandLastDimLayer,
    'AttentionNormalizeLayer': AttentionNormalizeLayer,
    'QualityAwareAttentionLayer': QualityAwareAttentionLayer,
    'SqueezeLastDimLayer': SqueezeLastDimLayer,
    'bulletproof_event_loss': bulletproof_event_loss,
    'bulletproof_trace_loss': bulletproof_trace_loss,
    'MaskedAccuracy': MaskedAccuracy,
    'weighted_binary_crossentropy': stable_weighted_binary_crossentropy,
    'quality_function': quality_function,
}

class H5FileManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self._file = None
        self._is_open = False

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        if not os.path.isfile(file_path):
            raise IsADirectoryError(f"{file_path} is a directory, not a file")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"HDF5 file is empty: {file_path}")

    def open(self):
        if self._is_open:
            return self._file

        self.close()
        try:
            self._file = h5py.File(self.file_path, "r", swmr=True)
            self._is_open = True
            if len(list(self._file.keys())) == 0:
                raise RuntimeError(f"HDF5 file is corrupted or empty: {self.file_path}")
            return self._file
        except Exception as e:
            raise RuntimeError(f"Failed to open HDF5 file: {str(e)}") from e

    def close(self):
        if self._is_open and self._file is not None:
            try:
                self._file.close()
            except Exception as e:
                error_logger.error(f"HDF5 close error: {str(e)}")
            finally:
                self._file = None
                self._is_open = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def split_feat_vector(feat_vector):
    norm_part = np.array([feat_vector[0],
                          feat_vector[1],
                          feat_vector[3],
                          feat_vector[5],
                          feat_vector[6],
                          feat_vector[7]], dtype=np.float32)
    dist = feat_vector[4]
    depth = feat_vector[2]
    return norm_part, dist, depth


def merge_feat_vector(norm_part, dist, depth):
    return np.array([norm_part[0],
                     norm_part[1],
                     depth,
                     norm_part[2],
                     dist,
                     norm_part[3],
                     norm_part[4],
                     norm_part[5]],
                    dtype=np.float32)


def print_memory_usage(prefix=""):
    try:
        process = psutil.Process(os.getpid())
        mem_used = process.memory_info().rss / (1024 ** 3)
        mem_percent = process.memory_percent()
        print(f"{prefix}Memory usage: {mem_used:.2f} GB ({mem_percent:.1f}%)")
    except Exception:
        pass


@contextmanager
def timing_context(description):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{description} took: {end_time - start_time:.2f} seconds")


def parse_trace_name(trace_name_str):
    if "$" not in trace_name_str:
        raise ValueError(f"Invalid trace_name format: {trace_name_str} (missing '$' separator)")
    bucket_part = trace_name_str.split("$")[0]
    event_idx_part = trace_name_str.split("$")[1].split(",")[0]
    try:
        event_idx = int(event_idx_part.strip())
    except ValueError:
        raise ValueError(f"Invalid index in trace_name: {trace_name_str} (index part: {event_idx_part})")
    return f"data/{bucket_part}", event_idx


def get_component_order(h5_file):
    comp_key = "data_format/component_order"
    if comp_key not in h5_file:
        raise KeyError(f"HDF5 missing required component key: {comp_key} (available keys: {list(h5_file.keys())})")
    comp_str = h5_file[comp_key][()].decode("utf-8").strip().upper()
    raw_comp = comp_str.split(",") if "," in comp_str else [c for c in comp_str]
    comp_map = {"E": "X", "N": "Y", "Z": "Z"}
    try:
        comp_order = [comp_map[c.strip()] for c in raw_comp]
    except KeyError as e:
        raise ValueError(f"Unsupported component: {e} (available components in HDF5: {raw_comp})")
    if VALID_COMPONENTS[0] not in comp_order:
        raise ValueError(
            f"Target component {VALID_COMPONENTS[0]} not found in HDF5 (available: {comp_order})")
    return comp_order


def calculate_arrival_time(start_dt, arrival_sample, sampling_rate):
    if arrival_sample < 0:
        raise ValueError(f"Negative arrival sample: {arrival_sample}")
    if sampling_rate <= 0:
        raise ValueError(f"Invalid sampling rate: {sampling_rate} (must be positive)")
    arrival_offset = arrival_sample / sampling_rate
    return start_dt + timedelta(seconds=arrival_offset)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
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
    """
    Modified preprocessing function: uses 10-18 Hz bandpass filter for P/S ratio calculation.
    Other processing remains consistent with original code.
    """
    from scipy.signal import butter, filtfilt

    if len(raw_wave) == 0:
        raise ValueError("Raw waveform data is empty")
    if np.all(raw_wave == 0):
        raise ValueError("Raw waveform is all zeros (no valid signal)")

    # Create Trace object
    trace = Trace(
        data=raw_wave.copy(),
        header={"sampling_rate": raw_sr, "starttime": obspy.UTCDateTime(start_dt)}
    )
    trace.detrend("demean").detrend("linear")
    trace.taper(max_percentage=0.05, type="hann")

    # Highpass filter (consistent with original code)
    nyquist_freq = raw_sr / 2.0
    actual_highpass = min(HIGHPASS_FREQ, nyquist_freq - 0.1)
    if actual_highpass > 0:
        trace.filter('highpass', freq=actual_highpass, corners=4, zerophase=True)

    # Resample to target sampling rate
    if trace.stats.sampling_rate != SAMPLE_RATE:
        trace.resample(SAMPLE_RATE, no_filter=True)

    # Normalization
    max_amp = np.max(np.abs(trace.data))
    if max_amp > 0:
        trace.data /= max_amp
    else:
        trace.data = np.zeros_like(trace.data)

    # Extract waveform (starting 15 seconds before P arrival)
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

    # Add noise during training (consistent with original code)
    if is_training:
        noise = np.random.normal(0, 0.01, size=processed_wave.shape)
        processed_wave = processed_wave + noise

    # =============================================================================
    # Key modification: Use 10-18 Hz bandpass filter for P/S ratio calculation
    # =============================================================================
    # Create bandpass-filtered trace copy specifically for P/S calculation
    trace_ps = trace.copy()

    # 10-18 Hz bandpass filter (4th order Butterworth)
    nyq = 0.5 * SAMPLE_RATE
    low = 10.0 / nyq
    high = 18.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    trace_ps.data = filtfilt(b, a, trace_ps.data)

    # Compute S-P time difference to determine time windows (consistent with original)
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

    # Compute Pg amplitude (after 10-18 Hz bandpass)
    pg_start_rel = p_time_rel - pg_window
    pg_end_rel = p_time_rel + pg_window
    pg_start_idx = max(0, int(pg_start_rel * SAMPLE_RATE))
    pg_end_idx = min(len(trace_ps.data), int(pg_end_rel * SAMPLE_RATE))
    pg_amp = np.max(np.abs(trace_ps.data[pg_start_idx:pg_end_idx])) if (pg_end_idx > pg_start_idx) else 1e-6
    pg_amp = max(pg_amp, 1e-6)

    # Compute Sg amplitude (after 10-18 Hz bandpass)
    sg_start_rel = (s_arrival - start_dt).total_seconds() - sg_window
    sg_end_rel = (s_arrival - start_dt).total_seconds() + sg_window
    sg_start_idx = max(0, int(sg_start_rel * SAMPLE_RATE))
    sg_end_idx = min(len(trace_ps.data), int(sg_end_rel * SAMPLE_RATE))
    sg_amp = np.max(np.abs(trace_ps.data[sg_start_idx:sg_end_idx])) if (sg_end_idx > sg_start_idx) else 1e-6
    sg_amp = max(sg_amp, 1e-6)

    # Compute P/S ratio (using amplitudes after 10-18 Hz bandpass)
    pg_sg_ratio = pg_amp / sg_amp
    pg_sg_ratio = np.clip(pg_sg_ratio, np.exp(-5), np.exp(5))
    log_pg_sg = np.log(pg_sg_ratio)
    if np.isnan(log_pg_sg):
        log_pg_sg = 0.0

    return processed_wave, log_pg_sg

def calculate_spectrogram(waveform_data):
    if len(waveform_data) != WAVEFORM_LENGTH:
        raise RuntimeError(f"Invalid waveform length: expected {WAVEFORM_LENGTH}, got {len(waveform_data)}")
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
            f"Invalid spectrogram dimensions! Expected ({SPEC_HEIGHT},{SPEC_WIDTH}), got {Sxx_filtered.shape} "
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
    if not np.isfinite(coeffs[0]):
        return 1.0
    return np.clip(coeffs[0], 1.0, 2.0)


def parse_time_str(time_str):
    time_str = str(time_str).strip()

    if "+00:00" in time_str:
        time_str = time_str.split("+00:00")[0].strip()
    elif "Z" in time_str:
        time_str = time_str.replace("Z", "").strip()

    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse time string: {time_str}")


def extract_time_features(event_time_str):
    try:
        dt = parse_time_str(event_time_str)
        return [
            np.sin(2 * np.pi * dt.hour / 24),
            np.cos(2 * np.pi * dt.hour / 24),
            1 if dt.weekday() < 5 else 0
        ]
    except Exception as e:
        print(f"Warning: Cannot extract time features from '{event_time_str}', using defaults: {e}")
        return [0.0, 0.0, 0.0]


def dynamic_padding(traces_list, max_traces):
    if max_traces is None:
        raise ValueError("max_traces cannot be None, must specify target padding length")
    if not isinstance(max_traces, int) or max_traces <= 0:
        raise ValueError(f"max_traces must be a positive integer, got {max_traces}")

    if isinstance(traces_list, np.ndarray):
        traces_list = traces_list.tolist()

    current_count = len(traces_list)
    if current_count >= max_traces:
        truncated = traces_list[:max_traces]
        mask = np.ones(max_traces, dtype=np.float32)
        return truncated, mask

    if current_count == 0:
        raise ValueError("dynamic_padding input is empty")

    padded = traces_list.copy()
    zero_sample = np.zeros_like(traces_list[0])
    mask = [1.0] * current_count

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
            print(f"Warning: Found {len(unique_missing)} non-existent HDF5 paths:")
            for path in unique_missing:
                print(f"  - {path}")
        return len(missing_paths) == 0


def check_h5_integrity(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            buckets = [f"data/bucket{i}" for i in range(1, 11)]
            for bucket in buckets:
                if bucket not in f:
                    print(f"Missing bucket: {bucket}")
                else:
                    print(f"Found bucket: {bucket}, shape: {f[bucket].shape}")
            return True
    except Exception as e:
        print(f"HDF5 integrity check failed: {e}")
        return False


def load_metadata_from_split(path):
    metadata_filename = "metadata.csv"
    metadata_path = os.path.join(path, metadata_filename)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    required_columns = list(COLUMN_MAPPING.values())
    if "event_id" not in required_columns:
        required_columns.append("event_id")

    metadata = pd.read_csv(metadata_path, usecols=required_columns)

    missing_cols = [col for col in required_columns if col not in metadata.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

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
        raise ValueError(f"No valid samples in {path}")

    event_dist = metadata["event_type"].value_counts().to_dict()
    print(
        f"Loaded {path}: {len(metadata)} valid samples (distribution: earthquake={event_dist.get('earthquake', 0)}, explosion={event_dist.get('explosion', 0)})")

    reverse_mapping = {v: k for k, v in COLUMN_MAPPING.items()}
    metadata_renamed = metadata.rename(columns=reverse_mapping)

    if "event_id" not in metadata_renamed.columns and "event_id" in metadata.columns:
        metadata_renamed["event_id"] = metadata["event_id"]

    return metadata_renamed


def group_metadata_by_event(metadata):
    global MAX_TRACES_PER_EVENT

    event_id_groups = metadata.groupby("event_id").groups
    all_events = []

    total_events = len(event_id_groups)
    events_discarded = 0
    trace_count_distribution = {}

    print(f"\n=== Event grouping detailed statistics (keeping single-trace events) ===")
    print(f"Total original events: {total_events}")

    for event_id, trace_indices in event_id_groups.items():
        traces = metadata.loc[trace_indices]
        event_type = traces["event_type"].iloc[0]
        trace_count = len(traces)

        if trace_count not in trace_count_distribution:
            trace_count_distribution[trace_count] = 0
        trace_count_distribution[trace_count] += 1

        all_events.append((event_type, traces))

    if all_events:
        trace_counts = [len(traces) for _, traces in all_events]
        proposed_max = int(np.percentile(trace_counts, 98))
        MAX_TRACES_PER_EVENT = max(1, proposed_max)
    else:
        MAX_TRACES_PER_EVENT = 1

    print(f"Total events kept: {len(all_events)}")
    print(f"Events discarded: {events_discarded}")
    print(f"Utilization rate: {len(all_events) / total_events * 100:.1f}%")
    print(f"Dynamically set MAX_TRACES_PER_EVENT = {MAX_TRACES_PER_EVENT}")

    print(f"\nTrace count distribution:")
    for count in sorted(trace_count_distribution.keys()):
        events_with_count = trace_count_distribution[count]
        percentage = events_with_count / total_events * 100
        print(f"  {count} traces: {events_with_count} events ({percentage:.1f}%) [all kept]")

    event_types = [event_type for event_type, _ in all_events]
    unique_types, counts = np.unique(event_types, return_counts=True)
    type_dist = dict(zip(unique_types, counts))
    print(f"\nFinal event type distribution: {type_dist}")

    np.random.shuffle(all_events)
    return all_events


def is_valid_trace(waveform):
    if len(waveform) == 0:
        return False
    if np.all(waveform == 0):
        return False
    if np.std(waveform) < 1e-8:
        return False
    if np.any(np.isnan(waveform)) or np.any(np.isinf(waveform)):
        return False
    return True


def trace_generator(metadata, h5_manager, scaler=None, shuffle=False, is_training=False, max_epochs=1):
    """
    Trace-level data generator - fixed validation set shuffle and repeat issues
    """
    if shuffle:
        metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)

    total_samples = len(metadata)
    epoch_count = 0

    try:
        h5_file = h5_manager.open()
        print(f"[Trace Generator] HDF5 file opened: {h5_manager.file_path}")
    except Exception as e:
        print(f"[Trace Generator] Failed to open HDF5: {e}")
        raise

    comp_order_cache = None

    try:
        # Key fix: validation set runs only specified epochs, training loops indefinitely
        while (is_training and True) or (epoch_count < max_epochs):
            error_count = 0
            success_count = 0
            empty_trace_count = 0

            # Reshuffle each epoch (training only)
            if epoch_count > 0 and shuffle and is_training:
                current_metadata = metadata.sample(frac=1, random_state=None).reset_index(drop=True)
                print(f"[Trace Generator] Epoch {epoch_count + 1}: data reshuffled")
            else:
                current_metadata = metadata

            for idx, (_, row) in enumerate(current_metadata.iterrows()):
                if idx % 5000 == 0 or (idx < 1000 and idx % 100 == 0):
                    print(f"Trace Generator [{'training' if is_training else 'validation'}] "
                          f"Epoch{epoch_count + 1} progress: {idx}/{total_samples} "
                          f"(success: {success_count}, failed: {error_count}, empty traces: {empty_trace_count})")

                trace_name = row["trace_name"]
                try:
                    if h5_file is None or not h5_file:
                        print(f"[Generator] Warning: HDF5 file status abnormal, attempting to reopen...")
                        h5_file = h5_manager.open()
                        comp_order_cache = None

                    if comp_order_cache is None:
                        comp_order_cache = get_component_order(h5_file)

                    target_comp_idx = comp_order_cache.index(VALID_COMPONENTS[0])
                    hdf5_path, event_idx = parse_trace_name(trace_name)

                    if hdf5_path not in h5_file:
                        error_count += 1
                        if error_count <= 5:
                            print(f"[Generator] Path does not exist: {hdf5_path}")
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

                    processed_wave, log_pg_sg = preprocess_waveform(
                        raw_wave, raw_sr, p_arrival, s_arrival, start_dt, is_training
                    )
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

                    if scaler is not None:
                        norm_part, dist, depth = split_feat_vector(feat_vector)
                        norm_part = scaler.transform(norm_part.reshape(1, -1)).flatten()
                        feat_vector = merge_feat_vector(norm_part, dist, depth)

                    label = 1 if row["event_type"] == "earthquake" else 0

                    success_count += 1
                    yield (processed_wave, spec_data, feat_vector), np.array(label, dtype=np.int8)

                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"[Generator] Processing {trace_name} failed: {str(e)}")
                    continue

            print(f"Trace Generator Epoch{epoch_count + 1} completed: "
                  f"success {success_count}, failed {error_count}, empty traces {empty_trace_count}")

            epoch_count += 1

            # Key fix: validation mode exits after completion, no repetition
            if not is_training:
                print(f"Validation mode completed, processed {success_count} samples")
                break

            if success_count < 1000:
                print(f"Warning: Epoch {epoch_count} only {success_count} successful samples, "
                      f"possible data reading issue, attempting to reopen file...")
                try:
                    h5_manager.close()
                    h5_file = h5_manager.open()
                    comp_order_cache = None
                    print("[Generator] File reopened")
                except Exception as e:
                    print(f"[Generator] Failed to reopen file: {e}")

    finally:
        print("[Generator] Closing HDF5 file")
        h5_manager.close()


def event_generator(event_groups, h5_path, scaler=None, shuffle=False,
                    trace_prob_cache=None, is_training=False, mode='auto',
                    max_traces=None):
    if trace_prob_cache is None:
        raise RuntimeError("trace_prob_cache must be provided!")
    if max_traces is None:
        raise ValueError("max_traces parameter must be specified, cannot be None")

    if mode == 'auto':
        mode = 'train' if is_training else 'inference'

    if shuffle:
        event_groups = event_groups.copy()
        np.random.shuffle(event_groups)

    total_events = len(event_groups)
    generated_events = 0
    skipped_events = 0

    with h5py.File(h5_path, 'r', swmr=True) as h5_file:
        comp_order_cache = None
        for event_type, traces in event_groups:
            try:
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

                        start_dt = parse_time_str(row["trace_start_time"])
                        raw_sr = float(row["trace_sampling_rate_hz"])
                        p_arrival = calculate_arrival_time(start_dt, int(row["trace_P_arrival_sample"]), raw_sr)
                        s_arrival = calculate_arrival_time(start_dt, int(row["trace_S_arrival_sample"]), raw_sr)
                        processed_wave, log_pg_sg = preprocess_waveform(
                            raw_wave, raw_sr, p_arrival, s_arrival, start_dt, is_training
                        )
                        spec_data = calculate_spectrogram(processed_wave)
                        processed_wave = np.expand_dims(processed_wave, axis=-1)
                        spec_data = np.expand_dims(spec_data, axis=-1)

                        fractal_dim = calculate_fractal_dimension(processed_wave[:, 0])
                        epicentral_distance = haversine_distance(
                            row["source_latitude_deg"], row["source_longitude_deg"],
                            row["station_latitude_deg"], row["station_longitude_deg"]
                        )
                        time_features = extract_time_features(row["origin_time"])

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

                        if scaler is not None:
                            norm_part, dist, depth = split_feat_vector(feat_vector)
                            norm_part = scaler.transform(norm_part.reshape(1, -1)).flatten()
                            feat_vector = merge_feat_vector(norm_part, dist, depth)

                        if trace_name not in trace_prob_cache:
                            continue
                        trace_prob = trace_prob_cache[trace_name]

                        # Fixed: completely removed ground truth dependency, using only prediction result to compute quality score
                        confidence = abs(trace_prob - 0.5) * 2.0
                        predicted_class = 1 if trace_prob > 0.5 else 0
                        quality_score = float((2 * predicted_class - 1) * confidence)

                        enhanced_feat_vector = np.append(feat_vector, [trace_prob, quality_score])

                        event_waves.append(processed_wave)
                        event_specs.append(spec_data)
                        event_feats.append(enhanced_feat_vector)
                        valid_traces_count += 1

                        del processed_wave, spec_data, raw_wave

                    except Exception as e:
                        continue

                if valid_traces_count > 0:
                    waves_padded, wave_mask = dynamic_padding(event_waves, max_traces)
                    specs_padded, spec_mask = dynamic_padding(event_specs, max_traces)
                    feats_padded, feat_mask = dynamic_padding(event_feats, max_traces)

                    assert np.array_equal(wave_mask, spec_mask) and np.array_equal(wave_mask, feat_mask)

                    event_waves_np = np.stack(waves_padded, axis=0)
                    event_specs_np = np.stack(specs_padded, axis=0)
                    event_feats_np = np.stack(feats_padded, axis=0)
                    mask_np = wave_mask

                    event_label = 1 if event_type == "earthquake" else 0
                    trace_labels = np.full((max_traces,), -1, dtype=np.int8)
                    trace_labels[:valid_traces_count] = event_label
                    feats_with_mask = np.concatenate([event_feats_np, mask_np.reshape(-1, 1)], axis=-1)

                    generated_events += 1
                    yield (event_waves_np, event_specs_np, feats_with_mask, mask_np), \
                        (np.int8(event_label), trace_labels)
                else:
                    skipped_events += 1

            except Exception as e:
                skipped_events += 1
                continue

    print(f"[Generator Stats] Total events: {total_events}, generated: {generated_events}, skipped: {skipped_events}")


def build_trace_tf_dataset(metadata, h5_manager, scaler=None, shuffle=False,
                           batch_size=TRACE_BATCH_SIZE, is_training=False):
    """
    Build trace-level TensorFlow dataset - fixed: strict separation of train/val, validation no shuffle/no repeat
    """
    output_types = ((tf.float32, tf.float32, tf.float32), tf.int8)
    output_shapes = (
        (
            tf.TensorShape([WAVEFORM_LENGTH, 1]),
            tf.TensorShape([SPEC_HEIGHT, SPEC_WIDTH, 1]),
            tf.TensorShape([8])
        ),
        tf.TensorShape([])
    )

    # Fixed: validation runs only 1 epoch, training loops indefinitely
    max_epochs = 1 if not is_training else 1  # Generator controls loop internally

    def generator_factory():
        return trace_generator(metadata, h5_manager, scaler, shuffle, is_training, max_epochs)

    dataset = tf.data.Dataset.from_generator(
        generator_factory,
        output_types=output_types,
        output_shapes=output_shapes
    )

    # Fixed: training repeats, validation does not repeat
    if is_training:
        dataset = dataset.repeat()
        print("[DEBUG] Trace training dataset: repeat enabled")
    else:
        dataset = dataset.take(len(metadata))  # Limit validation set size
        print("[DEBUG] Trace validation dataset: no repeat, traverse once only")

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_event_tf_dataset(event_groups, h5_path, scaler=None, shuffle=False,
                           batch_size=BATCH_SIZE, trace_prob_cache=None,
                           is_training=False, mode='auto', max_traces=None):
    if max_traces is None:
        raise ValueError("max_traces parameter must be specified, cannot be None")
    if trace_prob_cache is None:
        raise RuntimeError("trace_prob_cache must be provided!")

    print(
        f"[DEBUG] Building event dataset: {len(event_groups)} events, batch_size={batch_size}, mode={mode}, max_traces={max_traces}")

    MAX_RETRY = 3

    def gen():
        epoch = 0
        retry_count = 0

        while True:
            try:
                if shuffle and epoch > 0:
                    indices = np.random.permutation(len(event_groups))
                    current_groups = [event_groups[i] for i in indices]
                else:
                    current_groups = event_groups

                generated_events = 0
                skipped_events = 0

                try:
                    with h5py.File(h5_path, 'r', swmr=True) as h5_file:
                        comp_order_cache = None

                        for event_type, traces in current_groups:
                            try:
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

                                        start_dt = parse_time_str(row["trace_start_time"])
                                        raw_sr = float(row["trace_sampling_rate_hz"])
                                        p_arrival = calculate_arrival_time(start_dt, int(row["trace_P_arrival_sample"]),
                                                                           raw_sr)
                                        s_arrival = calculate_arrival_time(start_dt, int(row["trace_S_arrival_sample"]),
                                                                           raw_sr)
                                        processed_wave, log_pg_sg = preprocess_waveform(
                                            raw_wave, raw_sr, p_arrival, s_arrival, start_dt, is_training
                                        )
                                        spec_data = calculate_spectrogram(processed_wave)
                                        processed_wave = np.expand_dims(processed_wave, axis=-1)
                                        spec_data = np.expand_dims(spec_data, axis=-1)

                                        fractal_dim = calculate_fractal_dimension(processed_wave[:, 0])
                                        epicentral_distance = haversine_distance(
                                            row["source_latitude_deg"], row["source_longitude_deg"],
                                            row["station_latitude_deg"], row["station_longitude_deg"]
                                        )
                                        time_features = extract_time_features(row["origin_time"])

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

                                        if scaler is not None:
                                            norm_part, dist, depth = split_feat_vector(feat_vector)
                                            norm_part = scaler.transform(norm_part.reshape(1, -1)).flatten()
                                            feat_vector = merge_feat_vector(norm_part, dist, depth)

                                        if trace_name not in trace_prob_cache:
                                            continue
                                        trace_prob = trace_prob_cache[trace_name]

                                        if mode == 'train':
                                            true_label = 1 if row["event_type"] == "earthquake" else 0
                                            quality_score = float((2 * true_label - 1) * (2 * trace_prob - 1))
                                        else:
                                            confidence = abs(trace_prob - 0.5) * 2.0
                                            predicted_class = 1 if trace_prob > 0.5 else 0
                                            quality_score = float((2 * predicted_class - 1) * confidence)

                                        enhanced_feat_vector = np.append(feat_vector, [trace_prob, quality_score])

                                        event_waves.append(processed_wave)
                                        event_specs.append(spec_data)
                                        event_feats.append(enhanced_feat_vector)
                                        valid_traces_count += 1

                                        del processed_wave, spec_data, raw_wave

                                    except Exception as e:
                                        continue

                                if valid_traces_count > 0:
                                    waves_padded, wave_mask = dynamic_padding(event_waves, max_traces)
                                    specs_padded, spec_mask = dynamic_padding(event_specs, max_traces)
                                    feats_padded, feat_mask = dynamic_padding(event_feats, max_traces)

                                    assert np.array_equal(wave_mask, spec_mask) and np.array_equal(wave_mask, feat_mask)

                                    event_waves_np = np.stack(waves_padded, axis=0)
                                    event_specs_np = np.stack(specs_padded, axis=0)
                                    event_feats_np = np.stack(feats_padded, axis=0)
                                    mask_np = wave_mask

                                    event_label = 1 if event_type == "earthquake" else 0
                                    trace_labels = np.full((max_traces,), -1, dtype=np.int8)
                                    trace_labels[:valid_traces_count] = event_label
                                    feats_with_mask = np.concatenate([event_feats_np, mask_np.reshape(-1, 1)], axis=-1)

                                    generated_events += 1
                                    yield (event_waves_np, event_specs_np, feats_with_mask, mask_np), \
                                        (np.int8(event_label), trace_labels)
                                else:
                                    skipped_events += 1

                            except Exception as e:
                                skipped_events += 1
                                continue

                        retry_count = 0

                except Exception as e:
                    retry_count += 1
                    if retry_count >= MAX_RETRY:
                        print(f"[Generator epoch {epoch}] Critical error, giving up after {MAX_RETRY} retries: {e}")
                        raise RuntimeError(f"HDF5 processing failed: {e}")

                    print(f"[Generator epoch {epoch}] Error: {e}, retry {retry_count}...")
                    time.sleep(1)
                    continue

                print(f"[Generator epoch {epoch}] Generated: {generated_events}, Skipped: {skipped_events}")
                epoch += 1

                if not is_training:
                    break

            except Exception as e:
                retry_count += 1
                if retry_count >= MAX_RETRY:
                    print(f"[Generator epoch {epoch}] Uncaught exception, giving up after {MAX_RETRY} retries: {e}")
                    raise

                print(f"[Generator epoch {epoch}] Uncaught exception: {e}, retry {retry_count}...")
                epoch += 1
                time.sleep(1)

    output_types = (
        (tf.float32, tf.float32, tf.float32, tf.float32),
        (tf.int8, tf.int8)
    )
    output_shapes = (
        (tf.TensorShape([max_traces, WAVEFORM_LENGTH, 1]),
         tf.TensorShape([max_traces, SPEC_HEIGHT, SPEC_WIDTH, 1]),
         tf.TensorShape([max_traces, 11]),
         tf.TensorShape([max_traces])),
        (tf.TensorShape([]),
         tf.TensorShape([max_traces]))
    )

    dataset = tf.data.Dataset.from_generator(
        gen, output_types=output_types, output_shapes=output_shapes
    )

    if is_training:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_cosine_scheduler(initial_lr, total_epochs):
    def lr_scheduler(epoch):
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

    return lr_scheduler


def create_trace_adaptive_scheduler(initial_lr, total_epochs):
    warmup_epochs = 5

    def lr_scheduler(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))

    return lr_scheduler


class ExplosionRecallLogger(Callback):
    def __init__(self, val_dataset, val_steps, is_trace_model=False):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_steps = val_steps
        self.is_trace_model = is_trace_model

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred_prob = []

        try:
            if self.is_trace_model:
                for (x1, x2, x3), y_label in self.val_dataset.take(self.val_steps):
                    pred = self.model.predict([x1, x2, x3], verbose=0)
                    y_true.extend(y_label.numpy().tolist())
                    y_pred_prob.extend(pred.flatten().tolist())
            else:
                for (x1, x2, x3, mask_input), (y_event, y_trace) in self.val_dataset.take(self.val_steps):
                    pred = self.model.predict([x1, x2, x3, mask_input], verbose=0)
                    pred = pred[0]
                    y_true.extend(y_event.numpy().tolist())
                    y_pred_prob.extend(pred.flatten().tolist())
        except ValueError as e:
            print(f"Data unpacking error: {str(e)}")
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


class ValidationSanityCheck(Callback):
    """
    Enhanced: validation set integrity check callback - detects data leakage and metric anomalies
    """

    def __init__(self, val_dataset=None, val_steps=None):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_steps = val_steps
        self.epoch_logs = []
        self.train_val_overlap = False

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        # Get metrics (compatible with different naming)
        train_loss = logs.get('loss', float('inf'))
        val_loss = logs.get('val_loss', float('inf'))

        # Attempt to get accuracy (trace and event models have different naming)
        train_acc = (logs.get('accuracy') or
                     logs.get('event_output_accuracy') or
                     logs.get('trace_output_accuracy') or 0)
        val_acc = (logs.get('val_accuracy') or
                   logs.get('val_event_output_accuracy') or
                   logs.get('val_trace_output_accuracy') or 0)

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        self.epoch_logs.append(log_entry)

        print(f"\n[SanityCheck] Epoch {epoch + 1}")
        print(f"  Training: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Validation: loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Detect anomaly: validation accuracy significantly higher than training
        if val_acc > train_acc + 0.1:
            print(f"  Warning: validation accuracy ({val_acc:.3f}) significantly higher than training ({train_acc:.3f}), possible data leakage!")
            # Additional check: validation loss too low
            if val_loss < 0.1 and train_loss > 0.3:
                print(f"  Severe warning: validation loss abnormally low, confirm data separation!")
                self.train_val_overlap = True

        # Detect anomaly: validation loss significantly lower than training
        if val_loss < train_loss * 0.5 and train_loss > 0.1:
            print(f"  Warning: validation loss ({val_loss:.3f}) much lower than training ({train_loss:.3f}), possible data leakage!")

        # Detect violent fluctuations in validation metrics
        if len(self.epoch_logs) >= 3:
            recent_val_accs = [e['val_acc'] for e in self.epoch_logs[-3:]]
            acc_range = max(recent_val_accs) - min(recent_val_accs)
            if acc_range > 0.15:
                print(f"  Warning: validation accuracy fluctuates too much (range {acc_range:.3f}), validation set may be unstable!")

        # Detect inflated precision/recall
        val_precision = logs.get('val_precision') or logs.get('val_event_output_precision') or 0
        val_recall = logs.get('val_recall') or logs.get('val_event_output_recall') or 0

        if val_precision > 0.95 and train_acc < 0.8:
            print(f"  Warning: validation precision ({val_precision:.3f}) inflated, check class distribution!")

        if val_recall > 0.95 and train_acc < 0.8:
            print(f"  Warning: validation recall ({val_recall:.3f}) inflated, check mask handling!")


class EventAggregationMonitor(keras.callbacks.Callback):
    """
    Fixed version: event-level aggregation method comparison monitor
    Fixes:
    1. Removed dynamic unfreezing of encoder and recompilation logic (moved externally)
    2. Rebuild dataset each epoch to avoid iterator exhaustion
    3. Ensure attention weights computed using current model state
    4. Added early stopping detection to avoid ineffective training
    """

    def __init__(self, val_event_groups, val_trace_prob_cache, scaler, max_traces,
                 h5_path, batch_size=32):
        super().__init__()
        self.val_event_groups = val_event_groups
        self.val_trace_prob_cache = val_trace_prob_cache
        self.scaler = scaler
        self.max_traces = max_traces
        self.h5_path = h5_path
        self.batch_size = batch_size

        # Compute validation steps
        self.val_steps = min(50, max(1, len(val_event_groups) // batch_size))

        # History for stagnation detection
        self.attention_acc_history = []
        self.stagnant_epochs = 0
        self.best_attention_acc = 0.0

        print(f"[EventAggregationMonitor] Initialized:")
        print(f"  - Validation events: {len(val_event_groups)}")
        print(f"  - Validation steps: {self.val_steps}")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        print(f"\n--- Epoch {epoch + 1}: Starting aggregation method comparison ---")

        # ===== Key fix 1: Rebuild dataset each epoch to avoid iterator exhaustion =====
        # Use fresh generator instance to ensure fresh data
        val_dataset = build_event_tf_dataset(
            self.val_event_groups,
            self.h5_path,
            self.scaler,
            shuffle=False,  # Maintain reproducibility
            batch_size=self.batch_size,
            trace_prob_cache=self.val_trace_prob_cache,
            is_training=False,
            mode='inference',  # Explicitly use inference mode
            max_traces=self.max_traces
        )

        y_true_event = []
        attention_probs = []
        average_probs = []
        majority_vote_probs = []
        max_confidence_probs = []
        quality_weighted_probs = []
        top_k_probs = []

        # Counters for debugging
        batch_count = 0
        total_samples = 0

        # Iterate over newly built dataset
        for (x1, x2, x3, mask_input), (y_event, y_trace) in val_dataset.take(self.val_steps):
            batch_count += 1

            # ===== Key fix 2: Ensure prediction uses current model state =====
            # Use training=False to ensure BN layers use moving averages
            preds = self.model.predict([x1, x2, x3, mask_input], verbose=0, batch_size=self.batch_size)
            event_pred = preds[0].flatten()  # Event probability [batch]
            trace_pred = preds[1]  # Trace-level probability [batch, traces]

            batch_size_actual = y_event.shape[0]
            total_samples += batch_size_actual

            for b in range(batch_size_actual):
                # Process labels
                y_true_val = y_event[b].numpy() if hasattr(y_event[b], 'numpy') else y_event[b]
                y_true_event.append(int(y_true_val))

                # Get number of valid traces
                y_trace_b = y_trace[b]
                if hasattr(y_trace_b, 'numpy'):
                    y_trace_b = y_trace_b.numpy()

                valid_mask = y_trace_b != -1
                n_real = np.sum(valid_mask)

                if n_real == 0:
                    # No valid trace, fill defaults
                    attention_probs.append(0.5)
                    average_probs.append(0.5)
                    majority_vote_probs.append(0.5)
                    max_confidence_probs.append(0.5)
                    quality_weighted_probs.append(0.5)
                    top_k_probs.append(0.5)
                    continue

                # Get valid predictions and features for this event
                probs = trace_pred[b, :n_real]  # [n_real]

                # Process features
                x3_b = x3[b, :n_real, :]
                if hasattr(x3_b, 'numpy'):
                    feats = x3_b.numpy()
                else:
                    feats = x3_b

                # ===== Aggregation method 1: attention weights (using model's event probability) =====
                # This is the optimal aggregation learned by the model
                attention_probs.append(float(event_pred[b]))

                # ===== Aggregation method 2: simple average =====
                average_probs.append(float(np.mean(probs)))

                # ===== Aggregation method 3: majority voting =====
                votes = (probs > 0.5).astype(int)
                majority_vote_probs.append(float(np.mean(votes)))

                # ===== Aggregation method 4: maximum confidence =====
                confidences = np.abs(probs - 0.5)
                best_idx = np.argmax(confidences)
                max_confidence_probs.append(float(probs[best_idx]))

                # ===== Aggregation method 5: quality weighted =====
                # Use quality score from features (second to last dimension)
                quality_scores = feats[:, -2]  # [n_real]
                valid_q = quality_scores > -0.99

                if np.sum(valid_q) == 0:
                    # No valid quality score, fallback to simple average
                    weights = np.ones(len(probs)) / len(probs)
                    weighted_prob = np.sum(probs * weights)
                else:
                    # Use sigmoid weighting
                    weights = 1.0 / (1.0 + np.exp(-quality_scores[valid_q] * 2))
                    weights = weights / (np.sum(weights) + 1e-10)
                    weighted_prob = np.sum(probs[valid_q] * weights)

                quality_weighted_probs.append(float(weighted_prob))

                # ===== Aggregation method 6: Top-K average =====
                k = max(1, n_real // 2)
                top_k_idx = np.argsort(confidences)[-k:]
                top_k_probs.append(float(np.mean(probs[top_k_idx])))

        # Debug info
        print(f"  Processed {batch_count} batches, {total_samples} samples")

        # Check for data
        if len(y_true_event) == 0:
            print("  Warning: no validation data retrieved, skipping aggregation evaluation")
            return

        y_true = np.array(y_true_event)

        # Compute accuracy for all aggregation methods
        methods = {
            'attention': attention_probs,
            'average': average_probs,
            'majority_vote': majority_vote_probs,
            'max_confidence': max_confidence_probs,
            'quality_weighted': quality_weighted_probs,
            'top_k_average': top_k_probs,
        }

        current_attention_acc = 0.0

        for name, probs in methods.items():
            if len(probs) == 0:
                print(f"  Aggregation method {name:15s}: no data")
                continue

            y_pred = (np.array(probs) > 0.5).astype(int)
            acc = accuracy_score(y_true, y_pred)
            logs[f'val_agg_{name}_acc'] = acc

            # Record attention accuracy for stagnation detection
            if name == 'attention':
                current_attention_acc = acc

            print(f"  Aggregation method {name:15s}: accuracy = {acc:.4f} ({len(probs)} samples)")

        # ===== Key fix 3: Detect stagnation and provide suggestions =====
        self.attention_acc_history.append(current_attention_acc)

        # Detect changes over last 3 epochs
        if len(self.attention_acc_history) >= 3:
            recent_accs = self.attention_acc_history[-3:]
            acc_range = max(recent_accs) - min(recent_accs)

            if acc_range < 0.001:  # Change less than 0.1%
                self.stagnant_epochs += 1
                print(f"  Stagnation detected (consecutive {self.stagnant_epochs} epochs change < {acc_range:.4f})")

        # Record best performance
        if current_attention_acc > self.best_attention_acc:
            self.best_attention_acc = current_attention_acc
            print(f"  New best attention accuracy: {self.best_attention_acc:.4f}")

        print("--- Aggregation method comparison ended ---\n")


# New: Callback to unfreeze encoder (independently controlled)
class UnfreezeEncoderCallback(keras.callbacks.Callback):
    def __init__(self, unfreeze_epoch=5, lr=1e-5):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.lr = lr
        self.unfrozen = False

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch and not self.unfrozen:
            print(f"\n[Epoch {epoch + 1}] Unfreezing encoder...")

            # Unfreeze all encoder layers
            for layer in self.model.layers:
                if any(prefix in layer.name for prefix in ['shared_wave_', 'shared_spec_', 'shared_feat_']):
                    layer.trainable = True
                    print(f"  Unfrozen layer: {layer.name}")

            # Recompile model (at epoch start, ensure training function rebuilt)
            optimizer = build_safe_optimizer(self.lr)
            self.model.compile(
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
                    'trace_classifier': [MaskedAccuracy(mask_value=-1, name='trace_accuracy')]
                }
            )
            print(f"  Model recompiled, learning rate adjusted to: {self.lr:.2e}")
            self.unfrozen = True

    def on_train_batch_begin(self, batch, logs=None):
        # Ensure training function properly initialized after recompile
        if self.unfrozen and batch == 0:
            print("  Reinitializing training function...")

class FixTotalLossCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        ev_loss = logs.get('event_output_loss')
        tr_loss = logs.get('trace_classifier_loss')

        if (ev_loss is not None and tr_loss is not None and
                np.isfinite(ev_loss) and np.isfinite(tr_loss)):
            logs['loss'] = ev_loss + INTERMEDIATE_LOSS_WEIGHT * tr_loss
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f" - FixTotalLossCallback: corrected total loss = {logs['loss']:.4f}")

class MemoryCleaner(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        print_memory_usage("After training ")


class TracePerformanceLogger(Callback):
    """Fixed Trace performance logger - correctly compute precision/recall, handle class imbalance"""

    def __init__(self, val_dataset, val_steps):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_steps = val_steps

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred_prob = []

        try:
            for (x1, x2, x3, mask_input), (y_event, y_trace) in self.val_dataset.take(self.val_steps):
                preds = self.model.predict([x1, x2, x3, mask_input], verbose=0)

                # Fixed: correctly unpack output, handle 2D trace_pred
                if isinstance(preds, list) and len(preds) >= 2:
                    trace_pred = preds[1]  # now [batch, traces]
                else:
                    trace_pred = preds

                batch_size = y_event.shape[0]
                for b in range(batch_size):
                    real_mask = y_trace[b] != -1
                    n_real = tf.reduce_sum(tf.cast(real_mask, tf.int32)).numpy()

                    if n_real == 0:
                        continue

                    # Fixed: trace_pred is now 2D [batch, traces]
                    trace_pred_real = trace_pred[b, :n_real]
                    y_trace_real = y_trace[b, :n_real]

                    y_true.extend(y_trace_real.numpy().tolist())
                    y_pred_prob.extend(trace_pred_real.flatten().tolist())

            if len(y_true) == 0:
                print("Warning: no valid trace samples for metric calculation")
                logs['val_trace_accuracy'] = 0.0
                logs['val_trace_precision'] = 0.0
                logs['val_trace_recall'] = 0.0
                return

            y_pred = (np.array(y_pred_prob) > 0.5).astype(int)
            y_true_np = np.array(y_true)

            # Fixed: use sklearn metric functions, properly handle class imbalance
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            accuracy = accuracy_score(y_true_np, y_pred)
            # Use macro average to handle class imbalance
            precision = precision_score(y_true_np, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true_np, y_pred, average='macro', zero_division=0)

            if logs is not None:
                logs['val_trace_accuracy'] = accuracy
                logs['val_trace_precision'] = precision
                logs['val_trace_recall'] = recall
                print(f" - Validation trace accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")

                # Output class distribution for diagnosis
                unique, counts = np.unique(y_true_np, return_counts=True)
                dist = dict(zip(unique, counts))
                print(f" - Validation trace label distribution: {dist}")

        except Exception as e:
            print(f"Trace performance logger error: {e}")
            if logs is not None:
                logs['val_trace_accuracy'] = 0.0
                logs['val_trace_precision'] = 0.0
                logs['val_trace_recall'] = 0.0


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
        self.max_traces = None

        self.shared_wave_encoder = None
        self.shared_spec_encoder = None
        self.shared_feat_encoder = None

    def validate_shapes(self, dataset, steps=1):
        print("Validating dataset shapes:")
        try:
            for i, (inputs, outputs) in enumerate(dataset.take(steps)):
                x1, x2, x3, mask_input = inputs
                y_event, y_trace = outputs

                print(f"Batch {i + 1}:")
                print(f"  Input shapes: wave={x1.shape}, spec={x2.shape}, feat={x3.shape}, mask={mask_input.shape}")
                print(f"  Output shapes: event_label={y_event.shape}, trace_label={y_trace.shape}")

                if self.model is not None:
                    try:
                        pred_event, pred_trace = self.model.predict([x1, x2, x3, mask_input], verbose=0)
                        print(f"  Model output shapes: event_pred={pred_event.shape}, trace_pred={pred_trace.shape}")
                        loss = self.model.test_on_batch([x1, x2, x3, mask_input], [y_event, y_trace])
                        print(f"  Test loss: {loss}")
                    except Exception as e:
                        print(f"  Model prediction failed: {e}")
                print("-" * 50)
        except Exception as e:
            print(f"Shape validation failed: {e}")

    def build_trace_model(self):
        if self.shared_wave_encoder is None:
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
            wave_out = layers.Dense(32, activation="relu", name="wave_embed")(x)
            self.shared_wave_encoder = keras.Model(waveform_input, wave_out, name="shared_wave_encoder")

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
            spec_out = layers.Dense(32, activation="relu", name="spec_embed")(y)
            self.shared_spec_encoder = keras.Model(spectrogram_input, spec_out, name="shared_spec_encoder")

            features_input = layers.Input(shape=(8,), name="features_input")
            z = layers.Dense(32, activation="relu")(features_input)
            z = layers.BatchNormalization()(z)
            feat_out = layers.Dense(32, activation="relu", name="feat_embed")(z)
            self.shared_feat_encoder = keras.Model(features_input, feat_out, name="shared_feat_encoder")

        waveform_input = layers.Input(shape=(WAVEFORM_LENGTH, 1), name="waveform_input")
        spectrogram_input = layers.Input(shape=(SPEC_HEIGHT, SPEC_WIDTH, 1), name="spectrogram_input")
        features_input = layers.Input(shape=(8,), name="features_input")

        wave_emb = self.shared_wave_encoder(waveform_input)
        spec_emb = self.shared_spec_encoder(spectrogram_input)
        feat_emb = self.shared_feat_encoder(features_input)

        combined = layers.concatenate([wave_emb, spec_emb, feat_emb])
        combined = layers.Dense(64, activation="relu",
                                kernel_regularizer=keras.regularizers.l2(0.001))(combined)  # L2 regularization
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.5)(combined)
        # Additional regularization layers
        combined = layers.Dense(32, activation="relu",
                                kernel_regularizer=keras.regularizers.l2(0.001))(combined)
        combined = layers.Dropout(0.3)(combined)  # Extra dropout
        output = layers.Dense(1, activation="sigmoid", name="trace_output")(combined)

        self.trace_model = keras.Model(
            inputs=[waveform_input, spectrogram_input, features_input],
            outputs=output
        )

        optimizer = build_safe_optimizer(TRACE_LEARNING_RATE)
        self.trace_model.compile(
            optimizer=optimizer,
            loss=bulletproof_event_loss,
            metrics=["accuracy", "precision", "recall"]
        )
        return self.trace_model

    def precompute_trace_probs(self, metadata, h5_manager):
        print("Starting precomputation of trace probabilities...")
        probs = {}

        total_traces = len(metadata)
        success_count = 0
        error_count = 0
        invalid_trace_count = 0

        with h5_manager as h5_file:
            comp_order_cache = None

            for idx, (_, row) in enumerate(metadata.iterrows()):
                if idx % 1000 == 0:
                    print(
                        f"Precomputation progress: {idx}/{total_traces} (success: {success_count}, failed: {error_count}, invalid: {invalid_trace_count})")
                    if idx % 5000 == 0 and idx > 0:
                        gc.collect()
                        print_memory_usage(f"Precomputation progress {idx}")

                trace_name = row["trace_name"]

                try:
                    if comp_order_cache is None:
                        comp_order_cache = get_component_order(h5_file)
                    target_comp_idx = comp_order_cache.index(VALID_COMPONENTS[0])
                    hdf5_path, event_idx = parse_trace_name(trace_name)

                    if hdf5_path not in h5_file:
                        error_count += 1
                        if error_count <= 10:
                            print(f"Error: HDF5 path does not exist: {hdf5_path}, trace: {trace_name}")
                        continue

                    wave_group = h5_file[hdf5_path]
                    if event_idx < 0 or event_idx >= wave_group.shape[0]:
                        error_count += 1
                        if error_count <= 10:
                            print(
                                f"Error: event index out of range: {event_idx}, max index: {wave_group.shape[0] - 1}, trace: {trace_name}")
                        continue

                    raw_wave = wave_group[event_idx, target_comp_idx, :].copy()

                    if not is_valid_trace(raw_wave):
                        invalid_trace_count += 1
                        continue

                    start_dt = parse_time_str(row["trace_start_time"])
                    raw_sr = float(row["trace_sampling_rate_hz"])
                    p_arrival = calculate_arrival_time(start_dt, int(row["trace_P_arrival_sample"]), raw_sr)
                    s_arrival = calculate_arrival_time(start_dt, int(row["trace_S_arrival_sample"]), raw_sr)
                    processed_wave, log_pg_sg = preprocess_waveform(raw_wave, raw_sr, p_arrival, s_arrival, start_dt,
                                                                    is_training=False)

                    if len(processed_wave) != WAVEFORM_LENGTH:
                        invalid_trace_count += 1
                        continue

                    spec_data = calculate_spectrogram(processed_wave)
                    processed_wave = np.expand_dims(processed_wave, axis=-1)
                    spec_data = np.expand_dims(spec_data, axis=-1)

                    fractal_dim = calculate_fractal_dimension(processed_wave[:, 0])
                    epicentral_distance = haversine_distance(
                        row["source_latitude_deg"], row["source_longitude_deg"],
                        row["station_latitude_deg"], row["station_longitude_deg"])
                    time_features = extract_time_features(row["origin_time"])

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

                    if self.scaler is not None:
                        norm_part, dist, depth = split_feat_vector(feat_vector)
                        norm_part = self.scaler.transform(norm_part.reshape(1, -1)).flatten()
                        feat_vector = merge_feat_vector(norm_part, dist, depth)

                    wave_input = np.expand_dims(processed_wave, axis=0)
                    spec_input = np.expand_dims(spec_data, axis=0)
                    feat_input = np.expand_dims(feat_vector, axis=0)

                    pred_prob = self.trace_model.predict([wave_input, spec_input, feat_input], verbose=0)
                    pred_prob = float(np.clip(pred_prob[0, 0], 0, 1))

                    probs[trace_name] = pred_prob
                    success_count += 1

                    del processed_wave, spec_data, raw_wave

                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"Precomputation for trace {trace_name} failed: {str(e)}")
                    continue

        coverage = success_count / total_traces * 100
        print(f"\n=== Precomputation completed statistics ===")
        print(f"Total traces: {total_traces}")
        print(f"Successful predictions: {success_count} ({coverage:.2f}%)")
        print(f"Failed: {error_count}")
        print(f"Invalid waveforms: {invalid_trace_count}")
        print(f"Cache coverage: {len(probs)}/{total_traces} ({coverage:.2f}%)")

        if coverage < 98:
            print(f"Warning: cache coverage is low ({coverage:.2f}%), recommend checking data quality")

        return probs

    def build_event_model(self):
        """Event-level model: correctly handle mask input, fix trace_classifier output dimension, support progressive unfreezing"""

        # ========== Input definitions ==========
        event_wave_input = layers.Input(
            shape=(None, WAVEFORM_LENGTH, 1),
            name="event_waveform_input"
        )
        event_spec_input = layers.Input(
            shape=(None, SPEC_HEIGHT, SPEC_WIDTH, 1),
            name="event_spectrogram_input"
        )
        event_feat_input = layers.Input(
            shape=(None, 11),
            name="event_features_input"
        )
        mask_input = layers.Input(
            shape=(None,),
            name="mask_input",
            dtype=tf.float32
        )

        # Separate features: first 10 dims are features, last dim is mask (if present)
        feats_raw = event_feat_input[:, :, :10]
        effective_mask = mask_input

        # Slice for feature encoder (first 8 original features)
        feat_slice = feats_raw[:, :, :8]

        # ========== Apply encoders (add name markers for unfreezing identification) ==========
        wave_emb = layers.TimeDistributed(
            self.shared_wave_encoder,
            name='td_shared_wave_encoder'
        )(event_wave_input)

        spec_emb = layers.TimeDistributed(
            self.shared_spec_encoder,
            name='td_shared_spec_encoder'
        )(event_spec_input)

        feat_emb = layers.TimeDistributed(
            self.shared_feat_encoder,
            name='td_shared_feat_encoder'
        )(feat_slice)

        # ========== Feature fusion and attention mechanism ==========
        combined = layers.Concatenate()([wave_emb, spec_emb, feat_emb])
        masked_combined = layers.Masking(mask_value=0.0)(combined)

        # Quality-aware attention layer
        attention_weights = QualityAwareAttentionLayer(name="attention_weights")(
            [masked_combined, feats_raw]
        )

        # Mask expansion and normalization
        expand_layer = ExpandLastDimLayer(axis=-1, name="mask_expand")
        expanded_mask = expand_layer(effective_mask)

        masked_attention = layers.Multiply()([attention_weights, expanded_mask])

        normalize_layer = AttentionNormalizeLayer(name="attention_normalize")
        normalized_attention = normalize_layer(masked_attention)

        # Weighted aggregation
        weighted = layers.Multiply()([masked_combined, normalized_attention])
        event_emb = layers.GlobalAveragePooling1D()(weighted)

        # ========== Event-level output head ==========
        event_out = layers.Dense(64, activation='relu', name='event_dense_1')(event_emb)
        event_out = layers.Dropout(0.3, name='event_dropout')(event_out)
        # Add additional nonlinear layer to increase capacity (helps break saturation)
        event_out = layers.Dense(32, activation='relu', name='event_dense_2')(event_out)
        event_out = layers.Dense(1, activation='sigmoid', name='event_output')(event_out)

        # ========== Trace-level output head (fix dimension issue) ==========
        trace_dense = layers.TimeDistributed(
            layers.Dense(1, activation='sigmoid'),
            name='trace_dense'
        )(masked_combined)

        # Use SqueezeLastDimLayer to ensure correct output dimension
        trace_out = SqueezeLastDimLayer(name='trace_classifier')(trace_dense)

        # ========== Model construction ==========
        model = keras.Model(
            inputs=[event_wave_input, event_spec_input, event_feat_input, mask_input],
            outputs=[event_out, trace_out]
        )

        # ========== Freeze encoders (add name prefixes for later identification and unfreezing) ==========
        # Default freeze encoders, unfreezing timing controlled by callbacks
        print("[build_event_model] Freezing shared encoders, will unfreeze after specified epoch...")

        for layer in self.shared_wave_encoder.layers:
            layer.trainable = False
            # Add prefix for later identification and unfreezing
            if not layer.name.startswith('shared_wave_'):
                layer._name = f"shared_wave_{layer.name}"

        for layer in self.shared_spec_encoder.layers:
            layer.trainable = False
            if not layer.name.startswith('shared_spec_'):
                layer._name = f"shared_spec_{layer.name}"

        for layer in self.shared_feat_encoder.layers:
            layer.trainable = False
            if not layer.name.startswith('shared_feat_'):
                layer._name = f"shared_feat_{layer.name}"

        # Print trainable status summary
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

        print(f"[build_event_model] Model parameter statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")
        print(f"  Non-trainable: {non_trainable_params:,} ({non_trainable_params / total_params * 100:.1f}%)")

        # ========== Compile model ==========
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
                'trace_classifier': [MaskedAccuracy(mask_value=-1, name='trace_accuracy')]
            }
        )

        # ========== Auxiliary model: extract attention weights for visualization ==========
        self.attention_model = keras.Model(
            inputs=[event_wave_input, event_spec_input, event_feat_input],
            outputs=attention_weights
        )

        self.model = model
        return model

    def extract_trace_model_weights(self):
        if self.trace_model is None:
            return None
        wave_weights, spec_weights, feat_weights = {}, {}, {}
        return wave_weights, spec_weights, feat_weights

    def validate_model_output_shapes(self):
        print("Validating model output shapes:")
        dummy_batch_size = 2
        dummy_wave = tf.random.normal((dummy_batch_size, self.max_traces, WAVEFORM_LENGTH, 1))
        dummy_spec = tf.random.normal((dummy_batch_size, self.max_traces, SPEC_HEIGHT, SPEC_WIDTH, 1))
        dummy_feat = tf.random.normal((dummy_batch_size, self.max_traces, 11))
        dummy_mask = tf.ones((dummy_batch_size, self.max_traces), dtype=tf.float32)

        outputs = self.model([dummy_wave, dummy_spec, dummy_feat, dummy_mask])

        print(f"Event-level output shape: {outputs[0].shape}")
        print(f"Trace-level output shape: {outputs[1].shape}")

        dummy_event_labels = tf.constant([1, 0], dtype=tf.int8)
        dummy_trace_labels = tf.constant(
            [[1] * self.max_traces, [0] * self.max_traces],
            dtype=tf.int8
        )
        try:
            loss = self.model.test_on_batch(
                [dummy_wave, dummy_spec, dummy_feat, dummy_mask],
                [dummy_event_labels, dummy_trace_labels]
            )
            print(f"Test loss computation succeeded: {loss}")
        except Exception as e:
            print(f"Test loss computation failed: {e}")

    def _reconstruct_encoders_from_trace_model(self):
        """
        Extract the output tensors of the three encoders from the trace model and construct shared encoders.
        Assumes trace model structure: three inputs -> three encoders -> concatenation -> subsequent layers.
        """
        # Find the Concatenate layer in the trace model (layer that concatenates the three encoder outputs)
        concat_layer = None
        for layer in self.trace_model.layers:
            if isinstance(layer, layers.Concatenate):
                concat_layer = layer
                break
        if concat_layer is None:
            raise RuntimeError("Concatenate layer not found in trace model, cannot extract encoders")

        # The inputs to the Concatenate layer are three tensors, order should be [wave_emb, spec_emb, feat_emb]
        wave_emb_tensor, spec_emb_tensor, feat_emb_tensor = concat_layer.input

        # Get the three input tensors of the trace model
        wave_input = self.trace_model.input[0]  # waveform_input
        spec_input = self.trace_model.input[1]  # spectrogram_input
        feat_input = self.trace_model.input[2]  # features_input

        # Build new shared encoder models (directly use subgraphs from trace model, weights are automatically shared)
        self.shared_wave_encoder = keras.Model(inputs=wave_input, outputs=wave_emb_tensor,
                                               name="shared_wave_encoder")
        self.shared_spec_encoder = keras.Model(inputs=spec_input, outputs=spec_emb_tensor,
                                               name="shared_spec_encoder")
        self.shared_feat_encoder = keras.Model(inputs=feat_input, outputs=feat_emb_tensor,
                                               name="shared_feat_encoder")

        print("Successfully extracted encoders from trace model, weights are shared")

    def pretrain_trace_model(self, train_metadata, val_metadata):
        """
        Pretrain trace-level model - fixed validation set setup
        """
        print("Building single-trace training dataset...")
        train_dataset = build_trace_tf_dataset(
            metadata=train_metadata,
            h5_manager=self.h5_manager,
            scaler=self.scaler,
            shuffle=True,  # Shuffle during training
            batch_size=TRACE_BATCH_SIZE,
            is_training=True  # Key: mark as training mode
        )

        print("Building single-trace validation dataset...")
        val_dataset = build_trace_tf_dataset(
            metadata=val_metadata,
            h5_manager=self.h5_manager,
            scaler=self.scaler,
            shuffle=False,  # No shuffle during validation
            batch_size=TRACE_BATCH_SIZE,
            is_training=False  # Key: mark as validation mode
        )

        # Count actual data size (for step calculation)
        print("Counting actual available training data...")
        actual_train_samples = 0
        for _ in train_dataset:
            actual_train_samples += TRACE_BATCH_SIZE
            if actual_train_samples >= 100000:
                break

        print("Counting actual available validation data...")
        actual_val_samples = 0
        # Key fix: validation set does not repeat, can directly iterate and count
        for _ in val_dataset:
            actual_val_samples += TRACE_BATCH_SIZE
            if actual_val_samples >= 50000:
                break

        print(f"Actual available training samples: ~{actual_train_samples}, validation samples: ~{actual_val_samples}")

        train_steps = max(1, actual_train_samples // TRACE_BATCH_SIZE)
        val_steps = max(1, actual_val_samples // TRACE_BATCH_SIZE)

        print(f"Adjusted training steps: {train_steps}, validation steps: {val_steps}")

        if train_steps < 10:
            print(f"Warning: too few training data ({actual_train_samples} samples), recommend checking data quality")
            global TRACE_PRETRAIN_EPOCHS
            TRACE_PRETRAIN_EPOCHS = min(TRACE_PRETRAIN_EPOCHS, 5)

        print("Building single-trace model...")
        self.build_trace_model()

        if self.trace_model is None:
            raise RuntimeError("Single-trace model building failed")

        self.trace_model.summary()

        # Compute class weights
        y_train_labels = train_metadata["event_type"].apply(lambda x: 1 if x == "earthquake" else 0).values
        unique_classes = np.unique(y_train_labels)
        if len(unique_classes) < 2:
            print("Warning: training set contains only one event type")
            unique_classes = np.array([0, 1])

        try:
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
        except Exception as e:
            print(f"Class weight computation failed: {e}, using default weights")
            class_weight_dict = {0: 1.0, 1: 1.0}

        print(f"Single-trace model class weights: {class_weight_dict}")

        # Callbacks
        explosion_recall_cb = ExplosionRecallLogger(val_dataset=val_dataset, val_steps=val_steps, is_trace_model=True)
        memory_cleaner_cb = MemoryCleaner()
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, "trace_model_epoch_{epoch:02d}.keras"),
            save_freq='epoch',
            save_weights_only=False,
            verbose=1
        )
        lr_scheduler = keras.callbacks.LearningRateScheduler(
            create_trace_adaptive_scheduler(TRACE_LEARNING_RATE, TRACE_PRETRAIN_EPOCHS)
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',  # Key: monitor val_loss instead of val_accuracy
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            lr_scheduler,
            explosion_recall_cb,
            memory_cleaner_cb,
            checkpoint_cb
        ]

        print("Starting single-trace model pretraining...")
        print_memory_usage("Before pretraining ")

        with timing_context("Single-trace model pretraining"):
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

        print("Single-trace model pretraining completed")
        return history

    def fit_scaler(self, train_metadata, sample_size=5000):
        if self.h5_manager is None:
            raise RuntimeError("h5_manager not initialized, please set HDF5 file path first")

        feat_list = []
        total_samples = len(train_metadata)

        if sample_size > 0 and sample_size < total_samples:
            sampled_metadata = train_metadata.sample(n=sample_size, random_state=42)
            print(f"Sampling {sample_size} out of {total_samples} samples for scaler fitting")
        else:
            sampled_metadata = train_metadata
            sample_size = total_samples

        success_count = 0
        error_count = 0
        invalid_trace_count = 0
        detail_errors = []

        print(f"Starting feature extraction from actual data, number of samples: {len(sampled_metadata)}")

        with self.h5_manager as h5_file:
            comp_order_cache = None
            for idx, (_, row) in enumerate(sampled_metadata.iterrows()):
                if idx % 500 == 0:
                    print(f"Scaler progress: {idx}/{len(sampled_metadata)} "
                          f"(success: {success_count}, failed: {error_count}, invalid: {invalid_trace_count})")

                trace_name = row["trace_name"]
                try:
                    if comp_order_cache is None:
                        comp_order_cache = get_component_order(h5_file)
                    target_comp_idx = comp_order_cache.index(VALID_COMPONENTS[0])

                    try:
                        hdf5_path, event_idx = parse_trace_name(trace_name)
                    except ValueError as e:
                        error_count += 1
                        detail_errors.append(f"parse_trace_name failed: {trace_name} -> {e}")
                        continue

                    if hdf5_path not in h5_file:
                        error_count += 1
                        detail_errors.append(f"HDF5 path does not exist: {hdf5_path}")
                        continue

                    wave_group = h5_file[hdf5_path]
                    if event_idx < 0 or event_idx >= wave_group.shape[0]:
                        error_count += 1
                        detail_errors.append(f"Event index out of range: {event_idx} (max: {wave_group.shape[0] - 1})")
                        continue

                    raw_wave = wave_group[event_idx, target_comp_idx, :].copy()

                    if not is_valid_trace(raw_wave):
                        invalid_trace_count += 1
                        continue

                    try:
                        start_dt = parse_time_str(row["trace_start_time"])
                        raw_sr = float(row["trace_sampling_rate_hz"])
                        p_arrival = calculate_arrival_time(start_dt, int(row["trace_P_arrival_sample"]), raw_sr)
                        s_arrival = calculate_arrival_time(start_dt, int(row["trace_S_arrival_sample"]), raw_sr)
                        processed_wave, log_pg_sg = preprocess_waveform(
                            raw_wave, raw_sr, p_arrival, s_arrival, start_dt, is_training=False
                        )
                    except Exception as e:
                        error_count += 1
                        detail_errors.append(f"Waveform preprocessing failed: {trace_name} -> {e}")
                        continue

                    if len(processed_wave) != WAVEFORM_LENGTH:
                        invalid_trace_count += 1
                        continue

                    try:
                        fractal_dim = calculate_fractal_dimension(processed_wave)
                    except Exception as e:
                        fractal_dim = 1.0
                        detail_errors.append(f"Fractal dimension calculation failed, using default 1.0: {e}")

                    try:
                        mag = float(row["mag"])
                    except:
                        mag = 5.0
                        detail_errors.append(f"Magnitude conversion failed, using default 5.0")

                    try:
                        dt = parse_time_str(str(row["origin_time"]))
                        hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                        hour_cos = np.cos(2 * np.pi * dt.hour / 24)
                        weekday = 1 if dt.weekday() < 5 else 0
                    except Exception as e:
                        hour_sin = 0.0
                        hour_cos = 0.0
                        weekday = 0.0
                        detail_errors.append(f"Time feature extraction failed, using defaults: {e}")

                    norm_feat_vector = np.array([
                        fractal_dim,
                        mag,
                        log_pg_sg,
                        hour_sin,
                        hour_cos,
                        weekday
                    ], dtype=np.float32)

                    feat_list.append(norm_feat_vector)
                    success_count += 1

                    del processed_wave, raw_wave

                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        detail_errors.append(f"Unknown exception at row {idx}: {trace_name} -> {e}")
                    continue

        if detail_errors:
            print(f"\nDetailed error log (first 20 entries):")
            for err in detail_errors[:20]:
                print(f"  - {err}")

        print(f"\nScaler fitting completed statistics:")
        print(f"Total samples: {len(sampled_metadata)}")
        print(f"Successfully processed: {success_count} ({success_count / len(sampled_metadata) * 100:.2f}%)")
        print(f"Failed: {error_count}")
        print(f"Invalid waveforms: {invalid_trace_count}")

        if len(feat_list) == 0:
            raise ValueError("No valid features collected, cannot train scaler")

        feat_array = np.array(feat_list)
        print(f"\nSuccessfully collected {len(feat_list)} features, dimension: {feat_array.shape}")

        print("\nFeature statistics (before fitting):")
        feature_names = ["fractal_dim", "mag", "log_pg_sg", "hour_sin", "hour_cos", "weekday"]
        for i, name in enumerate(feature_names):
            if feat_array.shape[0] > 0:
                print(f"  {name}: mean={feat_array[:, i].mean():.4f}, std={feat_array[:, i].std():.4f}, "
                      f"min={feat_array[:, i].min():.4f}, max={feat_array[:, i].max():.4f}")

        print("\nStarting scaler fitting...")
        self.scaler.fit(feat_array)
        print("Scaler fitting completed!")

        test_sample = feat_array[0]
        transformed = self.scaler.transform([test_sample])
        print(f"\nScaler test:")
        print(f"  Original sample: {test_sample}")
        print(f"  Transformed: {transformed[0]}")
        print(f"  Mean: {self.scaler.mean_}")
        print(f"  Scale: {self.scaler.scale_}")

        return self.scaler

    def evaluate_test_set(self, test_name, test_metadata):
        try:
            test_event_groups = group_metadata_by_event(test_metadata)

            total_events = len(test_event_groups)
            earthquake_events = sum(1 for event_type, _ in test_event_groups if event_type == "earthquake")
            explosion_events = sum(1 for event_type, _ in test_event_groups if event_type == "explosion")
            single_trace_events = sum(1 for _, traces in test_event_groups if len(traces) == 1)
            multi_trace_events = total_events - single_trace_events

            print(f"{test_name} contains {total_events} events")

            all_traces = pd.concat([tr for _, tr in test_event_groups], ignore_index=True)
            if self.trace_prob_cache is None or len(self.trace_prob_cache) == 0:
                print("Trace probability cache is empty, precomputing...")
                self.trace_prob_cache = self.precompute_trace_probs(all_traces, self.h5_manager)
            trace_prob_cache = self.trace_prob_cache

            test_dataset = build_event_tf_dataset(
                event_groups=test_event_groups,
                h5_path=WAVEFORM_PATH,
                scaler=self.scaler,
                shuffle=False,
                batch_size=BATCH_SIZE,
                trace_prob_cache=trace_prob_cache,
                is_training=False,
                max_traces=self.max_traces
            )

            test_steps = max(1, len(test_event_groups) // BATCH_SIZE)

            y_true, y_pred_prob = [], []
            trace_pred_flat, trace_true_flat = [], []

            for (x1, x2, x3, mask_input), (y_event, y_trace) in test_dataset.take(test_steps):
                preds = self.model.predict([x1, x2, x3, mask_input], verbose=0)
                event_pred = preds[0].flatten()
                trace_pred = preds[1]  # [batch, traces] numpy array

                batch_y_event = np.asarray(y_event).ravel().tolist()
                y_true.extend(batch_y_event)
                y_pred_prob.extend(event_pred.tolist())

                batch_size = y_event.shape[0]
                for b in range(batch_size):
                    n_real = tf.reduce_sum(tf.cast(y_trace[b] != -1, tf.int32)).numpy()
                    if n_real == 0:
                        continue

                    # Key fix: trace_pred is a numpy array, use directly, do not call .numpy()
                    trace_pred_real = trace_pred[b, :n_real]
                    y_trace_real = y_trace[b, :n_real]

                    # Fix: y_trace_real may be a tensor, need .numpy()
                    if hasattr(y_trace_real, 'numpy'):
                        y_trace_real = y_trace_real.numpy()

                    trace_pred_flat.extend(trace_pred_real.tolist())
                    trace_true_flat.extend(y_trace_real.tolist())

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            flat_y_true = [int(item) for item in y_true]
            y_pred = (np.array(y_pred_prob) > 0.5).astype(int)

            event_accuracy = accuracy_score(flat_y_true, y_pred)
            event_precision = precision_score(flat_y_true, y_pred, zero_division=0)
            event_recall = recall_score(flat_y_true, y_pred, zero_division=0)
            event_f1 = f1_score(flat_y_true, y_pred, zero_division=0)

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

            trace_cm = None
            if len(trace_pred_flat) > 0:
                trace_pred_binary = (np.array(trace_pred_flat) > 0.5).astype(int)
                trace_accuracy = accuracy_score(trace_true_flat, trace_pred_binary)
                trace_precision = precision_score(trace_true_flat, trace_pred_binary, zero_division=0)
                trace_recall = recall_score(trace_true_flat, trace_pred_binary, zero_division=0)
                trace_f1 = f1_score(trace_true_flat, trace_pred_binary, zero_division=0)

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

            output_lines = [
                f"\n[{test_name} - Detailed Test Results]",
                "=" * 80,
                "\n[Dataset Statistics]",
                f"- Total original events: {len(test_metadata['event_id'].unique())}",
                f"- Final event count (including single-trace): {total_events}",
                f"- Event utilization: {total_events / len(test_metadata['event_id'].unique()) * 100:.1f}%",
                f"- Earthquake events: {earthquake_events}, Explosion events: {explosion_events}",
                f"- Single-trace events: {single_trace_events} ({single_trace_events / total_events * 100:.1f}%)",
                f"- Multi-trace events: {multi_trace_events} ({multi_trace_events / total_events * 100:.1f}%)",
                "\n[Event-Level Performance Metrics]",
                f"- Accuracy: {event_accuracy:.4f}",
                f"- Precision: {event_precision:.4f}",
                f"- Recall: {event_recall:.4f}",
                f"- F1 Score: {event_f1:.4f}",
                f"- Earthquake Recall: {earthquake_recall:.4f}",
                f"- Explosion Recall: {explosion_recall:.4f}",
                f"- Earthquake Precision: {earthquake_precision:.4f}",
                f"- Explosion Precision: {explosion_precision:.4f}",
                f"- Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
                "\n[Trace-Level Performance Metrics]",
                f"- Accuracy: {trace_accuracy:.4f}",
                f"- Precision: {trace_precision:.4f}",
                f"- Recall: {trace_recall:.4f}",
                f"- F1 Score: {trace_f1:.4f}",
                f"- Earthquake Recall: {trace_earthquake_recall:.4f}",
                f"- Explosion Recall: {trace_explosion_recall:.4f}",
                f"- Total Traces: {len(trace_pred_flat)}",
                f"- Confusion Matrix: TN={t_tn}, FP={t_fp}, FN={t_fn}, TP={t_tp}",
                "\n" + "=" * 80 + "\n"
            ]

            out_str = "\n".join(output_lines)
            print(out_str)
            with open(RESULT_OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(out_str)

            self._plot_roc_pr_curves(flat_y_true, y_pred_prob, test_name)

            fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Explosion', 'Earthquake'],
                        yticklabels=['Explosion', 'Earthquake'],
                        ax=axes[0], cbar_kws={'shrink': 0.8},
                        linewidths=0.5, linecolor='white',
                        annot_kws={'size': 9})
            axes[0].set_title('(a) Event-Level Confusion Matrix')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')
            if trace_cm is not None:
                sns.heatmap(trace_cm, annot=True, fmt='d', cmap='Oranges',
                            xticklabels=['Explosion', 'Earthquake'],
                            yticklabels=['Explosion', 'Earthquake'],
                            ax=axes[1], cbar_kws={'shrink': 0.8},
                            linewidths=0.5, linecolor='white',
                            annot_kws={'size': 9})
                axes[1].set_title('(b) Trace-Level Confusion Matrix')
                axes[1].set_ylabel('True Label')
                axes[1].set_xlabel('Predicted Label')
            else:
                axes[1].text(0.5, 0.5, 'No Trace Data', ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('(b) Trace-Level (No Data)')
            plt.tight_layout(pad=0.5)
            gji_path = CONFUSION_MATRIX_PATH.replace('.png', f'_{test_name.replace(" ", "_")}_GJI.png')
            plt.savefig(gji_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"GJI standard confusion matrix saved: {gji_path}")

        except Exception as e:
            import traceback
            err_msg = f"Loading {test_name} failed: {str(e)}\n" + traceback.format_exc()
            error_logger.error(err_msg)
            print(err_msg)
            with open(RESULT_OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(f"Error: {err_msg}\n\n" + "=" * 80 + "\n\n")

    def _plot_roc_pr_curves(self, y_true, y_pred_prob, test_name):
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            roc_auc = auc(fpr, tpr)

            ax1.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})', linewidth=2)
            ax1.plot([0, 1], [0, 1], 'k--', label='Random')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title(f'{test_name} - ROC Curve')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)

            precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
            pr_auc = auc(recall, precision)

            ax2.plot(recall, precision, label=f'PR (AUC={pr_auc:.3f})', linewidth=2)
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title(f'{test_name} - Precision-Recall Curve')
            ax2.legend(loc='lower left')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = CONFUSION_MATRIX_PATH.replace('.png', f'_{test_name.replace(" ", "_")}_ROC_PR.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"ROC/PR curves saved: {save_path}")
        except Exception as e:
            print(f"Failed to plot ROC/PR curves: {e}")

    class FixTotalLossCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                return
            ev_loss = logs.get('event_output_loss')
            tr_loss = logs.get('trace_classifier_loss')
            if ev_loss is not None and tr_loss is not None:
                logs['loss'] = ev_loss + INTERMEDIATE_LOSS_WEIGHT * tr_loss

    def train(self, train_path, val_path, test_sets, waveform_path, skip_training=False):
        """
        Main training pipeline - phased training version (with min val_loss early stopping mechanism)
        Phase 1: Train with frozen encoder for 5 epochs
        Phase 2: Unfreeze encoder, lower learning rate and continue training for remaining epochs
        """
        self.h5_manager = H5FileManager(waveform_path)
        trace_history = None
        event_history = None

        try:
            print("Checking HDF5 file integrity...")
            check_h5_integrity(waveform_path)

            print("Loading training metadata...")
            train_metadata = load_metadata_from_split(train_path)
            val_metadata = load_metadata_from_split(val_path)
            train_event_groups = group_metadata_by_event(train_metadata)
            val_event_groups = group_metadata_by_event(val_metadata)

            trace_counts = [len(traces) for _, traces in train_event_groups]
            self.max_traces = int(np.percentile(trace_counts, 98))
            self.max_traces = max(1, self.max_traces)
            print(f"Initial max_traces = {self.max_traces}")

            if not skip_training:
                # ========== Phase 0: Prepare encoders and data ==========
                trace_checkpoint_dir = CHECKPOINT_DIR

                # Priority: check if saved encoder files exist
                encoder_files_exist = (os.path.exists(os.path.join(trace_checkpoint_dir, "wave_encoder.keras")) and
                                       os.path.exists(os.path.join(trace_checkpoint_dir, "spec_encoder.keras")) and
                                       os.path.exists(os.path.join(trace_checkpoint_dir, "feat_encoder.keras")))

                if encoder_files_exist:
                    print("Loading saved encoder files from checkpoint...")
                    self.shared_wave_encoder = keras.models.load_model(
                        os.path.join(trace_checkpoint_dir, "wave_encoder.keras"),
                        custom_objects=CUSTOM_OBJECTS
                    )
                    self.shared_spec_encoder = keras.models.load_model(
                        os.path.join(trace_checkpoint_dir, "spec_encoder.keras"),
                        custom_objects=CUSTOM_OBJECTS
                    )
                    self.shared_feat_encoder = keras.models.load_model(
                        os.path.join(trace_checkpoint_dir, "feat_encoder.keras"),
                        custom_objects=CUSTOM_OBJECTS
                    )

                    # Load scaler
                    scaler_path = SAVE_MODEL_PATH.replace(".keras", "_scaler.joblib")
                    if os.path.exists(scaler_path):
                        self.scaler = joblib.load(scaler_path)
                        print("Scaler loaded successfully from main path")
                    else:
                        print("Scaler file not found, will refit")
                        self.fit_scaler(train_metadata)

                    trace_counts = [len(traces) for _, traces in train_event_groups]
                    self.max_traces = int(np.percentile(trace_counts, 98))
                    self.max_traces = max(1, self.max_traces)
                    print(f"Recalculated max_traces = {self.max_traces}")

                    trace_history = None
                    print("Encoders loaded successfully, skipping trace pretraining")

                elif os.path.exists(trace_checkpoint_dir) and any(f.startswith('trace_model_') and f.endswith('.keras')
                                                                  for f in os.listdir(trace_checkpoint_dir)):
                    # Fallback: extract encoders from latest trace model
                    trace_checkpoints = [f for f in os.listdir(trace_checkpoint_dir)
                                         if f.startswith('trace_model_') and f.endswith('.keras')]
                    latest_trace = sorted(trace_checkpoints,
                                          key=lambda x: os.path.getmtime(os.path.join(trace_checkpoint_dir, x)),
                                          reverse=True)[0]
                    trace_model_path = os.path.join(trace_checkpoint_dir, latest_trace)
                    print(f"Loading trace model from checkpoint: {trace_model_path}")

                    self.trace_model = keras.models.load_model(trace_model_path, custom_objects=CUSTOM_OBJECTS,
                                                               compile=False)
                    self._reconstruct_encoders_from_trace_model()

                    # Load scaler
                    scaler_path = SAVE_MODEL_PATH.replace(".keras", "_scaler.joblib")
                    if os.path.exists(scaler_path):
                        self.scaler = joblib.load(scaler_path)
                        print("Scaler loaded successfully from main path")
                    else:
                        print("Scaler file not found, will refit")
                        self.fit_scaler(train_metadata)

                    trace_counts = [len(traces) for _, traces in train_event_groups]
                    self.max_traces = int(np.percentile(trace_counts, 98))
                    self.max_traces = max(1, self.max_traces)
                    print(f"Recalculated max_traces = {self.max_traces}")

                    trace_history = None
                    print("Encoders extracted from trace model successfully, skipping trace pretraining")

                else:
                    # No checkpoint, perform full training
                    print("No checkpoint trace model or encoders found, performing full training pipeline...")
                    print("Fitting feature scaler...")
                    try:
                        self.fit_scaler(train_metadata)
                    except Exception as e:
                        print(f"Warning: Scaler fitting failed: {e}")
                        print("Continuing training with default scaler...")
                        self.scaler = StandardScaler()
                        default_data = np.random.randn(100, 6)
                        self.scaler.fit(default_data)

                    print("Starting pretraining of single-trace model...")
                    trace_history = self.pretrain_trace_model(train_metadata, val_metadata)

                # ========== Common data preparation ==========
                print("Precomputing training set trace probabilities...")
                train_trace_prob_cache = self.precompute_trace_probs(train_metadata, self.h5_manager)
                print("Precomputing validation set trace probabilities...")
                val_trace_prob_cache = self.precompute_trace_probs(val_metadata, self.h5_manager)

                # Filter events that have precomputed probabilities
                def filter_event_groups(event_groups, cache):
                    filtered = []
                    for ev_type, traces in event_groups:
                        if any(tname in cache for tname in traces['trace_name']):
                            filtered.append((ev_type, traces))
                    return filtered

                train_event_groups = filter_event_groups(train_event_groups, train_trace_prob_cache)
                val_event_groups = filter_event_groups(val_event_groups, val_trace_prob_cache)

                print(f"Filtered training events: {len(train_event_groups)}")
                print(f"Filtered validation events: {len(val_event_groups)}")

                if len(train_event_groups) == 0:
                    raise RuntimeError("No valid events in training set, please check data quality or precomputation success rate!")
                if len(val_event_groups) == 0:
                    print("Warning: Validation set is empty, validation metrics cannot be computed")

                # Recalculate max_traces
                trace_counts = [len(traces) for _, traces in train_event_groups]
                self.max_traces = int(np.percentile(trace_counts, 98))
                self.max_traces = max(1, self.max_traces)
                print(f"Final max_traces = {self.max_traces}")

                # Compute steps
                train_steps = max(1, len(train_event_groups) // BATCH_SIZE)
                val_steps = max(1, len(val_event_groups) // BATCH_SIZE) if len(val_event_groups) > 0 else 1
                print(f"Training steps: {train_steps}, Validation steps: {val_steps}")

                # ========== Phase 1: Train with frozen encoder (5 epochs) ==========
                print("\n" + "=" * 60)
                print("Phase 1: Frozen encoder training (epochs 1-5)")
                print("=" * 60)

                # Build event-level model (encoders frozen by default)
                print("Building event-level model (encoders frozen)...")
                self.build_event_model()

                # Validate model
                if self.model is None:
                    raise RuntimeError("Event-level model building failed, cannot train")

                self.model.summary()
                print(f"Phase 1 learning rate: {LEARNING_RATE:.2e}")

                # Build datasets
                print("Building training dataset (Phase 1)...")
                train_dataset_phase1 = build_event_tf_dataset(
                    train_event_groups,
                    waveform_path,
                    self.scaler,
                    shuffle=True,
                    batch_size=BATCH_SIZE,
                    trace_prob_cache=train_trace_prob_cache,
                    is_training=True,
                    max_traces=self.max_traces
                )

                print("Building validation dataset (Phase 1)...")
                val_dataset_phase1 = build_event_tf_dataset(
                    val_event_groups,
                    waveform_path,
                    self.scaler,
                    shuffle=False,
                    batch_size=BATCH_SIZE,
                    trace_prob_cache=val_trace_prob_cache,
                    is_training=False,
                    max_traces=self.max_traces
                )

                # Phase 1 callbacks - add min val_loss early stopping
                agg_monitor = EventAggregationMonitor(
                    val_event_groups=val_event_groups,
                    val_trace_prob_cache=val_trace_prob_cache,
                    scaler=self.scaler,
                    max_traces=self.max_traces,
                    h5_path=waveform_path,
                    batch_size=BATCH_SIZE
                )

                callbacks_phase1 = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',  # Monitor validation loss
                        min_delta=1e-4,  # Minimum improvement threshold
                        patience=10,  # Tolerate 10 epochs without improvement
                        verbose=1,
                        mode='min',  # Minimization mode
                        restore_best_weights=True,  # Restore best weights
                        start_from_epoch=2  # Start monitoring from epoch 3
                    ),
                    keras.callbacks.ModelCheckpoint(
                        os.path.join(CHECKPOINT_DIR, "event_model_phase1_best.keras"),
                        monitor='val_loss',
                        save_best_only=True,
                        mode='min',
                        verbose=1
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-7,
                        mode='min',
                        verbose=1
                    ),
                    keras.callbacks.LearningRateScheduler(
                        create_cosine_scheduler(LEARNING_RATE, 5)
                    ),
                    ExplosionRecallLogger(val_dataset=val_dataset_phase1, val_steps=val_steps),
                    TracePerformanceLogger(val_dataset=val_dataset_phase1, val_steps=val_steps),
                    MemoryCleaner(),
                    FixTotalLossCallback(),
                    agg_monitor,
                ]

                print("Starting Phase 1 training...")
                print(f"Early stopping settings: monitor='val_loss', mode='min', patience=10, min_delta=1e-4")
                print_memory_usage("Before Phase 1 training ")

                with timing_context("Phase 1 training"):
                    history_phase1 = self.model.fit(
                        train_dataset_phase1,
                        epochs=5,  # Only train 5 epochs
                        steps_per_epoch=train_steps,
                        validation_data=val_dataset_phase1,
                        validation_steps=val_steps,
                        callbacks=callbacks_phase1,
                        verbose=1
                    )

                print("Phase 1 training completed")
                best_val_loss_phase1 = min(history_phase1.history.get('val_loss', [float('inf')]))
                print(f"Phase 1 best validation loss: {best_val_loss_phase1:.4f}")
                print(f"Phase 1 final validation loss: {history_phase1.history['val_loss'][-1]:.4f}")
                print(f"Phase 1 final validation accuracy: {history_phase1.history['val_event_output_accuracy'][-1]:.4f}")

                # ========== Phase 2: Unfreeze encoder, continue training ==========
                print("\n" + "=" * 60)
                print("Phase 2: Unfreeze encoder and lower learning rate, continue training")
                print("=" * 60)

                # Unfreeze all encoder layers
                print("Unfreezing encoder layers...")
                unfrozen_layers = []
                for layer in self.model.layers:
                    # Match TimeDistributed wrapped encoder layers
                    if hasattr(layer, 'layer') and layer.layer.name.startswith('shared_'):
                        layer.layer.trainable = True
                        unfrozen_layers.append(f"{layer.name} -> {layer.layer.name}")
                    # Match directly referenced encoder layers
                    elif layer.name.startswith('shared_'):
                        layer.trainable = True
                        unfrozen_layers.append(layer.name)

                print(f"Unfrozen {len(unfrozen_layers)} layers:")
                for name in unfrozen_layers[:10]:
                    print(f"  - {name}")
                if len(unfrozen_layers) > 10:
                    print(f"  ... and {len(unfrozen_layers) - 10} more layers")

                # Recompile model
                phase2_lr = LEARNING_RATE * 0.1
                print(f"\nRecompiling model, learning rate: {phase2_lr:.2e}")
                optimizer = build_safe_optimizer(phase2_lr)

                self.model.compile(
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
                        'trace_classifier': [MaskedAccuracy(mask_value=-1, name='trace_accuracy')]
                    }
                )

                # Rebuild datasets
                print("Rebuilding training dataset (Phase 2)...")
                train_dataset_phase2 = build_event_tf_dataset(
                    train_event_groups,
                    waveform_path,
                    self.scaler,
                    shuffle=True,
                    batch_size=BATCH_SIZE,
                    trace_prob_cache=train_trace_prob_cache,
                    is_training=True,
                    max_traces=self.max_traces
                )

                print("Rebuilding validation dataset (Phase 2)...")
                val_dataset_phase2 = build_event_tf_dataset(
                    val_event_groups,
                    waveform_path,
                    self.scaler,
                    shuffle=False,
                    batch_size=BATCH_SIZE,
                    trace_prob_cache=val_trace_prob_cache,
                    is_training=False,
                    max_traces=self.max_traces
                )

                # Phase 2 callbacks - stricter min val_loss early stopping
                callbacks_phase2 = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',  # Monitor validation loss
                        min_delta=1e-5,  # Stricter minimum improvement threshold
                        patience=15,  # Longer tolerance period
                        verbose=1,
                        mode='min',  # Minimization mode
                        restore_best_weights=True,  # Restore best weights
                        start_from_epoch=5  # Start monitoring from epoch 6
                    ),
                    keras.callbacks.ModelCheckpoint(
                        os.path.join(CHECKPOINT_DIR, "event_model_best.keras"),
                        monitor='val_loss',
                        save_best_only=True,
                        mode='min',
                        verbose=1
                    ),
                    keras.callbacks.ModelCheckpoint(
                        os.path.join(CHECKPOINT_DIR, "event_model_epoch_{epoch:02d}.keras"),
                        save_freq='epoch',
                        save_weights_only=False,
                        verbose=1
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=8,
                        min_lr=1e-8,
                        mode='min',
                        verbose=1
                    ),
                    keras.callbacks.LearningRateScheduler(
                        create_cosine_scheduler(phase2_lr, EPOCHS - 5)
                    ),
                    ExplosionRecallLogger(val_dataset=val_dataset_phase2, val_steps=val_steps),
                    TracePerformanceLogger(val_dataset=val_dataset_phase2, val_steps=val_steps),
                    MemoryCleaner(),
                    FixTotalLossCallback(),
                    agg_monitor,
                ]

                print(f"\nStarting Phase 2 training (from epoch 6 to epoch {EPOCHS})...")
                print(f"Early stopping settings: monitor='val_loss', mode='min', patience=15, min_delta=1e-5")
                print(f"Learning rate schedule: initial {phase2_lr:.2e}, cosine annealing to epoch {EPOCHS}")
                print_memory_usage("Before Phase 2 training ")

                with timing_context("Phase 2 training"):
                    history_phase2 = self.model.fit(
                        train_dataset_phase2,
                        epochs=EPOCHS,
                        initial_epoch=5,
                        steps_per_epoch=train_steps,
                        validation_data=val_dataset_phase2,
                        validation_steps=val_steps,
                        callbacks=callbacks_phase2,
                        verbose=1
                    )

                print("Phase 2 training completed")
                best_val_loss_phase2 = min(history_phase2.history.get('val_loss', [float('inf')]))
                print(f"Phase 2 best validation loss: {best_val_loss_phase2:.4f}")
                print(f"Phase 2 final validation loss: {history_phase2.history['val_loss'][-1]:.4f}")

                # Merge histories
                print("Merging training histories...")
                event_history = type('History', (), {'history': {}})()
                for key in history_phase1.history.keys():
                    if key in history_phase2.history:
                        event_history.history[key] = (
                                history_phase1.history[key] + history_phase2.history[key]
                        )
                    else:
                        event_history.history[key] = history_phase1.history[key]

                # Save final model
                self.save_model(SAVE_MODEL_PATH)
                print("Model saved to:", SAVE_MODEL_PATH)

                # Output final early stopping statistics
                print(f"\n{'=' * 60}")
                print("Early stopping statistics:")
                print(f"  Phase 1 best val_loss: {best_val_loss_phase1:.6f}")
                print(f"  Phase 2 best val_loss: {best_val_loss_phase2:.6f}")
                print(f"  Overall best val_loss: {min(best_val_loss_phase1, best_val_loss_phase2):.6f}")
                print(f"{'=' * 60}")

            else:
                print("Skipping training, using pretrained model directly")
                trace_counts = [len(traces) for _, traces in train_event_groups]
                self.max_traces = int(np.percentile(trace_counts, 98))
                self.max_traces = max(1, self.max_traces)
                print(f"Fixed max_traces = {self.max_traces}")
                self.load_model(SAVE_MODEL_PATH)

            # ========== Test set evaluation ==========
            all_test_event_groups = []
            for test_name, test_path in test_sets:
                print(f"Evaluating test set: {test_name}")
                test_metadata = load_metadata_from_split(test_path)
                test_event_groups = group_metadata_by_event(test_metadata)
                all_test_event_groups.extend(test_event_groups)

                self.evaluate_test_set(test_name, test_metadata)

            print("Checking unified trace probability cache...")
            if self.trace_prob_cache is not None and len(self.trace_prob_cache) > 0:
                print(f"Using existing cache ({len(self.trace_prob_cache):,} entries), no need to recompute")
            else:
                print("Building unified cache...")

            # Plot training history
            if not skip_training and event_history:
                self.plot_optimized_training_history(trace_history, event_history)

            # Plot geophysical insight figures
            print("\n====== Plotting geophysical insight figures (independent outputs) ======")
            try:
                self.plot_ps_ratio_distribution(val_event_groups, all_test_event_groups)
                print("P/S ratio distribution figure saved")
            except Exception as e:
                print(f"P/S ratio distribution plotting failed: {e}")

            try:
                self.plot_complexity_distance(val_event_groups, all_test_event_groups)
                print("Complexity-distance figure saved")
            except Exception as e:
                print(f"Complexity-distance plotting failed: {e}")

            try:
                if hasattr(self, 'attention_model') and self.attention_model is not None:
                    self.plot_quality_vs_attention(val_event_groups, all_test_event_groups)
                    print("Quality-attention figure saved")
            except Exception as e:
                print(f"Quality-attention plotting failed: {e}")

        except Exception as e:
            error_logger.error("Training interrupted: %s", e, exc_info=True)
            raise e
        finally:
            self.h5_manager.close()
            print("Training pipeline ended (HDF5 closed)")

    def plot_complexity_distance(self, val_event_groups, test_event_groups=None, save_suffix='complexity'):
        from scipy.stats import binned_statistic

        all_groups = val_event_groups + (test_event_groups or [])
        if not all_groups:
            return

        if self.trace_prob_cache is None:
            print("Trace probability cache empty, precomputing...")
            all_traces = pd.concat([tr for _, tr in all_groups], ignore_index=True)
            self.trace_prob_cache = self.precompute_trace_probs(all_traces, self.h5_manager)

        # ========== Key modification 1: Disable normalization for fractal dimension ==========
        # Set scaler to None to avoid normalizing fractal dimension features
        dataset = build_event_tf_dataset(
            all_groups, WAVEFORM_PATH, scaler=None,  # Core change: scaler from self.scaler to None, disable normalization
            shuffle=False, batch_size=32,  # Explicit parameter names for readability
            trace_prob_cache=self.trace_prob_cache, max_traces=self.max_traces
        )
        steps = min(150, max(1, len(all_groups) // 32))

        dist_eq, dist_ex = [], []
        frac_eq, frac_ex = [], []

        for (x1, x2, x3, mask_input), (y_event, y_trace) in dataset.take(steps):
            batch_size = y_event.shape[0]
            for b in range(batch_size):
                feats = x3.numpy()[b]
                labels = y_trace.numpy()[b]
                is_eq = (y_event.numpy()[b] == 1)
                n_real = tf.reduce_sum(tf.cast(labels != -1, tf.int32)).numpy()
                if n_real == 0:
                    continue
                for i in range(n_real):
                    dist = feats[i, 4]
                    frac_dim = feats[i, 0]  # At this point, original fractal dimension value, no normalization
                    # ========== Key modification 2: Add fractal dimension validity check ==========
                    # Ensure fractal dimension is within reasonable range (1~3), filter outliers
                    if 1.0 <= frac_dim <= 3.0:
                        if is_eq:
                            dist_eq.append(dist)
                            frac_eq.append(frac_dim)
                        else:
                            dist_ex.append(dist)
                            frac_ex.append(frac_dim)

        # ========== Plotting optimization: Preserve physical meaning of original fractal dimension ==========
        fig, ax = plt.subplots(figsize=(7.48, 4.5))
        color_eq = '#1a9850'
        color_ex = '#d73027'

        # Earthquake data plotting (preserve original fractal dimension)
        if len(dist_eq) > 10:
            # Use original distance values for binning, avoid information loss after truncation
            bins_eq = np.linspace(0, np.percentile(dist_eq, 95), 10)
            bin_centers_eq = (bins_eq[:-1] + bins_eq[1:]) / 2
            mean_frac_eq, _, _ = binned_statistic(dist_eq, frac_eq, statistic='mean', bins=bins_eq)
            std_frac_eq, _, _ = binned_statistic(dist_eq, frac_eq, statistic='std', bins=bins_eq)
            valid_eq = ~np.isnan(mean_frac_eq)
            ax.errorbar(bin_centers_eq[valid_eq], mean_frac_eq[valid_eq],
                        yerr=std_frac_eq[valid_eq],
                        fmt='o-', color=color_eq, capsize=4, capthick=2,
                        markersize=7, linewidth=2, label='Earthquake', alpha=0.9, zorder=3)

        # Explosion data plotting (preserve original fractal dimension)
        if len(dist_ex) > 10:
            bins_ex = np.linspace(0, np.percentile(dist_ex, 95), 10)
            bin_centers_ex = (bins_ex[:-1] + bins_ex[1:]) / 2
            mean_frac_ex, _, _ = binned_statistic(dist_ex, frac_ex, statistic='mean', bins=bins_ex)
            std_frac_ex, _, _ = binned_statistic(dist_ex, frac_ex, statistic='std', bins=bins_ex)
            valid_ex = ~np.isnan(mean_frac_ex)
            ax.errorbar(bin_centers_ex[valid_ex], mean_frac_ex[valid_ex],
                        yerr=std_frac_ex[valid_ex],
                        fmt='s--', color=color_ex, capsize=4, capthick=2,
                        markersize=7, linewidth=2, label='Explosion', alpha=0.9, zorder=3)

        # ========== Key modification 3: Optimize axis labels, emphasize original fractal dimension meaning ==========
        # Label white noise original fractal dimension reference value (1.5)
        ax.text(0.02, 0.95, 'Ref: D=1.5 (white noise, original scale)',
                transform=ax.transAxes, fontsize=8, style='italic', va='top', alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))
        ax.set_xlabel('Epicentral distance (km)', fontsize=9)
        # Explicitly label "original fractal dimension"
        ax.set_ylabel('Fractal dimension (box-counting, original scale)', fontsize=9)
        ax.set_title('Waveform complexity vs. distance\n(scattering & attenuation, no normalization)',
                     fontsize=10, fontweight='bold')
        ax.legend(loc='lower right', frameon=True, fancybox=False,
                  edgecolor='black', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # Optimize fractal dimension explanation text, emphasize original value meaning
        ax.text(0.98, 0.15, 'Higher D -> more complex (original scale)\nRapid rise = strong scattering',
                transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan',
                          edgecolor='steelblue', alpha=0.8))

        # ========== Key modification 4: Set reasonable y-axis range, match fractal dimension physical meaning ==========
        # Fractal dimension typically between 1~3, fix y-axis range for better comparability
        all_frac = frac_eq + frac_ex
        if all_frac:
            y_min = max(1.0, np.percentile(all_frac, 5) - 0.1)  # Lower bound not below 1.0
            y_max = min(3.0, np.percentile(all_frac, 95) + 0.1)  # Upper bound not above 3.0
            ax.set_ylim(y_min, y_max)

        plt.tight_layout(pad=0.3)
        save_path = TRACE_ATTENTION_HEATMAP_PATH.replace('.png', f'_{save_suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # ========== Key modification 5: Output more detailed original fractal dimension statistics ==========
        if all_frac:
            print(f'Complexity-Distance saved: {save_path}')
            print(f"Original fractal dimension statistics (no normalization):")
            print(f"  Min={np.min(all_frac):.4f}, Max={np.max(all_frac):.4f}, Mean={np.mean(all_frac):.4f}")
            print(f"  Earthquake dimension range: {np.min(frac_eq):.4f}~{np.max(frac_eq):.4f} (n={len(frac_eq)})")
            print(f"  Explosion dimension range: {np.min(frac_ex):.4f}~{np.max(frac_ex):.4f} (n={len(frac_ex)})")
        else:
            print(f"Warning: No valid fractal dimension data, figure not generated")

    def plot_ps_ratio_distribution(self, val_event_groups, test_event_groups=None, save_suffix='ps_ratio'):
        from scipy import stats
        from scipy.stats import gaussian_kde
        import numpy as np
        import matplotlib.pyplot as plt

        all_groups = val_event_groups + (test_event_groups or [])
        if not all_groups:
            return

        if self.trace_prob_cache is None:
            print("Trace probability cache empty, precomputing...")
            all_traces = pd.concat([tr for _, tr in all_groups], ignore_index=True)
            self.trace_prob_cache = self.precompute_trace_probs(all_traces, self.h5_manager)

        dataset = build_event_tf_dataset(
            all_groups, WAVEFORM_PATH, self.scaler, False, 32,
            self.trace_prob_cache, max_traces=self.max_traces
        )
        steps = min(150, max(1, len(all_groups) // 32))

        ps_ratios_eq, ps_ratios_ex = [], []
        print("Collecting P/S ratio data...")
        for (x1, x2, x3, mask_input), (y_event, y_trace) in dataset.take(steps):
            batch_size = y_event.shape[0]
            for b in range(batch_size):
                feats = x3.numpy()[b]
                labels = y_trace.numpy()[b]
                ev_type = 'Earthquake' if y_event.numpy()[b] == 1 else 'Explosion'
                n_real = tf.reduce_sum(tf.cast(labels != -1, tf.int32)).numpy()
                if n_real == 0:
                    continue
                for i in range(n_real):
                    log_ps = feats[i, 3]
                    ps_ratio = np.exp(log_ps)
                    if 0.01 < ps_ratio < 100:
                        if ev_type == 'Earthquake':
                            ps_ratios_eq.append(ps_ratio)
                        else:
                            ps_ratios_ex.append(ps_ratio)

        ps_ratios_eq = np.array(ps_ratios_eq)
        ps_ratios_ex = np.array(ps_ratios_ex)
        ps_ratios_eq = ps_ratios_eq[ps_ratios_eq > 0]
        ps_ratios_ex = ps_ratios_ex[ps_ratios_ex > 0]

        if len(ps_ratios_eq) < 3 or len(ps_ratios_ex) < 3:
            print(f"Warning: Insufficient P/S ratio data (EQ: {len(ps_ratios_eq)}, EX: {len(ps_ratios_ex)}), skipping plot")
            return

        log_ps_eq = np.log10(ps_ratios_eq)
        log_ps_ex = np.log10(ps_ratios_ex)
        log_ps_eq = log_ps_eq[np.isfinite(log_ps_eq)]
        log_ps_ex = log_ps_ex[np.isfinite(log_ps_ex)]

        fig, ax = plt.subplots(figsize=(9, 6))
        color_eq = '#1a9850'
        color_ex = '#d73027'

        x_min = min(log_ps_eq.min(), log_ps_ex.min()) - 0.5
        x_max = max(log_ps_eq.max(), log_ps_ex.max()) + 0.5
        x_range = np.linspace(x_min, x_max, 200)

        try:
            if len(log_ps_eq) > 5 and np.std(log_ps_eq) > 0:
                kde_eq = gaussian_kde(log_ps_eq)
                ax.fill_between(x_range, kde_eq(x_range), alpha=0.6, color=color_eq,
                                label=f'Earthquake (n={len(log_ps_eq)})')
                ax.plot(x_range, kde_eq(x_range), color=color_eq, linewidth=2.5)
                median_eq = np.median(log_ps_eq).item()
                ax.axvline(median_eq, color=color_eq, linestyle='--', linewidth=1.5, alpha=0.7,
                           label=f'EQ median ({median_eq:.2f})')
            if len(log_ps_ex) > 5 and np.std(log_ps_ex) > 0:
                kde_ex = gaussian_kde(log_ps_ex)
                ax.fill_between(x_range, kde_ex(x_range), alpha=0.6, color=color_ex,
                                label=f'Explosion (n={len(log_ps_ex)})')
                ax.plot(x_range, kde_ex(x_range), color=color_ex, linewidth=2.5)
                median_ex = np.median(log_ps_ex).item()
                ax.axvline(median_ex, color=color_ex, linestyle='--', linewidth=1.5, alpha=0.7,
                           label=f'EX median ({median_ex:.2f})')
        except Exception as e:
            print(f"KDE computation failed: {e}")
            ax.hist(log_ps_eq, bins=30, alpha=0.6, color=color_eq, density=True, label='Earthquake')
            ax.hist(log_ps_ex, bins=30, alpha=0.6, color=color_ex, density=True, label='Explosion')

        ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='P/S = 1')
        ax.axvspan(x_min, 0, alpha=0.1, color='blue', label='S-dominant')
        ax.axvspan(0, x_max, alpha=0.1, color='red', label='P-dominant')
        ax.set_xlabel(r'$\log_{10}(A_P/A_S)$ amplitude ratio', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability density', fontsize=11, fontweight='bold')
        ax.set_title('P/S amplitude ratio discrimination\n(source mechanism difference)',
                     fontsize=12, fontweight='bold', pad=15)
        ax.legend(loc='upper left', frameon=True, fancybox=False,
                  edgecolor='black', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        if len(log_ps_eq) > 1 and len(log_ps_ex) > 1:
            try:
                if np.var(log_ps_eq) > 0 and np.var(log_ps_ex) > 0:
                    t_stat, p_val = stats.ttest_ind(log_ps_eq, log_ps_ex, equal_var=False)

                    mean_eq = float(np.mean(log_ps_eq))
                    mean_ex = float(np.mean(log_ps_ex))
                    var_eq = float(np.var(log_ps_eq))
                    var_ex = float(np.var(log_ps_ex))
                    effect_size = (mean_eq - mean_ex) / np.sqrt((var_eq + var_ex) / 2)
                    effect_size = float(effect_size)
                    p_val = float(p_val)

                    ax.text(0.98, 0.95, f't-test p={p_val:.2e}\nCohen\'s d={effect_size:.2f}',
                            transform=ax.transAxes, fontsize=9, va='top', ha='right',
                            bbox=dict(boxstyle='square,pad=0.4', facecolor='white',
                                      edgecolor='gray', alpha=0.9))
            except Exception as e:
                print(f"Statistical test failed: {e}")

        plt.tight_layout(pad=0.3)
        save_path = TRACE_ATTENTION_HEATMAP_PATH.replace('.png', f'_{save_suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'P/S Ratio distribution saved: {save_path}')

    def plot_aggregation_methods_history(self, event_history):
        """
        Plot validation accuracy curves for different aggregation methods during event model training.
        Key format: val_agg_{method}_acc, e.g., val_agg_attention_acc
        """
        if event_history is None:
            return

        # Extract all validation accuracy keys for aggregation methods
        agg_keys = [key for key in event_history.history.keys()
                    if key.startswith('val_agg_') and key.endswith('_acc')]
        if not agg_keys:
            print("Aggregation method metrics not found in training history, skipping plot.")
            return

        # Extract method names (remove prefix and suffix)
        method_names = [key.replace('val_agg_', '').replace('_acc', '')
                        for key in agg_keys]

        fig, ax = plt.subplots(figsize=(7.48, 4.5))  # GJI standard single column width

        for key, name in zip(agg_keys, method_names):
            ax.plot(event_history.history[key],
                    label=name.replace('_', ' ').title(),
                    linewidth=1.2)

        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Validation Accuracy', fontsize=9)
        ax.set_title('Comparison of Aggregation Methods', fontsize=10, fontweight='bold')
        ax.legend(loc='best', frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.tick_params(labelsize=8)

        # Generate unique path: based on EVENT_HISTORY_PLOT_PATH add '_aggregation' suffix, ensure GJI format
        base = EVENT_HISTORY_PLOT_PATH
        # If original path contains '_GJI', remove it before re-adding
        if '_GJI' in base:
            base = base.replace('_GJI', '')
        save_path = base.replace('.png', '_aggregation_GJI.png')

        plt.tight_layout(pad=0.5)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Aggregation methods history figure saved: {save_path}")

    def plot_optimized_training_history(self, trace_history, event_history):
        """
        Optimized training history plotting: plot trace-level and event-level histories separately, and add aggregation method comparison figure.
        """
        if trace_history is not None:
            self.plot_trace_training_history(trace_history)
        if event_history is not None:
            self.plot_event_training_history(event_history)
            # New: plot aggregation method comparison figure
            self.plot_aggregation_methods_history(event_history)

    def plot_quality_vs_attention(self, val_event_groups, test_event_groups=None, save_suffix='quality'):
        import matplotlib.pyplot as plt
        from statsmodels.nonparametric.smoothers_lowess import lowess
        import numpy as np
        import pandas as pd
        import tensorflow as tf

        all_groups = val_event_groups + (test_event_groups or [])
        if not all_groups:
            print("Warning: No event data available")
            return

        if self.trace_prob_cache is None:
            print("Trace probability cache empty, precomputing...")
            all_traces = pd.concat([tr for _, tr in all_groups], ignore_index=True)
            self.trace_prob_cache = self.precompute_trace_probs(all_traces, self.h5_manager)

        dataset = build_event_tf_dataset(
            all_groups, WAVEFORM_PATH, self.scaler, False, 32,
            self.trace_prob_cache, max_traces=self.max_traces
        )
        steps = min(100, max(1, len(all_groups) // 32))

        distances, qualities, attention_weights = [], [], []
        event_types, depths, correctness_labels = [], [], []

        print("Collecting quality-attention data...")
        for (x1, x2, x3, mask_input), (y_event, y_trace) in dataset.take(steps):
            preds = self.model.predict([x1, x2, x3, mask_input], verbose=0)
            attn_weights = self.attention_model.predict([x1, x2, x3], verbose=0)
            trace_pred = preds[1]  # Now [batch, traces]

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
                    # Fix: trace_pred is 2D
                    pred_lab = 1 if trace_pred[b, i] > 0.5 else 0
                    true_lab = int(labels[i])
                    correct = (pred_lab == true_lab)
                    prob = np.clip(trace_pred[b, i], 1e-5, 1 - 1e-5)
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

        fig, axes = plt.subplots(2, 2, figsize=(7.48, 6.69))
        axes = axes.flatten()
        labels = ['(a)', '(b)', '(c)', '(d)']

        color_eq = '#1b7837'
        color_ex = '#d73027'

        # (a) Distance vs Quality
        ax = axes[0]
        correct_mask = correctness_labels
        wrong_mask = ~correctness_labels
        ax.scatter(distances[correct_mask], qualities[correct_mask],
                   c='#2166ac', s=20, alpha=0.5,
                   label='Correct', edgecolors='none', rasterized=True, zorder=2)
        ax.scatter(distances[wrong_mask], qualities[wrong_mask],
                   c='#b2182b', s=20, alpha=0.5,
                   label='Incorrect', edgecolors='none', rasterized=True, zorder=2)
        if len(distances) > 10:
            trend = lowess(qualities, distances, frac=0.3, return_sorted=False)
            sort_idx = np.argsort(distances)
            ax.plot(distances[sort_idx], trend[sort_idx], 'k-',
                    linewidth=1.5, label='Trend', zorder=5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=1)
        ax.set_xlabel('Epicentral Distance (km)', fontsize=9)
        ax.set_ylabel('Signed Quality Score', fontsize=9)
        ax.set_title('Distance vs. Quality', fontsize=10, fontweight='bold')
        ax.text(0.02, 0.98, labels[0], transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        ax.legend(loc='upper right', frameon=True, fancybox=False,
                  edgecolor='black', fontsize=7)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # (b) Quality Distribution
        ax = axes[1]
        bins = np.linspace(-1, 1, 31)
        ax.hist(qualities[correctness_labels], bins=bins, color='#2166ac',
                alpha=0.7, label='Correct', edgecolor='white', linewidth=0.5)
        ax.hist(qualities[~correctness_labels], bins=bins, color='#b2182b',
                alpha=0.7, label='Incorrect', edgecolor='white', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Signed Quality Score', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title('Distribution of Quality Scores', fontsize=10, fontweight='bold')
        ax.text(0.02, 0.98, labels[1], transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        ax.legend(loc='upper right', frameon=True, fancybox=False,
                  edgecolor='black', fontsize=7)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # (c) Quality by Event Type
        ax = axes[2]
        eq_mask = event_types == 'Earthquake'
        ex_mask = ~eq_mask
        ax.scatter(distances[eq_mask], qualities[eq_mask], c=color_eq,
                   s=25, alpha=0.60, edgecolors='none',
                   label='Earthquake', rasterized=True, zorder=2)
        ax.scatter(distances[ex_mask], qualities[ex_mask], c=color_ex,
                   s=25, alpha=0.60, edgecolors='none',
                   label='Explosion', rasterized=True, zorder=2)
        if np.sum(eq_mask) > 10:
            sort_eq = np.argsort(distances[eq_mask])
            trend_eq = lowess(qualities[eq_mask], distances[eq_mask],
                              frac=0.4, return_sorted=False)
            ax.plot(distances[eq_mask][sort_eq], trend_eq[sort_eq],
                    linestyle='-', color='#2166ac', linewidth=1.2,
                    marker='o', markersize=2.5, markerfacecolor='white',
                    markeredgewidth=0.8, markeredgecolor='#2166ac',
                    label='EQ trend', zorder=5)
        if np.sum(ex_mask) > 10:
            sort_ex = np.argsort(distances[ex_mask])
            trend_ex = lowess(qualities[ex_mask], distances[ex_mask],
                              frac=0.4, return_sorted=False)
            ax.plot(distances[ex_mask][sort_ex], trend_ex[sort_ex],
                    linestyle='--', color='k', linewidth=1.2,
                    marker='s', markersize=2.5, markerfacecolor='white',
                    markeredgewidth=0.8, markeredgecolor='k',
                    label='EX trend', zorder=5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
        ax.set_xlabel('Epicentral Distance (km)', fontsize=9)
        ax.set_ylabel('Signed Quality Score', fontsize=9)
        ax.set_title('Quality by Event Type', fontsize=10, fontweight='bold')
        ax.text(0.02, 0.98, labels[2], transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        ax.legend(loc='best', frameon=True, fancybox=False,
                  edgecolor='black', fontsize=7, ncol=2,
                  columnspacing=0.5, handletextpad=0.3)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # (d) Depth vs Quality
        ax = axes[3]
        ax.scatter(depths[correct_mask], qualities[correct_mask],
                   c='#2166ac', s=20, alpha=0.5, edgecolors='none',
                   label='Correct', zorder=2)
        ax.scatter(depths[wrong_mask], qualities[wrong_mask],
                   c='#b2182b', s=20, alpha=0.5, edgecolors='none',
                   label='Incorrect', zorder=2)
        if len(depths) > 10:
            trend = lowess(qualities, depths, frac=0.3, return_sorted=False)
            sort_idx = np.argsort(depths)
            ax.plot(depths[sort_idx], trend[sort_idx], 'k-', linewidth=1.5,
                    label='Trend', zorder=5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
        ax.set_xlabel('Source Depth (km)', fontsize=9)
        ax.set_ylabel('Signed Quality Score', fontsize=9)
        ax.set_title('Depth vs. Quality', fontsize=10, fontweight='bold')
        ax.text(0.02, 0.98, labels[3], transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        ax.legend(loc='upper right', frameon=True, fancybox=False,
                  edgecolor='black', fontsize=7)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        plt.tight_layout(pad=0.5)
        combo_path = TRACE_ATTENTION_HEATMAP_PATH.replace('.png', f'_{save_suffix}_GJI_combo.png')
        plt.savefig(combo_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f'Combined figure saved: {combo_path}')
        plt.close(fig)

    @staticmethod
    def plot_trace_training_history(trace_history):
        if trace_history is None:
            print("No trace training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(7.48, 6.69))
        axes = axes.flatten()
        labels = ['(a)', '(b)', '(c)', '(d)']
        metrics = [
            ('accuracy', 'Accuracy'),
            ('loss', 'Loss'),
            ('precision', 'Precision'),
            ('recall', 'Recall')
        ]

        color_train = '#2874A6'
        color_val = '#E67E22'

        for i, (ax, (key, title)) in enumerate(zip(axes, metrics)):
            train_key = key
            val_key = f'val_{key}'

            if train_key in trace_history.history:
                ax.plot(trace_history.history[train_key], color=color_train, linestyle='-',
                        linewidth=1.5, label='Training', alpha=0.9)
            if val_key in trace_history.history:
                ax.plot(trace_history.history[val_key], color=color_val, linestyle='--',
                        linewidth=1.5, label='Validation', alpha=0.9)

            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel(title, fontsize=9)
            ax.set_title(f'{labels[i]} {title}', fontsize=9)
            ax.text(0.02, 0.98, labels[i], transform=ax.transAxes,
                    fontsize=9, fontweight='bold', va='top', ha='left')
            ax.legend(frameon=False, loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.tick_params(labelsize=8)

            if key in ['accuracy', 'precision', 'recall']:
                ax.set_ylim(0, 1.05)

        plt.tight_layout(pad=0.5)
        plt.savefig(TRACE_HISTORY_PLOT_PATH.replace('.png', '_GJI.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Trace training history saved: {TRACE_HISTORY_PLOT_PATH.replace('.png', '_GJI.png')}")

    def plot_event_training_history(self, event_history):
        if event_history is None:
            print("No event training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(7.48, 6.69))
        axes = axes.flatten()
        labels = ['(a)', '(b)', '(c)', '(d)']
        keys = ['event_output_accuracy', 'event_output_loss',
                'event_output_precision', 'event_output_recall']
        titles = ['Accuracy', 'Loss', 'Precision', 'Recall']

        color_train = '#2874A6'
        color_val = '#E67E22'

        for ax, k, t, lbl in zip(axes, keys, titles, labels):
            train_key = k
            val_key = f'val_{k}'

            if train_key in event_history.history:
                ax.plot(event_history.history[train_key], color=color_train, linestyle='-',
                        linewidth=1.5, label='Training', alpha=0.9)
            if val_key in event_history.history:
                ax.plot(event_history.history[val_key], color=color_val, linestyle='--',
                        linewidth=1.5, label='Validation', alpha=0.9)

            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel(t, fontsize=9)
            ax.set_title(f'{lbl} {t}', fontsize=9)
            ax.text(0.02, 0.98, lbl, transform=ax.transAxes,
                    fontsize=9, fontweight='bold', va='top', ha='left')
            ax.legend(frameon=False, loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.tick_params(labelsize=8)

        plt.tight_layout(pad=0.5)
        plt.savefig(EVENT_HISTORY_PLOT_PATH.replace('.png', '_GJI.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Event training history saved: {EVENT_HISTORY_PLOT_PATH.replace('.png', '_GJI.png')}")

    def _build_attention_model(self):
        if self.model is None:
            raise RuntimeError("Must load self.model before building attention_model")
        inputs = self.model.input
        attention_layer = self.model.get_layer("attention_weights")
        attention_output = attention_layer.output
        self.attention_model = keras.Model(inputs=inputs[:3], outputs=attention_output)

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

        if self.shared_wave_encoder is not None:
            wave_encoder_path = path.replace(".keras", "_wave_encoder.keras")
            self.shared_wave_encoder.save(wave_encoder_path)

        if self.shared_spec_encoder is not None:
            spec_encoder_path = path.replace(".keras", "_spec_encoder.keras")
            self.shared_spec_encoder.save(spec_encoder_path)

        if self.shared_feat_encoder is not None:
            feat_encoder_path = path.replace(".keras", "_feat_encoder.keras")
            self.shared_feat_encoder.save(feat_encoder_path)

        if self.trace_prob_cache is not None:
            cache_path = path.replace(".keras", "_trace_probs.joblib")
            joblib.dump(self.trace_prob_cache, cache_path)
            print(f"Trace probability cache saved to: {cache_path}")

        max_traces_path = path.replace(".keras", "_max_traces.txt")
        with open(max_traces_path, 'w') as f:
            f.write(str(self.max_traces))
        print(f"max_traces ({self.max_traces}) saved")

    def load_model(self, path):
        import keras
        keras.config.enable_unsafe_deserialization()

        self.model = keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, safe_mode=False)
        self._build_attention_model()

        trace_model_path = path.replace(".keras", "_trace_model.keras")
        if os.path.exists(trace_model_path):
            self.trace_model = keras.models.load_model(trace_model_path, custom_objects=CUSTOM_OBJECTS, safe_mode=False)

        scaler_path = path.replace(".keras", "_scaler.joblib")
        self.scaler = joblib.load(scaler_path)

        wave_encoder_path = path.replace(".keras", "_wave_encoder.keras")
        if os.path.exists(wave_encoder_path):
            self.shared_wave_encoder = keras.models.load_model(wave_encoder_path, custom_objects=CUSTOM_OBJECTS,
                                                               safe_mode=False)

        spec_encoder_path = path.replace(".keras", "_spec_encoder.keras")
        if os.path.exists(spec_encoder_path):
            self.shared_spec_encoder = keras.models.load_model(spec_encoder_path, custom_objects=CUSTOM_OBJECTS,
                                                               safe_mode=False)

        feat_encoder_path = path.replace(".keras", "_feat_encoder.keras")
        if os.path.exists(feat_encoder_path):
            self.shared_feat_encoder = keras.models.load_model(feat_encoder_path, custom_objects=CUSTOM_OBJECTS,
                                                               safe_mode=False)

        cache_path = path.replace(".keras", "_trace_probs.joblib")
        if os.path.exists(cache_path):
            self.trace_prob_cache = joblib.load(cache_path)
            print(f"Trace probability cache loaded: {cache_path}")
        else:
            print("Warning: Trace probability cache not found, will recompute during plotting")

        max_traces_path = path.replace(".keras", "_max_traces.txt")
        if os.path.exists(max_traces_path):
            with open(max_traces_path, 'r') as f:
                self.max_traces = int(f.read().strip())
            print(f"max_traces loaded: {self.max_traces}")

    def load_latest_checkpoint(self):
        if not os.path.exists(CHECKPOINT_DIR):
            return None

        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('event_model_') and f.endswith('.keras')]
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)), reverse=True)
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[0])
        print(f"Loading latest event model checkpoint: {latest_checkpoint}")

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


def main():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {[gpu.name for gpu in gpus]}")
        else:
            print("Using CPU")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

    model_exists = os.path.exists(SAVE_MODEL_PATH)
    trace_model_path = SAVE_MODEL_PATH.replace(".keras", "_trace_model.keras")
    trace_model_exists = os.path.exists(trace_model_path)
    scaler_path = SAVE_MODEL_PATH.replace(".keras", "_scaler.joblib")
    scaler_exists = os.path.exists(scaler_path)

    skip_training = model_exists and trace_model_exists and scaler_exists

    if skip_training:
        print(f"Complete pretrained model found: {SAVE_MODEL_PATH}")
        print("Skipping training, proceeding directly to testing and visualization")
    else:
        print("Complete pretrained model not found")
        print("Starting training process")

    classifier = EarthquakeClassifier(COLUMN_MAPPING)

    try:
        print("=" * 60)
        if skip_training:
            print("Improved earthquake vs explosion classification - Using pretrained model (weight sharing version)")
        else:
            print("Improved earthquake vs explosion classification training started (weight sharing version)")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        if skip_training:
            print(f"Loading pretrained model from {SAVE_MODEL_PATH}")
            try:
                classifier.load_model(SAVE_MODEL_PATH)
                classifier.h5_manager = H5FileManager(WAVEFORM_PATH)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Model loading failed: {e}")
                print("Retraining model...")
                skip_training = False

        if not skip_training:
            classifier.h5_manager = H5FileManager(WAVEFORM_PATH)

        print("Starting training pipeline...")
        hist = classifier.train(
            train_path=TRAIN_PATH,
            val_path=VAL_PATH,
            test_sets=TEST_SETS,
            waveform_path=WAVEFORM_PATH,
            skip_training=skip_training
        )

        if classifier.model is None:
            print("Error: Model training failed, model is None!")
            return

        print("Saving model...")
        classifier.save_model(SAVE_MODEL_PATH)

        print("Verifying model reload...")
        try:
            keras.config.enable_unsafe_deserialization()
            loaded_model = keras.models.load_model(SAVE_MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
            print("Model can be reloaded successfully!")
        except Exception as e:
            print(f"Model reload failed: {e}")

        print("Preparing validation and test event groups...")
        val_metadata = load_metadata_from_split(VAL_PATH)
        val_event_groups = group_metadata_by_event(val_metadata)

        test_event_groups = []
        for _, test_path in TEST_SETS:
            test_meta = load_metadata_from_split(test_path)
            test_event_groups.extend(group_metadata_by_event(test_meta))

        print('\n====== Starting to plot performance and visualization figures ======')

        if hist is not None:
            trace_history, event_history = hist
            if trace_history is not None and event_history is not None:
                classifier.plot_optimized_training_history(trace_history, event_history)
                print("Training history figures saved")
            else:
                print('(Skipping training history figures: no history data)')
        else:
            print('(Skipping training history figures: no training history)')

        print("Plotting event model quality diagnostic figures...")
        try:
            classifier.plot_quality_vs_attention(val_event_groups, test_event_groups)
            print("Event model quality diagnostic figures saved")
        except Exception as e:
            print(f"Event model quality diagnostic plotting failed: {e}")

        print('====== All figures saved to project directory ======')

        print("\n" + "=" * 60)
        print("Final statistics:")
        print(f"- Validation events: {len(val_event_groups)}")
        print(
            f"- Test events: {sum(len(group_metadata_by_event(load_metadata_from_split(test_path))) for _, test_path in TEST_SETS)}")

        total_traces = 0
        for _, traces in val_event_groups:
            total_traces += len(traces)
        for test_name, test_path in TEST_SETS:
            test_meta = load_metadata_from_split(test_path)
            test_groups = group_metadata_by_event(test_meta)
            for _, traces in test_groups:
                total_traces += len(traces)

        print(f"- Total traces: {total_traces}")
        print(f"- Model saved to: {SAVE_MODEL_PATH}")
        print(f"- Results output to: {RESULT_OUTPUT_PATH}")
        print("=" * 60)

    except Exception as e:
        import traceback
        print("\n" + "=" * 60)
        print("An error occurred during processing:")
        traceback.print_exc()
        print("=" * 60)

        if hasattr(classifier, 'model') and classifier.model is not None:
            print("Attempting to save current model...")
            try:
                backup_path = SAVE_MODEL_PATH.replace('.keras', '_backup.keras')
                classifier.save_model(backup_path)
                print(f"Model backup saved to: {backup_path}")
            except Exception as save_error:
                print(f"Model backup failed: {save_error}")
        exit(1)


if __name__ == "__main__":
    main()
