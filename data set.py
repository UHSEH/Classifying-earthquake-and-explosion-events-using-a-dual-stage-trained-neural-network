import os
import random
import numpy as np
import pandas as pd
import h5py
from obspy import UTCDateTime
from obspy.signal.invsim import cosine_taper
from obspy.core.trace import Trace, Stats

# -------------------------- 1. 基础配置（不变） --------------------------
METADATA_PATH = "/home/he/PycharmProjects/PythonProject/dataset/PNW-ML/comcat_metadata.csv"
WAVEFORM_HDF5_PATH = "/home/he/PycharmProjects/PythonProject/dataset/PNW-ML/comcat_waveforms.hdf5"
OUTPUT_ROOT = "/home/he/PycharmProjects/PythonProject/dataset/processed_comcat"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 关键参数
TRAIN_TEST_SPLIT = 0.9
VAL_SPLIT_FROM_TRAIN = 0.2
AUGMENT_TIMES = 5  # 爆炸事件固定增强5倍
SAMPLE_RATE = 100
WAVEFORM_LENGTH = 60
WAVEFORM_POINTS = WAVEFORM_LENGTH * SAMPLE_RATE + 1  # 6001
HIGHPASS_FREQ = 2
BATCH_SIZE = 800
RANDOM_SEED = 42  # 固定种子确保结果可复现
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------- 2. 加载元数据（不进行平衡，只筛选有效数据） --------------------------
def load_metadata(metadata_path):
    metadata = pd.read_csv(metadata_path)

    required_cols = [
        "event_id", "source_type", "preferred_source_magnitude",
        "source_origin_time", "trace_start_time", "trace_P_arrival_sample",
        "trace_sampling_rate_hz", "trace_name"
    ]
    missing_cols = [col for col in required_cols if col not in metadata.columns]
    if missing_cols:
        raise ValueError(f"缺少关键列：{missing_cols}")

    # 解析 trace_name
    parsed_buckets = []
    parsed_tindexes = []
    invalid_count = 0
    for _, row in metadata.iterrows():
        trace_name = str(row["trace_name"])
        try:
            bucket_part = trace_name.split("$")[0]
            tindex_str = trace_name.split("$")[1].split(",")[0]
            parsed_buckets.append(bucket_part)
            parsed_tindexes.append(int(tindex_str))
        except:
            invalid_count += 1
            parsed_buckets.append(None)
            parsed_tindexes.append(None)

    metadata["bucket"] = parsed_buckets
    metadata["tindex"] = parsed_tindexes
    metadata = metadata[metadata["bucket"].notna() & metadata["tindex"].notna()].copy()
    print(f"✅ trace_name 解析：成功{len(metadata)}个，失败{invalid_count}个")

    # 筛选有效事件
    metadata = metadata[metadata["source_type"].isin(["earthquake", "explosion"])].copy()
    metadata["mag"] = pd.to_numeric(metadata["preferred_source_magnitude"], errors="coerce")
    metadata = metadata[(metadata["mag"].notna()) & (metadata["mag"] >= 0) & (metadata["mag"] <= 10)].copy()
    metadata = metadata[
        (metadata["trace_P_arrival_sample"] >= 0) &
        (metadata["trace_P_arrival_sample"] < metadata["trace_sampling_rate_hz"] * 150)
        ].copy()

    # 计算P波时间
    p_arrival_times = []
    for _, row in metadata.iterrows():
        try:
            start_utc = UTCDateTime(row["trace_start_time"])
            p_arrival = start_utc + row["trace_P_arrival_sample"] / row["trace_sampling_rate_hz"]
            p_arrival_times.append(p_arrival.isoformat())
        except:
            p_arrival_times.append(None)
    metadata["p_arrival_time"] = p_arrival_times
    metadata = metadata[metadata["p_arrival_time"].notna()].copy()

    # 重命名列
    metadata = metadata.rename(columns={"source_type": "event_type", "source_origin_time": "origin_time"})

    # 添加震级分箱
    metadata["mag_bin"] = np.select(
        [(metadata["mag"] >= 0) & (metadata["mag"] < 1),
         (metadata["mag"] >= 1) & (metadata["mag"] < 2),
         (metadata["mag"] >= 2) & (metadata["mag"] < 3),
         (metadata["mag"] >= 3) & (metadata["mag"] <= 10)],
        ["0-1", "1-2", "2-3", "3-10"], default="other"
    )

    # 统计原始数据
    explosion_traces = metadata[metadata["event_type"] == "explosion"]
    earthquake_traces = metadata[metadata["event_type"] == "earthquake"]

    print(f"\n 原始数据统计：")
    print(f"  - 地震trace：{len(earthquake_traces)}个")
    print(f"  - 爆炸trace：{len(explosion_traces)}个")
    print(f"  - 地震事件：{earthquake_traces['event_id'].nunique()}个")
    print(f"  - 爆炸事件：{explosion_traces['event_id'].nunique()}个")

    # 统计震级分箱
    print(f"\n 震级分箱统计：")
    for event_type in ["earthquake", "explosion"]:
        type_traces = metadata[metadata["event_type"] == event_type]
        print(f"  - {event_type}:")
        for mag_bin in ["0-1", "1-2", "2-3", "3-10"]:
            bin_count = len(type_traces[type_traces["mag_bin"] == mag_bin])
            if bin_count > 0:
                print(f"    {mag_bin}: {bin_count}个trace")

    return metadata


# -------------------------- 3. 事件级时间分割（保持震级分箱平衡） --------------------------
def split_train_test_val(metadata_df):
    # 生成事件级数据
    event_level_df = metadata_df.groupby("event_id").agg({
        "origin_time": "first",
        "event_type": "first",
        "mag": "first",
        "mag_bin": "first",
        "trace_name": "count"
    }).rename(columns={"trace_name": "trace_count"}).reset_index()

    # 按事件发生时间排序
    event_level_df = event_level_df.sort_values("origin_time").reset_index(drop=True)
    total_events = len(event_level_df)
    total_traces = len(metadata_df)

    print(f"\n 事件级统计（分割前）：")
    print(f"  - 总事件数：{total_events}个")
    print(f"  - 总trace数：{total_traces}个")
    print(f"  - 平均每个事件trace数：{total_traces / total_events:.2f}个")

    # 分割Train/Test事件ID
    train_event_num = int(total_events * TRAIN_TEST_SPLIT)
    train_event_ids = set(event_level_df.iloc[:train_event_num]["event_id"].tolist())
    test_event_ids = set(event_level_df.iloc[train_event_num:]["event_id"].tolist())

    # 从Train事件中分割Val事件ID
    train_events_subset = event_level_df[event_level_df["event_id"].isin(train_event_ids)]
    val_event_num = int(len(train_events_subset) * VAL_SPLIT_FROM_TRAIN)
    val_event_ids = set(train_events_subset.sample(val_event_num, random_state=RANDOM_SEED)["event_id"].tolist())
    train_event_ids = train_event_ids - val_event_ids

    # 校验事件ID无重叠
    overlap_train_val = train_event_ids & val_event_ids
    overlap_train_test = train_event_ids & test_event_ids
    overlap_val_test = val_event_ids & test_event_ids
    assert len(overlap_train_val) == 0, f"❌ Train与Val存在重叠事件ID：{list(overlap_train_val)[:5]}..."
    assert len(overlap_train_test) == 0, f"❌ Train与Test存在重叠事件ID：{list(overlap_train_test)[:5]}..."
    assert len(overlap_val_test) == 0, f"❌ Val与Test存在重叠事件ID：{list(overlap_val_test)[:5]}..."
    print(f"✅ 所有子集事件ID无重叠，无数据泄露风险！")

    # 按事件ID分配对应的所有trace到子集
    train_df = metadata_df[metadata_df["event_id"].isin(train_event_ids)].copy().reset_index(drop=True)
    val_df = metadata_df[metadata_df["event_id"].isin(val_event_ids)].copy().reset_index(drop=True)
    test_df = metadata_df[metadata_df["event_id"].isin(test_event_ids)].copy().reset_index(drop=True)

    # 统计各子集的事件和trace分布
    def count_subset_stats(df, subset_name):
        total_event = df["event_id"].nunique()
        eq_event = df[df["event_type"] == "earthquake"]["event_id"].nunique()
        ex_event = df[df["event_type"] == "explosion"]["event_id"].nunique()
        total_trace = len(df)
        eq_trace = len(df[df["event_type"] == "earthquake"])
        ex_trace = len(df[df["event_type"] == "explosion"])
        event_ratio = (total_event / total_events) * 100
        trace_ratio = (total_trace / total_traces) * 100
        return {
            "subset": subset_name,
            "total_event": total_event, "eq_event": eq_event, "ex_event": ex_event, "event_ratio": event_ratio,
            "total_trace": total_trace, "eq_trace": eq_trace, "ex_trace": ex_trace, "trace_ratio": trace_ratio
        }

    # 生成统计结果
    train_stats = count_subset_stats(train_df, "Train")
    val_stats = count_subset_stats(val_df, "Val")
    test_stats = count_subset_stats(test_df, "Test")

    # 打印分割结果
    print(f"\n 事件级时间分割最终结果：")
    print("  " + "-" * 120)
    print(f"  {'子集':<8} {'事件数(总/震/爆)':<20} {'事件占比(%)':<12} {'trace数(总/震/爆)':<20} {'trace占比(%)':<12}")
    print("  " + "-" * 120)
    for stats in [train_stats, val_stats, test_stats]:
        event_str = f"{stats['total_event']}/{stats['eq_event']}/{stats['ex_event']}"
        trace_str = f"{stats['total_trace']}/{stats['eq_trace']}/{stats['ex_trace']}"
        print(
            f"  {stats['subset']:<8} {event_str:<20} {stats['event_ratio']:<12.1f} {trace_str:<20} {stats['trace_ratio']:<12.1f}")
    print("  " + "-" * 120)
    print(
        f"  {'总计':<8} {total_events}/{event_level_df[event_level_df['event_type'] == 'earthquake'].shape[0]}/{event_level_df[event_level_df['event_type'] == 'explosion'].shape[0]:<12} 100.0{'':<10} {total_traces}/{len(metadata_df[metadata_df['event_type'] == 'earthquake'])}/{len(metadata_df[metadata_df['event_type'] == 'explosion']):<12} 100.0")

    return train_df, val_df, test_df


# -------------------------- 4. 读取HDF5波形数据（不变） --------------------------
def read_waveform_from_hdf5(hdf5_file, bucket_name, tindex, sampling_rate, start_time):
    waveform_traces = []
    required_comps = ["Z", "N", "E"]
    bucket_path = f"data/{bucket_name}"

    if bucket_path not in hdf5_file:
        raise ValueError(f"HDF5无bucket：{bucket_path}")

    bucket_dataset = hdf5_file[bucket_path]
    n_events = bucket_dataset.shape[0]
    if tindex < 0 or tindex >= n_events:
        raise ValueError(f"tindex {tindex} 超出范围（0~{n_events - 1}）")

    wave_data = bucket_dataset[tindex]
    if wave_data.shape[0] != 3:
        raise ValueError(f"分量数异常：{wave_data.shape[0]}（需为3）")

    comp_order = hdf5_file["data_format/component_order"][()].decode('utf-8')
    comp_map = {c: i for i, c in enumerate(comp_order)}
    missing = [c for c in required_comps if c not in comp_map]
    if missing:
        raise ValueError(f"缺少分量：{missing}（顺序：{comp_order}）")

    start_utc = UTCDateTime(start_time)
    for comp in required_comps:
        comp_data = wave_data[comp_map[comp]].astype(np.float32)
        stats = Stats({
            "npts": len(comp_data),
            "sampling_rate": sampling_rate,
            "starttime": start_utc,
            "channel": comp
        })
        waveform_traces.append(Trace(data=comp_data, header=stats))

    return waveform_traces


# -------------------------- 5. 波形预处理（不变） --------------------------
def preprocess_waveform(waveform_traces, p_arrival_time, random_seed=None):
    processed_comps = []

    try:
        p_utc = UTCDateTime(p_arrival_time)
        p_ts = p_utc.timestamp
    except:
        print("⚠️ P波时间无效，零填充")
        return np.zeros((WAVEFORM_POINTS, 3), dtype=np.float32)

    try:
        wave_ts = waveform_traces[0].stats.starttime.timestamp
        time_before_p = p_ts - wave_ts
    except:
        time_before_p = 0

    if random_seed is not None:
        random.seed(random_seed)
    if time_before_p >= 20:
        offset = random.uniform(5, 20)
    elif 5 <= time_before_p < 20:
        offset = random.uniform(5, time_before_p)
    else:
        offset = 0

    for trace in waveform_traces:
        trace = trace.copy()
        if trace.stats.sampling_rate != SAMPLE_RATE:
            trace.resample(SAMPLE_RATE)
        trace.detrend("constant")
        trace.detrend("linear")
        trace.data *= cosine_taper(trace.stats.npts, 0.05)
        trace.filter("highpass", freq=HIGHPASS_FREQ, corners=4, zerophase=True)

        try:
            cut_start = trace.stats.starttime + offset
            cut_end = cut_start + WAVEFORM_LENGTH
            cut_data = trace.slice(cut_start, cut_end).data
            if len(cut_data) < WAVEFORM_POINTS:
                cut_data = np.pad(cut_data, (0, WAVEFORM_POINTS - len(cut_data)), "constant")
            else:
                cut_data = cut_data[:WAVEFORM_POINTS]
        except:
            cut_data = np.zeros(WAVEFORM_POINTS, dtype=np.float32)

        max_amp = np.max(np.abs(cut_data))
        if max_amp != 0:
            cut_data /= max_amp
        processed_comps.append(cut_data)

    return np.stack(processed_comps, axis=-1)


# -------------------------- 6. 事件级别平衡（按震级分箱） --------------------------
def balance_events_by_mag_bin(df):
    """
    按震级分箱进行事件级别平衡
    返回平衡后的事件ID列表
    """
    # 按事件分组，每个事件取第一条记录
    event_first_traces = df.groupby("event_id").first().reset_index()

    # 分别统计地震和爆炸事件
    earthquake_events = event_first_traces[event_first_traces["event_type"] == "earthquake"]
    explosion_events = event_first_traces[event_first_traces["event_type"] == "explosion"]

    print(f"\n 事件平衡前统计：")
    print(f"  - 地震事件：{len(earthquake_events)}个")
    print(f"  - 爆炸事件：{len(explosion_events)}个")

    # 按震级分箱平衡
    balanced_earthquake_events = []
    balanced_explosion_events = []

    for mag_bin in ["0-1", "1-2", "2-3", "3-10"]:
        eq_bin_events = earthquake_events[earthquake_events["mag_bin"] == mag_bin]
        ex_bin_events = explosion_events[explosion_events["mag_bin"] == mag_bin]

        min_count = min(len(eq_bin_events), len(ex_bin_events))

        if min_count == 0:
            print(f"⚠️  震级分箱 {mag_bin} 中某一类事件数量为0，跳过该分箱")
            continue

        # 从每个分箱中随机选择相同数量的事件
        if len(eq_bin_events) > 0:
            balanced_eq = eq_bin_events.sample(min_count, random_state=RANDOM_SEED)
            balanced_earthquake_events.append(balanced_eq)

        if len(ex_bin_events) > 0:
            balanced_ex = ex_bin_events.sample(min_count, random_state=RANDOM_SEED)
            balanced_explosion_events.append(balanced_ex)

        print(f"  - 震级分箱 {mag_bin}: 地震{len(balanced_eq)}个事件, 爆炸{len(balanced_ex)}个事件")

    # 合并平衡后的事件
    if balanced_earthquake_events:
        balanced_earthquake_events = pd.concat(balanced_earthquake_events, ignore_index=True)
    else:
        balanced_earthquake_events = pd.DataFrame(columns=earthquake_events.columns)

    if balanced_explosion_events:
        balanced_explosion_events = pd.concat(balanced_explosion_events, ignore_index=True)
    else:
        balanced_explosion_events = pd.DataFrame(columns=explosion_events.columns)

    balanced_events = pd.concat([balanced_earthquake_events, balanced_explosion_events], ignore_index=True)
    balanced_event_ids = balanced_events["event_id"].tolist()

    print(f"\n✅ 事件平衡后统计：")
    print(f"  - 地震事件：{len(balanced_earthquake_events)}个")
    print(f"  - 爆炸事件：{len(balanced_explosion_events)}个")
    print(f"  - 总事件：{len(balanced_events)}个")

    return balanced_event_ids


# -------------------------- 7. 训练集TRACE级别平衡和数据增强（以少的事件为基准） --------------------------
def balance_and_augment_training_traces(df, balanced_event_ids):
    """
    对训练集进行TRACE级别平衡和数据增强
    爆炸事件固定增强5倍，地震事件根据震级分箱增强以达到平衡
    """
    print(f"\n 正在进行训练集TRACE级别平衡和数据增强...")
    print(f" 爆炸事件固定增强{AUGMENT_TIMES}倍，地震事件根据震级分箱调整增强倍数以达到平衡")

    # 筛选平衡后的事件对应的所有trace
    balanced_df = df[df["event_id"].isin(balanced_event_ids)].copy()

    # 按震级分箱分别处理
    augmented_traces = []

    for mag_bin in ["0-1", "1-2", "2-3", "3-10"]:
        # 获取当前震级分箱的所有trace
        bin_traces = balanced_df[balanced_df["mag_bin"] == mag_bin]

        if len(bin_traces) == 0:
            print(f"⚠️  震级分箱 {mag_bin} 无trace，跳过")
            continue

        # 分离地震和爆炸trace
        eq_traces = bin_traces[bin_traces["event_type"] == "earthquake"]
        ex_traces = bin_traces[bin_traces["event_type"] == "explosion"]

        # 爆炸事件固定增强5倍
        ex_augmented_count = len(ex_traces) * AUGMENT_TIMES

        # 计算地震事件需要增强的倍数以达到平衡
        if len(eq_traces) > 0:
            eq_augment_factor = max(1, ex_augmented_count // len(eq_traces))
            if ex_augmented_count % len(eq_traces) != 0:
                eq_augment_factor += 1  # 向上取整确保足够数量
        else:
            eq_augment_factor = 0

        print(f"\n 震级分箱 {mag_bin}:")
        print(f"  - 原始地震trace: {len(eq_traces)}, 爆炸trace: {len(ex_traces)}")
        print(f"  - 爆炸增强{AUGMENT_TIMES}倍后: {ex_augmented_count}个")
        print(f"  - 地震需要增强{eq_augment_factor}倍以达到平衡")

        # 对爆炸trace进行增强（固定5倍）
        for _, trace_row in ex_traces.iterrows():
            for aug_idx in range(AUGMENT_TIMES):
                augmented_trace = trace_row.copy()
                augmented_trace["augment_seed"] = hash(f"{trace_row.name}_ex_{aug_idx}") % 1000000
                augmented_traces.append(augmented_trace)

        # 对地震trace进行增强（根据震级分箱计算倍数）
        for _, trace_row in eq_traces.iterrows():
            for aug_idx in range(eq_augment_factor):
                augmented_trace = trace_row.copy()
                augmented_trace["augment_seed"] = hash(f"{trace_row.name}_eq_{aug_idx}") % 1000000
                augmented_traces.append(augmented_trace)

    # 转换为DataFrame
    augmented_df = pd.DataFrame(augmented_traces)

    # 如果列名重复，重置索引
    if 'index' in augmented_df.columns:
        augmented_df = augmented_df.reset_index(drop=True)

    # 统计增强后的分布
    print(f"\n✅ 训练集TRACE级别平衡和数据增强完成：")
    print(f"  - 增强后总trace数：{len(augmented_df)}")

    total_eq = len(augmented_df[augmented_df["event_type"] == "earthquake"])
    total_ex = len(augmented_df[augmented_df["event_type"] == "explosion"])
    print(f"  - 总地震trace: {total_eq}, 总爆炸trace: {total_ex}")

    for mag_bin in ["0-1", "1-2", "2-3", "3-10"]:
        bin_traces = augmented_df[augmented_df["mag_bin"] == mag_bin]
        if len(bin_traces) > 0:
            eq_count = len(bin_traces[bin_traces["event_type"] == "earthquake"])
            ex_count = len(bin_traces[bin_traces["event_type"] == "explosion"])
            balance_ratio = min(eq_count, ex_count) / max(eq_count, ex_count) if max(eq_count, ex_count) > 0 else 0
            print(f"  - 震级分箱 {mag_bin}: 地震{eq_count}, 爆炸{ex_count}, 平衡比: {balance_ratio:.3f}")

    return augmented_df


# -------------------------- 8. HDF5增量写入 --------------------------
def save_dataset_with_hdf5(df, save_dir, is_train=True):
    os.makedirs(save_dir, exist_ok=True)
    augment_times = AUGMENT_TIMES if is_train else 1

    # 输出文件路径
    hdf5_path = os.path.join(save_dir, "dataset.h5")
    metadata_path = os.path.join(save_dir, "metadata.csv")
    event_ids_path = os.path.join(save_dir, "event_ids.npy")

    # 对数据集进行平衡
    print(f"\n⚖️  正在平衡{'训练集' if is_train else '验证/测试集'}...")

    if is_train:
        # 训练集：先事件级平衡，再TRACE级平衡和数据增强
        balanced_event_ids = balance_events_by_mag_bin(df)
        balanced_df = balance_and_augment_training_traces(df, balanced_event_ids)
    else:
        # 验证/测试集：只进行事件级平衡，保留所有trace
        balanced_event_ids = balance_events_by_mag_bin(df)
        balanced_df = df[df["event_id"].isin(balanced_event_ids)].copy()

    # 保存平衡后的元数据
    balanced_df.to_csv(metadata_path, index=False, encoding="utf-8")

    total_events = balanced_df["event_id"].nunique()
    total_traces = len(balanced_df)
    print(f"📄 已保存元数据：{metadata_path}（{total_traces}个trace，{total_events}个事件）")

    # 初始化HDF5文件
    with h5py.File(hdf5_path, "w") as hf:
        hf.create_dataset(
            "waveforms",
            shape=(0, WAVEFORM_POINTS, 3),
            dtype=np.float32,
            maxshape=(None, WAVEFORM_POINTS, 3),
            chunks=True
        )
        hf.create_dataset(
            "labels",
            shape=(0,),
            dtype=np.int8,
            maxshape=(None,),
            chunks=True
        )
        # 保存当前子集的事件ID列表
        subset_event_ids = np.array(balanced_df["event_id"].unique(), dtype="S")
        hf.create_dataset("subset_event_ids", data=subset_event_ids)

    # 收集所有事件ID
    all_event_ids = []

    # 分批处理并增量写入
    current_offset = 0

    # 准备要处理的trace列表
    trace_groups = [(idx, row) for idx, row in balanced_df.iterrows()]
    total_traces_to_process = len(trace_groups)

    for batch_start in range(0, total_traces_to_process, BATCH_SIZE):
        batch_traces = trace_groups[batch_start:batch_start + BATCH_SIZE]
        if not batch_traces:
            break
        batch_waveforms = []
        batch_labels = []
        batch_event_ids_list = []

        with h5py.File(WAVEFORM_HDF5_PATH, "r") as src_hf:
            for idx, trace_row in batch_traces:
                event_id = trace_row["event_id"]
                label = 0 if trace_row["event_type"] == "earthquake" else 1
                p_arrival = trace_row["p_arrival_time"]

                try:
                    traces = read_waveform_from_hdf5(
                        src_hf,
                        bucket_name=trace_row["bucket"],
                        tindex=trace_row["tindex"],
                        sampling_rate=trace_row["trace_sampling_rate_hz"],
                        start_time=trace_row["trace_start_time"]
                    )
                except Exception as e:
                    print(f"⚠️  跳过事件{event_id}：{e}")
                    continue

                # 数据增强 - 对每个trace进行增强
                if is_train and "augment_seed" in trace_row:
                    # 训练集使用预计算的增强种子
                    random_seed = trace_row["augment_seed"]
                else:
                    # 验证/测试集或没有预计算种子的情况
                    random_seed = hash(f"{event_id}_{idx}") % 1000000

                wave = preprocess_waveform(traces, p_arrival, random_seed)
                batch_waveforms.append(wave)
                batch_labels.append(label)
                batch_event_ids_list.append(event_id)

                # 打印进度
                current_idx = batch_start + len(batch_waveforms)
                if current_idx % 100 == 0 or current_idx >= total_traces_to_process:
                    print(f"🔄 已处理trace：{current_idx}/{total_traces_to_process}（{os.path.basename(save_dir)}）")

        # 增量写入HDF5
        if batch_waveforms:
            batch_waveforms = np.array(batch_waveforms, dtype=np.float32)
            batch_labels = np.array(batch_labels, dtype=np.int8)
            batch_size = len(batch_waveforms)

            with h5py.File(hdf5_path, "a") as hf:
                hf["waveforms"].resize(current_offset + batch_size, axis=0)
                hf["labels"].resize(current_offset + batch_size, axis=0)
                hf["waveforms"][current_offset:current_offset + batch_size] = batch_waveforms
                hf["labels"][current_offset:current_offset + batch_size] = batch_labels

            current_offset += batch_size
            all_event_ids.extend(batch_event_ids_list)
            del batch_waveforms, batch_labels  # 释放内存

    # 保存事件ID
    np.save(event_ids_path, np.array(all_event_ids, dtype=str))

    # 最终统计和平衡性验证
    with h5py.File(hdf5_path, "r") as hf:
        total_waveforms = hf["waveforms"].shape[0]
        labels = hf["labels"][:]

    earthquake_count = np.sum(labels == 0)
    explosion_count = np.sum(labels == 1)

    print(f"\n✅ {os.path.basename(save_dir)} 保存完成：")
    print(f"   - 事件数：{total_events}个")
    print(f"   - 总波形数：{total_waveforms}个")
    if is_train:
        print(f"   - 爆炸事件固定增强：{AUGMENT_TIMES}倍")
        print(f"   - 地震事件动态增强：根据震级分箱调整")
    print(f"   - 地震波形：{earthquake_count}个 ({earthquake_count / total_waveforms * 100:.1f}%)")
    print(f"   - 爆炸波形：{explosion_count}个 ({explosion_count / total_waveforms * 100:.1f}%)")

    # 检查平衡性
    balance_ratio = min(earthquake_count, explosion_count) / max(earthquake_count, explosion_count)
    if is_train and balance_ratio < 0.95:  # 训练集要求严格平衡
        print(f"⚠️  警告：训练集不平衡，平衡比：{balance_ratio:.3f}")
    elif not is_train and balance_ratio < 0.8:  # 验证/测试集允许一定不平衡
        print(f"⚠️  警告：验证/测试集严重不平衡，平衡比：{balance_ratio:.3f}")
    else:
        print(f"✅ 数据集平衡性良好，平衡比：{balance_ratio:.3f}")

    print(f"   - HDF5路径：{hdf5_path}")


# -------------------------- 9. 主函数 --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" 开始 ComCat 数据集处理（震级分箱平衡）")
    print("=" * 60)

    try:
        print("\n【1/3】 加载元数据...")
        metadata_df = load_metadata(METADATA_PATH)

        print("\n【2/3】  事件级时间分割数据集（无泄露）...")
        train_df, val_df, test_df = split_train_test_val(metadata_df)

        print("\n【3/3】 处理并保存数据集...")
        print("\n  处理训练集（事件平衡 + TRACE级别平衡 + 爆炸5倍增强 + 地震动态增强）...")
        save_dataset_with_hdf5(train_df, os.path.join(OUTPUT_ROOT, "train"), is_train=True)

        print("\n  处理验证集（事件平衡，保留所有TRACE，无增强）...")
        save_dataset_with_hdf5(val_df, os.path.join(OUTPUT_ROOT, "val"), is_train=False)

        print("\n  处理测试集（事件平衡，保留所有TRACE，无增强）...")
        save_dataset_with_hdf5(test_df, os.path.join(OUTPUT_ROOT, "test"), is_train=False)

        print("\n" + "=" * 60)
        print(" 所有处理完成！")
        print(f" 最终数据存储路径：{OUTPUT_ROOT}")
        print(" 已通过震级分箱平衡确保所有数据集平衡")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 处理失败：{str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)
