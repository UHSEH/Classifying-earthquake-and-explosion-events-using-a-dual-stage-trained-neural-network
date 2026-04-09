import os
import random
import numpy as np
import pandas as pd
import h5py
from obspy import UTCDateTime
from obspy.signal.invsim import cosine_taper
from obspy.core.trace import Trace, Stats

# -------------------------- 1. Basic Configuration --------------------------
METADATA_PATH = "/home/he/PycharmProjects/PythonProject/dataset/PNW-ML/comcat_metadata.csv"
WAVEFORM_HDF5_PATH = "/home/he/PycharmProjects/PythonProject/dataset/PNW-ML/comcat_waveforms.hdf5"
OUTPUT_ROOT = "/home/he/PycharmProjects/PythonProject/dataset/processed_comcat"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Key parameters
TRAIN_TEST_SPLIT = 0.9
VAL_SPLIT_FROM_TRAIN = 0.2
AUGMENT_TIMES = 5  # Explosion events are augmented by a fixed factor of 5
SAMPLE_RATE = 100
WAVEFORM_LENGTH = 60
WAVEFORM_POINTS = WAVEFORM_LENGTH * SAMPLE_RATE + 1  # 6001
HIGHPASS_FREQ = 2
BATCH_SIZE = 800
RANDOM_SEED = 42  # Fixed seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------- 2. Load Metadata --------------------------
def load_metadata(metadata_path):
    metadata = pd.read_csv(metadata_path)

    required_cols = [
        "event_id", "source_type", "preferred_source_magnitude",
        "source_origin_time", "trace_start_time", "trace_P_arrival_sample",
        "trace_sampling_rate_hz", "trace_name"
    ]
    missing_cols = [col for col in required_cols if col not in metadata.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Parse trace_name to extract bucket and tindex
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
    print(f"Trace name parsing: {len(metadata)} succeeded, {invalid_count} failed")

    # Filter valid events
    metadata = metadata[metadata["source_type"].isin(["earthquake", "explosion"])].copy()
    metadata["mag"] = pd.to_numeric(metadata["preferred_source_magnitude"], errors="coerce")
    metadata = metadata[(metadata["mag"].notna()) & (metadata["mag"] >= 0) & (metadata["mag"] <= 10)].copy()
    metadata = metadata[
        (metadata["trace_P_arrival_sample"] >= 0) &
        (metadata["trace_P_arrival_sample"] < metadata["trace_sampling_rate_hz"] * 150)
        ].copy()

    # Compute P-wave arrival time
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

    # Rename columns
    metadata = metadata.rename(columns={"source_type": "event_type", "source_origin_time": "origin_time"})

    # Add magnitude bins
    metadata["mag_bin"] = np.select(
        [(metadata["mag"] >= 0) & (metadata["mag"] < 1),
         (metadata["mag"] >= 1) & (metadata["mag"] < 2),
         (metadata["mag"] >= 2) & (metadata["mag"] < 3),
         (metadata["mag"] >= 3) & (metadata["mag"] <= 10)],
        ["0-1", "1-2", "2-3", "3-10"], default="other"
    )

    # Statistics of raw data
    explosion_traces = metadata[metadata["event_type"] == "explosion"]
    earthquake_traces = metadata[metadata["event_type"] == "earthquake"]

    print(f"\nRaw data statistics:")
    print(f"  - Earthquake traces: {len(earthquake_traces)}")
    print(f"  - Explosion traces: {len(explosion_traces)}")
    print(f"  - Earthquake events: {earthquake_traces['event_id'].nunique()}")
    print(f"  - Explosion events: {explosion_traces['event_id'].nunique()}")

    # Magnitude bin statistics
    print(f"\nMagnitude bin statistics:")
    for event_type in ["earthquake", "explosion"]:
        type_traces = metadata[metadata["event_type"] == event_type]
        print(f"  - {event_type}:")
        for mag_bin in ["0-1", "1-2", "2-3", "3-10"]:
            bin_count = len(type_traces[type_traces["mag_bin"] == mag_bin])
            if bin_count > 0:
                print(f"    {mag_bin}: {bin_count} traces")

    return metadata


# -------------------------- 3. Event-Level Temporal Split --------------------------
def split_train_test_val(metadata_df):
    # Generate event-level dataframe
    event_level_df = metadata_df.groupby("event_id").agg({
        "origin_time": "first",
        "event_type": "first",
        "mag": "first",
        "mag_bin": "first",
        "trace_name": "count"
    }).rename(columns={"trace_name": "trace_count"}).reset_index()

    # Sort by event origin time
    event_level_df = event_level_df.sort_values("origin_time").reset_index(drop=True)
    total_events = len(event_level_df)
    total_traces = len(metadata_df)

    print(f"\nEvent-level statistics (before split):")
    print(f"  - Total events: {total_events}")
    print(f"  - Total traces: {total_traces}")
    print(f"  - Average traces per event: {total_traces / total_events:.2f}")

    # Split Train/Test event IDs
    train_event_num = int(total_events * TRAIN_TEST_SPLIT)
    train_event_ids = set(event_level_df.iloc[:train_event_num]["event_id"].tolist())
    test_event_ids = set(event_level_df.iloc[train_event_num:]["event_id"].tolist())

    # Split validation event IDs from training events
    train_events_subset = event_level_df[event_level_df["event_id"].isin(train_event_ids)]
    val_event_num = int(len(train_events_subset) * VAL_SPLIT_FROM_TRAIN)
    val_event_ids = set(train_events_subset.sample(val_event_num, random_state=RANDOM_SEED)["event_id"].tolist())
    train_event_ids = train_event_ids - val_event_ids

    # Verify no event ID overlap across subsets
    overlap_train_val = train_event_ids & val_event_ids
    overlap_train_test = train_event_ids & test_event_ids
    overlap_val_test = val_event_ids & test_event_ids
    assert len(overlap_train_val) == 0, f"Overlap between Train and Val event IDs: {list(overlap_train_val)[:5]}..."
    assert len(overlap_train_test) == 0, f"Overlap between Train and Test event IDs: {list(overlap_train_test)[:5]}..."
    assert len(overlap_val_test) == 0, f"Overlap between Val and Test event IDs: {list(overlap_val_test)[:5]}..."
    print(f"No event ID overlap across subsets; no data leakage.")

    # Assign all traces of each event to corresponding subset
    train_df = metadata_df[metadata_df["event_id"].isin(train_event_ids)].copy().reset_index(drop=True)
    val_df = metadata_df[metadata_df["event_id"].isin(val_event_ids)].copy().reset_index(drop=True)
    test_df = metadata_df[metadata_df["event_id"].isin(test_event_ids)].copy().reset_index(drop=True)

    # Helper function to compute subset statistics
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

    # Generate statistics
    train_stats = count_subset_stats(train_df, "Train")
    val_stats = count_subset_stats(val_df, "Val")
    test_stats = count_subset_stats(test_df, "Test")

    # Print split results
    print(f"\nEvent-level temporal split results:")
    print("  " + "-" * 120)
    print(f"  {'Subset':<8} {'Events(Total/Eq/Ex)':<22} {'Event%':<10} {'Traces(Total/Eq/Ex)':<22} {'Trace%':<10}")
    print("  " + "-" * 120)
    for stats in [train_stats, val_stats, test_stats]:
        event_str = f"{stats['total_event']}/{stats['eq_event']}/{stats['ex_event']}"
        trace_str = f"{stats['total_trace']}/{stats['eq_trace']}/{stats['ex_trace']}"
        print(
            f"  {stats['subset']:<8} {event_str:<22} {stats['event_ratio']:<10.1f} {trace_str:<22} {stats['trace_ratio']:<10.1f}")
    print("  " + "-" * 120)
    print(
        f"  {'Total':<8} {total_events}/{event_level_df[event_level_df['event_type'] == 'earthquake'].shape[0]}/{event_level_df[event_level_df['event_type'] == 'explosion'].shape[0]:<12} 100.0{'':<10} {total_traces}/{len(metadata_df[metadata_df['event_type'] == 'earthquake'])}/{len(metadata_df[metadata_df['event_type'] == 'explosion']):<12} 100.0")

    return train_df, val_df, test_df


# -------------------------- 4. Read Waveform from HDF5 --------------------------
def read_waveform_from_hdf5(hdf5_file, bucket_name, tindex, sampling_rate, start_time):
    waveform_traces = []
    required_comps = ["Z", "N", "E"]
    bucket_path = f"data/{bucket_name}"

    if bucket_path not in hdf5_file:
        raise ValueError(f"Bucket not found in HDF5: {bucket_path}")

    bucket_dataset = hdf5_file[bucket_path]
    n_events = bucket_dataset.shape[0]
    if tindex < 0 or tindex >= n_events:
        raise ValueError(f"tindex {tindex} out of range (0~{n_events - 1})")

    wave_data = bucket_dataset[tindex]
    if wave_data.shape[0] != 3:
        raise ValueError(f"Unexpected number of components: {wave_data.shape[0]} (expected 3)")

    comp_order = hdf5_file["data_format/component_order"][()].decode('utf-8')
    comp_map = {c: i for i, c in enumerate(comp_order)}
    missing = [c for c in required_comps if c not in comp_map]
    if missing:
        raise ValueError(f"Missing components: {missing} (order: {comp_order})")

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


# -------------------------- 5. Waveform Preprocessing --------------------------
def preprocess_waveform(waveform_traces, p_arrival_time, random_seed=None):
    processed_comps = []

    try:
        p_utc = UTCDateTime(p_arrival_time)
        p_ts = p_utc.timestamp
    except:
        print("Warning: Invalid P arrival time, zero-padding")
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


# -------------------------- 6. Event-Level Balancing by Magnitude Bin --------------------------
def balance_events_by_mag_bin(df):
    """
    Perform event-level balancing based on magnitude bins.
    Returns a list of balanced event IDs.
    """
    # Group by event and take the first record per event
    event_first_traces = df.groupby("event_id").first().reset_index()

    # Separate earthquake and explosion events
    earthquake_events = event_first_traces[event_first_traces["event_type"] == "earthquake"]
    explosion_events = event_first_traces[event_first_traces["event_type"] == "explosion"]

    print(f"\nBefore event balancing:")
    print(f"  - Earthquake events: {len(earthquake_events)}")
    print(f"  - Explosion events: {len(explosion_events)}")

    balanced_earthquake_events = []
    balanced_explosion_events = []

    for mag_bin in ["0-1", "1-2", "2-3", "3-10"]:
        eq_bin_events = earthquake_events[earthquake_events["mag_bin"] == mag_bin]
        ex_bin_events = explosion_events[explosion_events["mag_bin"] == mag_bin]

        min_count = min(len(eq_bin_events), len(ex_bin_events))

        if min_count == 0:
            print(f"Warning: Magnitude bin {mag_bin} has zero events for one class, skipping")
            continue

        # Randomly sample equal number of events from each class within the bin
        if len(eq_bin_events) > 0:
            balanced_eq = eq_bin_events.sample(min_count, random_state=RANDOM_SEED)
            balanced_earthquake_events.append(balanced_eq)

        if len(ex_bin_events) > 0:
            balanced_ex = ex_bin_events.sample(min_count, random_state=RANDOM_SEED)
            balanced_explosion_events.append(balanced_ex)

        print(f"  - Magnitude bin {mag_bin}: Earthquake {len(balanced_eq)} events, Explosion {len(balanced_ex)} events")

    # Concatenate balanced events
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

    print(f"\nAfter event balancing:")
    print(f"  - Earthquake events: {len(balanced_earthquake_events)}")
    print(f"  - Explosion events: {len(balanced_explosion_events)}")
    print(f"  - Total events: {len(balanced_events)}")

    return balanced_event_ids


# -------------------------- 7. Training Set Trace-Level Balancing and Augmentation --------------------------
def balance_and_augment_training_traces(df, balanced_event_ids):
    """
    Perform trace-level balancing and data augmentation for training set.
    Explosion events are augmented by a fixed factor of AUGMENT_TIMES.
    Earthquake events are augmented per magnitude bin to match the number of augmented explosion traces.
    """
    print(f"\nPerforming training set trace-level balancing and augmentation...")
    print(f"Explosion events fixed augmentation: {AUGMENT_TIMES}x; earthquake events dynamically adjusted per magnitude bin.")

    # Keep only traces belonging to balanced events
    balanced_df = df[df["event_id"].isin(balanced_event_ids)].copy()

    augmented_traces = []

    for mag_bin in ["0-1", "1-2", "2-3", "3-10"]:
        bin_traces = balanced_df[balanced_df["mag_bin"] == mag_bin]

        if len(bin_traces) == 0:
            print(f"Warning: Magnitude bin {mag_bin} has no traces, skipping")
            continue

        eq_traces = bin_traces[bin_traces["event_type"] == "earthquake"]
        ex_traces = bin_traces[bin_traces["event_type"] == "explosion"]

        # Explosion traces are augmented by AUGMENT_TIMES
        ex_augmented_count = len(ex_traces) * AUGMENT_TIMES

        # Determine augmentation factor for earthquake traces to balance the bin
        if len(eq_traces) > 0:
            eq_augment_factor = max(1, ex_augmented_count // len(eq_traces))
            if ex_augmented_count % len(eq_traces) != 0:
                eq_augment_factor += 1  # round up to ensure enough
        else:
            eq_augment_factor = 0

        print(f"\nMagnitude bin {mag_bin}:")
        print(f"  - Original earthquake traces: {len(eq_traces)}, explosion traces: {len(ex_traces)}")
        print(f"  - After explosion {AUGMENT_TIMES}x augmentation: {ex_augmented_count}")
        print(f"  - Earthquake augmentation factor: {eq_augment_factor}x")

        # Augment explosion traces
        for _, trace_row in ex_traces.iterrows():
            for aug_idx in range(AUGMENT_TIMES):
                augmented_trace = trace_row.copy()
                augmented_trace["augment_seed"] = hash(f"{trace_row.name}_ex_{aug_idx}") % 1000000
                augmented_traces.append(augmented_trace)

        # Augment earthquake traces
        for _, trace_row in eq_traces.iterrows():
            for aug_idx in range(eq_augment_factor):
                augmented_trace = trace_row.copy()
                augmented_trace["augment_seed"] = hash(f"{trace_row.name}_eq_{aug_idx}") % 1000000
                augmented_traces.append(augmented_trace)

    augmented_df = pd.DataFrame(augmented_traces)

    if 'index' in augmented_df.columns:
        augmented_df = augmented_df.reset_index(drop=True)

    print(f"\nTraining set trace-level balancing and augmentation completed:")
    print(f"  - Total traces after augmentation: {len(augmented_df)}")

    total_eq = len(augmented_df[augmented_df["event_type"] == "earthquake"])
    total_ex = len(augmented_df[augmented_df["event_type"] == "explosion"])
    print(f"  - Total earthquake traces: {total_eq}, Total explosion traces: {total_ex}")

    for mag_bin in ["0-1", "1-2", "2-3", "3-10"]:
        bin_traces = augmented_df[augmented_df["mag_bin"] == mag_bin]
        if len(bin_traces) > 0:
            eq_count = len(bin_traces[bin_traces["event_type"] == "earthquake"])
            ex_count = len(bin_traces[bin_traces["event_type"] == "explosion"])
            balance_ratio = min(eq_count, ex_count) / max(eq_count, ex_count) if max(eq_count, ex_count) > 0 else 0
            print(f"  - Magnitude bin {mag_bin}: Earthquake {eq_count}, Explosion {ex_count}, Balance ratio: {balance_ratio:.3f}")

    return augmented_df


# -------------------------- 8. HDF5 Incremental Write --------------------------
def save_dataset_with_hdf5(df, save_dir, is_train=True):
    os.makedirs(save_dir, exist_ok=True)
    augment_times = AUGMENT_TIMES if is_train else 1

    hdf5_path = os.path.join(save_dir, "dataset.h5")
    metadata_path = os.path.join(save_dir, "metadata.csv")
    event_ids_path = os.path.join(save_dir, "event_ids.npy")

    print(f"\n  Balancing {'training' if is_train else 'validation/test'} set...")

    if is_train:
        # Training set: event-level balancing then trace-level balancing and augmentation
        balanced_event_ids = balance_events_by_mag_bin(df)
        balanced_df = balance_and_augment_training_traces(df, balanced_event_ids)
    else:
        # Validation/Test set: event-level balancing only, keep all traces
        balanced_event_ids = balance_events_by_mag_bin(df)
        balanced_df = df[df["event_id"].isin(balanced_event_ids)].copy()

    # Save balanced metadata
    balanced_df.to_csv(metadata_path, index=False, encoding="utf-8")

    total_events = balanced_df["event_id"].nunique()
    total_traces = len(balanced_df)
    print(f"Metadata saved: {metadata_path} ({total_traces} traces, {total_events} events)")

    # Initialize HDF5 file
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
        subset_event_ids = np.array(balanced_df["event_id"].unique(), dtype="S")
        hf.create_dataset("subset_event_ids", data=subset_event_ids)

    all_event_ids = []
    current_offset = 0

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
                    print(f"Warning: Skipping event {event_id}: {e}")
                    continue

                if is_train and "augment_seed" in trace_row:
                    random_seed = trace_row["augment_seed"]
                else:
                    random_seed = hash(f"{event_id}_{idx}") % 1000000

                wave = preprocess_waveform(traces, p_arrival, random_seed)
                batch_waveforms.append(wave)
                batch_labels.append(label)
                batch_event_ids_list.append(event_id)

                current_idx = batch_start + len(batch_waveforms)
                if current_idx % 100 == 0 or current_idx >= total_traces_to_process:
                    print(f" Processed traces: {current_idx}/{total_traces_to_process} ({os.path.basename(save_dir)})")

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
            del batch_waveforms, batch_labels

    np.save(event_ids_path, np.array(all_event_ids, dtype=str))

    with h5py.File(hdf5_path, "r") as hf:
        total_waveforms = hf["waveforms"].shape[0]
        labels = hf["labels"][:]

    earthquake_count = np.sum(labels == 0)
    explosion_count = np.sum(labels == 1)

    print(f"\n{os.path.basename(save_dir)} saved successfully:")
    print(f"   - Number of events: {total_events}")
    print(f"   - Total waveforms: {total_waveforms}")
    if is_train:
        print(f"   - Explosion events fixed augmentation: {AUGMENT_TIMES}x")
        print(f"   - Earthquake events dynamic augmentation: adjusted per magnitude bin")
    print(f"   - Earthquake waveforms: {earthquake_count} ({earthquake_count / total_waveforms * 100:.1f}%)")
    print(f"   - Explosion waveforms: {explosion_count} ({explosion_count / total_waveforms * 100:.1f}%)")

    balance_ratio = min(earthquake_count, explosion_count) / max(earthquake_count, explosion_count)
    if is_train and balance_ratio < 0.95:
        print(f"Warning: Training set is imbalanced, balance ratio: {balance_ratio:.3f}")
    elif not is_train and balance_ratio < 0.8:
        print(f"Warning: Validation/Test set is severely imbalanced, balance ratio: {balance_ratio:.3f}")
    else:
        print(f"Dataset is well balanced, balance ratio: {balance_ratio:.3f}")

    print(f"   - HDF5 path: {hdf5_path}")


# -------------------------- 9. Main Function --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" Starting ComCat Dataset Processing (Magnitude Bin Balancing)")
    print("=" * 60)

    try:
        print("\n[1/3] Loading metadata...")
        metadata_df = load_metadata(METADATA_PATH)

        print("\n[2/3] Event-level temporal split (no data leakage)...")
        train_df, val_df, test_df = split_train_test_val(metadata_df)

        print("\n[3/3] Processing and saving datasets...")
        print("\n  Processing training set (event balancing + trace balancing + explosion 5x augmentation + earthquake dynamic augmentation)...")
        save_dataset_with_hdf5(train_df, os.path.join(OUTPUT_ROOT, "train"), is_train=True)

        print("\n  Processing validation set (event balancing, all traces kept, no augmentation)...")
        save_dataset_with_hdf5(val_df, os.path.join(OUTPUT_ROOT, "val"), is_train=False)

        print("\n  Processing test set (event balancing, all traces kept, no augmentation)...")
        save_dataset_with_hdf5(test_df, os.path.join(OUTPUT_ROOT, "test"), is_train=False)

        print("\n" + "=" * 60)
        print(" All processing completed!")
        print(f" Final data stored at: {OUTPUT_ROOT}")
        print(" All subsets balanced via magnitude bin matching.")
        print("=" * 60)
    except Exception as e:
        print(f"\nProcessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
