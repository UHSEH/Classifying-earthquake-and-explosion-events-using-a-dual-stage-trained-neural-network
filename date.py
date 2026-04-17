import os
import requests

def download_with_resume(url, save_path, chunk_size=1024*1024):
    """
    File download function with resume support
    :param url: Download URL
    :param save_path: Local save path
    :param chunk_size: Chunk size (1MB to avoid excessive memory usage)
    """
    # Check if file is partially downloaded, resume from breakpoint if exists
    file_size = 0
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"Found partially downloaded file ({file_size/1024/1024:.2f} MB), will resume from breakpoint...")

    # Set request headers to specify resuming from the breakpoint position
    headers = {"Range": f"bytes={file_size}-"} if file_size > 0 else {}

    try:
        # Send request (stream=True enables chunked download, avoiding loading entire file into memory)
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()  # Raise exception if status code is not 200

        # Get total file size (if supported by server)
        total_size = int(response.headers.get("content-length", 0)) + file_size
        print(f"Starting download: {os.path.basename(save_path)}")
        print(f"Total size: {total_size/1024/1024:.2f} MB")

        # Write file in chunks (append mode to avoid overwriting downloaded part)
        with open(save_path, "ab") as f:
            downloaded = file_size  # Downloaded bytes
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out empty chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Real-time progress display (update every 1% downloaded)
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}% | Downloaded: {downloaded/1024/1024:.2f} MB", end="")

        print(f"\n{os.path.basename(save_path)} download completed!")

    except Exception as e:
        print(f"\nDownload failed: {str(e)}")
        print("Please check network or link, re-run the script to resume.")

# -------------------------- Dataset URL Configuration --------------------------
datasets = [
    # Full ComCat dataset
    ("https://dasway.ess.washington.edu/shared/niyiyu/PNW-ML/comcat_waveforms.hdf5", "comcat_waveforms.hdf5"),

]

# -------------------------- Execute Download --------------------------
if __name__ == "__main__":
    # Save directory (consistent with method 1, customizable)
    save_dir = "dataset/PNW-ML"
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    # Batch download all files
    for url, filename in datasets:
        save_path = os.path.join(save_dir, filename)
        download_with_resume(url, save_path)