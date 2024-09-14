import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Set global font properties to "Times New Roman" and size 24
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
    'axes.titlesize': 24,       # Font size for titles
    'axes.labelsize': 24,       # Font size for x and y labels
    'xtick.labelsize': 24,      # Font size for x-axis tick labels
    'ytick.labelsize': 24       # Font size for y-axis tick labels
})

def m4a_to_cqt_png(m4a_file, png_file, size=(15, 10)):
    """
    Takes a .m4a file, computes its CQT, and saves the constant-Q spectrogram as a .png file with x and y axes and labels.

    Args:
        m4a_file (str): Path to the .m4a file.
        png_file (str): Path to save the .png file.
        size (tuple): Size of the output .png file (width, height).
    """

    # 1. Load the audio file (librosa can handle .m4a with ffmpeg/audioread installed)
    y, sr = librosa.load(m4a_file)

    # 2. Perform Constant-Q Transform (CQT)
    cqt_result = librosa.cqt(y, sr=sr)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt_result))

    # 3. Create a figure and axes for plotting
    fig, ax = plt.subplots(figsize=size, dpi=300)  # Adjust figsize for desired output size

    # 4. Display the CQT result
    img = librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_hz', ax=ax)

    # 5. Add x and y axis labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    # 6. Save the figure as a .png file
    fig.savefig(png_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_m4a_files_in_folder(input_folder, output_folder, size=(12, 8)):
    """
    Processes all .m4a files in a folder and saves their CQT spectrums as .png files.

    Args:
        input_folder (str): Path to the folder containing .m4a files.
        output_folder (str): Path to save the .png files.
        size (tuple): Size of the output .png files (width, height).
    """

    # 1. Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # 2. Collect all .m4a files in the folder
    m4a_files = [f for f in os.listdir(input_folder) if f.endswith(".m4a")]

    # 3. Limit to files 1-10, if more exist
    if len(m4a_files) > 10:
        m4a_files = m4a_files[:10]

    # 4. Randomly select one of the first 10 files
    selected_file = random.choice(m4a_files)

    input_file_path = os.path.join(input_folder, selected_file)
    output_file_name = os.path.splitext(selected_file)[0] + ".png"  # Replace .m4a with .png
    output_file_path = os.path.join(output_folder, output_file_name)

    # 5. Process the selected .m4a file and save the CQT spectrum
    m4a_to_cqt_png(input_file_path, output_file_path, size)
    print(f"Processed: {input_file_path} -> {output_file_path}")

# Usage
input_folder = "iy-code/data-all"
output_folder = "iy"
process_m4a_files_in_folder(input_folder, output_folder)
