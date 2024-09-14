import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pywt

# Set global font properties to "Times New Roman" and size 24
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
    'axes.titlesize': 24,       # Font size for titles
    'axes.labelsize': 24,       # Font size for x and y labels
    'xtick.labelsize': 24,      # Font size for x-axis tick labels
    'ytick.labelsize': 24       # Font size for y-axis tick labels
})

def m4a_to_wavelet_png(m4a_file, png_file, size=(15, 10)):
    """
    Takes a .m4a file, applies Continuous Wavelet Transform (CWT), and saves the wavelet scalogram as a .png file.

    Args:
        m4a_file (str): Path to the .m4a file.
        png_file (str): Path to save the .png file.
        size (tuple): Size of the output .png file (width, height).
    """

    # 1. Load the audio file (librosa can handle .m4a with ffmpeg/audioread installed)
    y, sr = librosa.load(m4a_file)

    # 2. Define scales and perform the continuous wavelet transform (CWT)
    widths = np.arange(1, 128)
    cwtmatr, freqs = pywt.cwt(y, widths, 'morl', sampling_period=1/sr)

    # 3. Create a figure and axes for plotting
    fig, ax = plt.subplots(figsize=size, dpi=300)  # Adjust figsize for desired output size

    # 4. Display the wavelet transform (scalogram) with axes and labels
    # Use a darker color map like 'inferno' and adjust vmin and vmax for stronger contrast
    ax.imshow(np.abs(cwtmatr), extent=[0, len(y)/sr, 1, 128], cmap='inferno', aspect='auto',
              vmax=abs(cwtmatr).max(), vmin=0)

    # 5. Add x and y axis labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scale (log scale)')

    # 6. Save the figure as a .png file
    fig.savefig(png_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_m4a_files_in_folder(input_folder, output_folder, size=(15, 10)):
    """
    Processes all .m4a files in a folder and saves their wavelet scalograms as .png files.

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

    # 5. Process the selected .m4a file and save the wavelet scalogram
    m4a_to_wavelet_png(input_file_path, output_file_path, size)
    print(f"Processed: {input_file_path} -> {output_file_path}")

# Usage
input_folder = "iy-code/data-all"
output_folder = "iy"
process_m4a_files_in_folder(input_folder, output_folder)
