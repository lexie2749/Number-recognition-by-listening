import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from scipy.optimize import curve_fit

# 设置全局字体为 Times New Roman 和字体大小为 24
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24

# Define the path to the folder containing M4A files
m4a_folder_path = 'iy-code/data-all'

# Dictionary to store the energy for each I value
energy_dict = defaultdict(list)

# Function to calculate the energy of each frame in the audio data (normal energy, not RMS)
def calculate_energy(wave_data, frame_size):
    """计算每帧的能量（非均方根能量）"""
    energy = []
    for i in range(0, len(wave_data), frame_size):
        frame = wave_data[i:i + frame_size]
        e = np.sum(np.square(frame))  # Calculate the energy of each frame
        energy.append(e)
    return energy

# Read all M4A files in the folder
for filename in os.listdir(m4a_folder_path):
    if filename.endswith('.m4a'):
        filepath = os.path.join(m4a_folder_path, filename)
        
        # Load the audio file
        y, sr = librosa.load(filepath, sr=None)

        # Calculate the energy for each frame
        frame_size = 256  # Frame size for calculating energy
        energy = calculate_energy(y, frame_size)
        
        # Calculate the average energy
        avg_energy = np.mean(energy)
        print(f"\nFilename: {filename}")
        print(f"Average Energy: {avg_energy:.2f}")
        
        # Extract the I value from the filename
        i_value = filename.split('-')[0]
        
        # Add the average energy to the corresponding I value list
        energy_dict[i_value].append(avg_energy)

# Filter and calculate the average energy for I values between 0 and 30
average_energies = {i: np.mean(energies) for i, energies in energy_dict.items() if 0 <= float(i) <= 30}
errors_energy = {i: np.std(energies) for i, energies in energy_dict.items() if 0 <= float(i) <= 30}  # Calculate standard deviation for error bars

# Initialize lists for plotting, skipping missing values
sorted_i_values = [i for i in range(0, 31) if str(i) in average_energies]  # Only include i with data
sorted_average_energies = [average_energies[str(i)] for i in sorted_i_values]
sorted_errors_energy = [errors_energy[str(i)] for i in sorted_i_values]

def polynomial_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d  # 三次多项式

# Perform the polynomial fitting (you can extend to higher degrees by modifying the function and the number of parameters)
popt_poly, pcov_poly = curve_fit(polynomial_func, sorted_i_values, sorted_average_energies)

# Plot the scatter plot with error bars and the fitted polynomial line
plt.figure(figsize=(20, 8))
plt.errorbar(sorted_i_values, sorted_average_energies, yerr=sorted_errors_energy, fmt='o', color='blue', ecolor='blue', capsize=0, label='Data')

# Plot the polynomial fitted line
x_fit = np.linspace(min(sorted_i_values), max(sorted_i_values), 100)
y_fit_poly = polynomial_func(x_fit, *popt_poly)
plt.plot(x_fit, y_fit_poly, 'r-', label=f'Polynomial fit: y = {popt_poly[0]:.2e} * x^2 + {popt_poly[1]:.2e} * x + {popt_poly[2]:.2e}')

# Set font size to 24 for plot settings, and set Times New Roman font
plt.xlabel('The number of balls', fontsize=24, fontname='Times New Roman')
plt.ylabel('Average Energy', fontsize=24, fontname='Times New Roman')

# Set x-ticks to every 5 units from 0 to 30
plt.xticks(np.arange(0, 31, 5), fontsize=24, fontname='Times New Roman')  # Set ticks from 0 to 30, step size of 5
plt.yticks(fontsize=24, fontname='Times New Roman')

# Display the plot
plt.legend(loc='upper right', fontsize=24)
plt.grid(False)
plt.gca().set_facecolor('white')
plt.show()

# Output the fitted polynomial formula
print(f'Polynomial fit formula: y = {popt_poly[0]:.2e} * x^2 + {popt_poly[1]:.2e} * x + {popt_poly[2]:.2e}')
