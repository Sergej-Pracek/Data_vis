import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib import rcParams

# Set font to Lato
rcParams['font.family'] = 'Lato'

# Load the data
file_path_5utr = "dapseq-MYC2-tfs - 5utr.tsv"
utr = pd.read_csv(file_path_5utr, sep="\t")

# Ensure required columns are present
required_columns = ["peak_distance_tss", "peak_signalValue", "peak_pValue", "tf"]
if not all(col in utr.columns for col in required_columns):
    raise ValueError(f"Missing required columns: {required_columns}")

# Calculate histogram data
hist_data = utr.groupby("peak_distance_tss")["peak_signalValue"].mean()

# Plot histogram
fig, ax = plt.subplots(figsize=(18, 8))
ax.bar(
    hist_data.index, 
    hist_data, 
    width=2, 
    color="black", 
    alpha=0.6, 
    edgecolor="black"
)

# Connect duplicate TF positions with arcs
for tf, group in utr.groupby("tf"):
    positions = group["peak_distance_tss"].values
    if len(positions) > 1:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                start, end = positions[i], positions[j]
                center, radius_x = (start + end) / 2, (end - start) / 2
                radius_y = min(18, radius_x)
                theta = np.linspace(0, np.pi, 100)
                x = center + radius_x * np.cos(theta)
                y = -radius_y * np.sin(theta)
                ax.plot(x, y, color="grey", alpha=0.7, linewidth=0.8)

# Customize plot
ax.grid(axis="y", linestyle="--", alpha=0.6, color="grey")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linestyle="--", linewidth=0.8)  # Add vertical line at x=0
ax.set_xlim(hist_data.index.min() - 30, hist_data.index.max() + 30)
ax.set_ylim(-20,39)

# Set major and minor ticks
ax.yaxis.set_major_locator(MultipleLocator(5))  # Major ticks every 10
ax.yaxis.set_minor_locator(AutoMinorLocator())   # Minor ticks automatically
ax.tick_params(axis="y", which="major", length=6, color="grey", labelsize=12)
ax.tick_params(axis="y", which="minor", length=4, color="grey")
ax.tick_params(axis="x", labelsize=12)

# Remove y-axis marks below y=0
ax.set_yticks([tick for tick in ax.get_yticks() if tick >= 0])
ax.spines['left'].set_visible(True)
# Position x-axis labels at y=0
ax.spines['bottom'].set_position(('data', 0))

plt.tight_layout()
#save figure
#plt.savefig("dapseq-MYC2-tfs-5utr.png", dpi=600)
plt.show()