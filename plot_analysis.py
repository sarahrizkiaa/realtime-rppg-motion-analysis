import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hasil_eksperimen_rppg.csv")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot 1: Estimated Heart Rate (BPM)
axes[0].plot(df["Detik_Ke"], df["BPM"], color="#1f77b4", marker="o", markersize=3, label="Estimated HR (BPM)")
axes[0].set_ylabel("Heart Rate (BPM)")
axes[0].set_title("Impact of Head Motion on Webcam rPPG Heart Rate & Signal Quality")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Signal-to-Noise Ratio (SNR)
axes[1].plot(df["Detik_Ke"], df["SNR_dB"], color="#2ca02c", linewidth=1.8, label="BVP Spectral SNR (dB)")
axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_ylabel("SNR (dB)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Plot 3: Head Motion Score
axes[2].fill_between(df["Detik_Ke"], df["Motion_Px"], color="#ff7f0e", alpha=0.4, label="Head Motion (px/frame)")
axes[2].plot(df["Detik_Ke"], df["Motion_Px"], color="#ff7f0e", linewidth=1.5)
axes[2].set_ylabel("Motion (px)")
axes[2].set_xlabel("Time (seconds)")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig("rppg_motion_analysis.png", dpi=300)
print("Grafik berhasil disimpan sebagai 'rppg_motion_analysis.png'!")