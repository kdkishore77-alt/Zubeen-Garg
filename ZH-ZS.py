"""
=============================================================
Acoustic Comparison of Zubeen Garg's Pure Humming vs  Singing
=============================================================
Quantifies and visualizes the spectral and temporal distinctiveness
of Zubeen Garg’s humming using the same AcousticHummingAnalysis framework.
=============================================================
"""

import os
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})

# --------------------------------------------------------------
# Acoustic Feature Extraction Utilities (refined + consistent)
# --------------------------------------------------------------

def extract_features(file_path):
    """Extracts acoustic features from a single audio file."""
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)

    # --- Stable region detection ---
    f0 = librosa.yin(y, fmin=80, fmax=600, sr=sr)
    voiced_idx = np.where(~np.isnan(f0))[0]
    if len(voiced_idx) == 0:
        return None

    f0_stable = f0[voiced_idx]
    stability_mask = np.abs(np.diff(f0_stable)) < 0.05 * np.mean(f0_stable)
    stable_idx = voiced_idx[:-1][stability_mask]
    if len(stable_idx) == 0:
        stable_idx = voiced_idx
    y_stable = y[stable_idx[0]:stable_idx[-1]]

    # --- Pad short clips ---
    if len(y_stable) < 2048:
        y_stable = np.pad(y_stable, (0, 2048 - len(y_stable)), mode="reflect")

    # --- Temporal stability metrics ---
    F0_mean = np.nanmean(f0_stable)
    jitter = np.std(np.diff(f0_stable)) / np.mean(f0_stable)
    rms = librosa.feature.rms(y=y_stable)[0]
    shimmer = np.std(rms) / (np.mean(rms) + 1e-10)
    harmonic = librosa.effects.harmonic(y_stable)
    hnr_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(y_stable)) + 1e-10)

    # --- Spectral features ---
    S = np.abs(librosa.stft(y_stable, n_fft=2048, hop_length=512))
    centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(S=S))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mags = np.mean(S, axis=1)
    mags = np.maximum(mags, 1e-10)
    slope, _ = np.polyfit(np.log10(freqs[1:]), 20 * np.log10(mags[1:]), 1)
    tilt = slope
    roughness = 1 - flatness  # perceptual proxy

    # --- Timbre and resonance features ---
    mfccs = librosa.feature.mfcc(y=y_stable, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    formants = librosa.lpc(y_stable, order=14)
    energy_low = np.sum(S[(freqs <= 1000)])
    energy_high = np.sum(S[(freqs >= 2000) & (freqs <= 4000)])
    brightness_ratio = energy_high / (energy_low + 1e-6)

    # --- Assemble features ---
    features = {
        'F0_mean': F0_mean,
        'jitter': jitter,
        'shimmer': shimmer,
        'HNR': hnr_ratio,
        'spectral_centroid': centroid,
        'spectral_rolloff': rolloff,
        'spectral_tilt': tilt,
        'spectral_roughness': roughness,
        'brightness_ratio': brightness_ratio,
    }
    for i, m in enumerate(mfcc_means, start=1):
        features[f'MFCC_{i}'] = m
    return features


# --------------------------------------------------------------
# Main Comparative Analysis
# --------------------------------------------------------------

def main():
    zubeen_folder = "data/Zubeen-Humming-full"     # contains PH1.wav, PH2.wav, ...
    sung_folder  = "data/Zubeen-Singing"           # contains PS1.wav, PS2.wav, ...

    zubeen_humming = [os.path.join(zubeen_folder, f) for f in os.listdir(zubeen_folder) if f.endswith(".wav")]
    zubeen_singing = [os.path.join(sung_folder, f) for f in os.listdir(sung_folder) if f.endswith(".wav")]

    data = []
    for f in zubeen_humming:
        feats = extract_features(f)
        if feats:
            feats["label"] = "Humming"
            data.append(feats)
    for f in zubeen_singing:
        feats = extract_features(f)
        if feats:
            feats["label"] = "Singing"
            data.append(feats)

    df = pd.DataFrame(data).dropna(axis=1, how="any")
    print(f"\nExtracted features from {len(df)} files")
    print(df.head())
    # --- Feature name mapping for publication-ready plots ---
    feature_labels = {
        'F0_mean': 'Mean F0',
        'jitter': 'Jitter',
        'shimmer': 'Shimmer',
        'HNR': 'HNR Ratio',
        'spectral_centroid': 'Spectral Centroid',
        'spectral_rolloff': 'Spectral Rolloff',
        'spectral_tilt': 'Spectral Tilt',
        'spectral_roughness': 'Spectral Roughness',
        'brightness_ratio': 'Brightness Ratio',
        **{f'MFCC_{i}': f'MFCC {i}' for i in range(1, 14)}

    }
    
    # ----------------------------------------------------------
    # Statistical comparison
    # ----------------------------------------------------------
    z = df[df["label"] == "Humming"]
    o = df[df["label"] == "Singing"]
    common_features = [c for c in df.columns if c != "label"]

    stats_results = []
    for feat in common_features:
        x = z[feat].dropna()
        y = o[feat].dropna()
        if len(x) > 1 and len(y) > 1:
            stat, pval = mannwhitneyu(x, y, alternative='two-sided')
        else:
            pval = np.nan
            print(f"Skipped {feat} — insufficient data: Humming={len(x)}, Singing={len(y)}")

        pretty_name = feature_labels.get(feat, feat.replace("_", " ").title())
        stats_results.append({"Feature": pretty_name, "p_value": pval})


    stats_df = pd.DataFrame(stats_results).sort_values("p_value")
    print("\n--- Statistical significance summary ---")
    print(stats_df.head(10))


    # ----------------------------------------------------------
    # Feature correlation heatmap
    # ----------------------------------------------------------
    corr = df[common_features].corr()
    corr.rename(columns=feature_labels, index=feature_labels, inplace=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title("Feature Correlation Map: Humming vs Singing", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('HS-feature-correlation.png', dpi=600, bbox_inches='tight')
    plt.close()


    # ----------------------------------------------------------
    # PCA visualization (multidimensional timbre clustering)
    # ----------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[common_features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["label"] = df["label"].values

    plt.style.use("seaborn-v0_8-paper")
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df_pca, x="PC1", y="PC2", hue="label",
        s=80, palette="Set1", alpha=0.8
    )
    plt.title("Timbre Space of Zubeen's Humming vs Singling", fontsize=14, fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    plt.legend(title="", fontsize=10)
    plt.tight_layout()
    plt.savefig("HS-timbre-space-pca.png", dpi=600, bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------
    # Publication-ready spectral feature comparison
    # ----------------------------------------------------------
    spectral_features = ["spectral_centroid", "spectral_rolloff", "spectral_tilt", "spectral_roughness"]

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), dpi=300)

    for idx, feature in enumerate(spectral_features):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        sns.boxplot(
            data=df, x='label', y=feature, hue='label', palette='coolwarm',
            ax=ax, width=0.6, fliersize=3, legend=False
        )
        sns.stripplot(
            data=df, x="label", y=feature, color="black", size=3, alpha=0.5,
            ax=ax, dodge=True, jitter=0.12
        )
        ax.set_xlabel("")
        ax.set_ylabel(feature_labels.get(feature, feature), fontsize=10)
        ax.set_title(feature_labels.get(feature, feature), fontsize=11, fontweight="semibold", pad=8)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{x:.2f}" if abs(x) < 1000 else f"{x/1000:.1f}k"
        ))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.suptitle(
        "Spectral Feature Distributions: Zubeen's Humming vs Singling",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("HS-spectral-comparison.png", bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------
    # Temporal feature comparison (Jitter, Shimmer, HNR)
    # ----------------------------------------------------------
    temporal_features = ['jitter', 'shimmer', 'HNR']

    fig_t, axes_t = plt.subplots(1, 3, figsize=(12, 4))
    for idx, feature in enumerate(temporal_features):
        ax = axes_t[idx]
        sns.boxplot(
            data=df, x="label", y=feature, hue="label",
            palette=["#377eb8", "#e41a1c"], ax=ax, width=0.6,
            linewidth=1.2, fliersize=2.5, legend=False
        )
        sns.swarmplot(
            data=df, x='label', y=feature, color='0.25', ax=ax, size=4, alpha=0.6
        )
        ax.set_xlabel('')
        ax.set_ylabel(feature.title(), fontsize=11)
        ax.set_title(f'{feature.title()} Comparison', fontsize=12, fontweight='semibold')
        ax.tick_params(axis='both', labelsize=9)
    plt.suptitle("Temporal Stability: Humming  vs Singing", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('HS-temporal-comparison.png', dpi=600, bbox_inches='tight')
    plt.close(fig_t)

    # ----------------------------------------------------------
    # Feature significance visualization
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.barplot(data=stats_df, x='p_value', y='Feature', hue='Feature', palette='viridis_r', legend=False)
    plt.axvline(0.05, color='red', linestyle='--', label='p = 0.05')
    plt.xlabel('Wilcoxon p-value (Zubeen vs Others)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Statistical Significance of Acoustic Feature Differences', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('HS-feature-significance.png', dpi=600, bbox_inches='tight')
    plt.close()



    # ----------------------------------------------------------
    # Unified multi-panel figure (Nature-style layout)
    # ----------------------------------------------------------
    plt.style.use("seaborn-v0_8-paper")

    fig = plt.figure(figsize=(14, 12), dpi=300)
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1], hspace=0.45)

    # ----------------------------------------------------------
    # (a–d) Spectral features
    # ----------------------------------------------------------
    spectral_features = [
        ("spectral_centroid", "(a) Spectral Centroid", "Centroid"),
        ("spectral_rolloff", "(b) Spectral Rolloff", "Rolloff (Hz)"),
        ("spectral_tilt", "(c) Spectral Tilt", "Tilt"),
        ("spectral_roughness", "(d) Spectral Roughness", "Roughness")
    ]

    for idx, (feature, title, ylabel) in enumerate(spectral_features):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        sns.boxplot(
            data=df, x="label", y=feature, hue="label",
            palette=["#377eb8", "#e41a1c"], legend=False, ax=ax, width=0.6
        )
        sns.stripplot(
            data=df, x="label", y=feature, color="black", size=3,
            alpha=0.4, ax=ax, dodge=True, jitter=0.12
        )
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.tick_params(axis="both", labelsize=9)

    # ----------------------------------------------------------
    # (e–g) Temporal features
    # ----------------------------------------------------------
    temporal_features = [
        ("jitter", "(e) Jitter", "Jitter"),
        ("shimmer", "(f) Shimmer", "Shimmer"),
        ("HNR", "(g) Harmonics-to-Noise Ratio", "HNR (dB)")
    ]

    for i, (feature, title, ylabel) in enumerate(temporal_features):
        ax = fig.add_subplot(gs[2, i % 2] if i < 2 else gs[3, 0])
        sns.boxplot(
            data=df, x="label", y=feature, hue="label",
            palette=["#377eb8", "#e41a1c"], legend=False, ax=ax, width=0.6
        )
        sns.stripplot(
            data=df, x="label", y=feature, color="black", size=3,
            alpha=0.4, ax=ax, dodge=True, jitter=0.12
        )
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.tick_params(axis="both", labelsize=9)

    # ----------------------------------------------------------
    # (h) PCA timbre space
    # ----------------------------------------------------------
    ax_pca = fig.add_subplot(gs[3, 1])
    sns.scatterplot(
        data=df_pca, x="PC1", y="PC2", hue="label",
        palette="Set1", alpha=0.8, s=70, ax=ax_pca
    )
    ax_pca.set_title("(h) Timbre Space (PCA)", fontsize=12, fontweight="semibold")
    ax_pca.legend(title="Vocal Mode", loc="best", fontsize=9)

    # ----------------------------------------------------------
    # Add another figure for feature significance + correlation map
    # ----------------------------------------------------------
    fig2 = plt.figure(figsize=(14, 10), dpi=300)
    gs2 = gridspec.GridSpec(2, 1, figure=fig2, height_ratios=[1, 1], hspace=0.4)

    # (a) Feature Significance
    ax_sig = fig2.add_subplot(gs2[0])
    sns.barplot(
        data=stats_df, x="p_value", y="Feature", hue="Feature",
        palette="viridis_r", legend=False, ax=ax_sig
    )
    ax_sig.axvline(0.05, color="red", linestyle="--", label="p=0.05")
    ax_sig.set_title("(a) Feature Significance", fontsize=12, fontweight="semibold")
    ax_sig.set_xlabel("p-value"); ax_sig.set_ylabel("")

    # (b) Feature Correlation Map
    ax_corr = fig2.add_subplot(gs2[1])
    sns.heatmap(
        corr, cmap='coolwarm', center=0, square=True,
        cbar_kws={'shrink': 0.7}, ax=ax_corr
    )
    ax_corr.set_title("(b) Feature Correlation Map", fontsize=12, fontweight="semibold")

    # ----------------------------------------------------------
    # Titles and saving
    # ----------------------------------------------------------
    fig.suptitle(
        "Comprehensive Acoustic Characterization of Zubeen Garg’s Humming – Spectral, Temporal, and Timbre Features",
        fontsize=14, fontweight="bold", y=0.99
    )
    fig2.suptitle(
        "Statistical and Correlative Structure of Acoustic Features",
        fontsize=14, fontweight="bold", y=0.99
    )

    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.07, right=0.97, wspace=0.3, hspace=0.35)

    fig.savefig("HS-composite-features.png", dpi=600, bbox_inches="tight")
    fig2.savefig("HS-composite-stats.png", dpi=600, bbox_inches="tight")

    plt.close('all')



    print("\nAnalysis complete. Plots saved as:")
    print("  - HS-timbre-space-pca.png")
    print("  - HS-spectral-comparison-others.png")
    print("  - HS-temporal-comparison.png")
    print("  - HS-feature-significance.png")
    print("  - HS-feature-correlation.png")

    
# --------------------------------------------------------------
if __name__ == "__main__":
    main()
