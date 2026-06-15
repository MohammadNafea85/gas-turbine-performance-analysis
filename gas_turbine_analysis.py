# =============================================================
# 🔥 Power Plant Efficiency & Emissions Analysis
# Gas Turbine Performance — UCI Dataset (2011-2015)
# Author: Mohammad Nafea | Power Plant Senior Operation Engineer
# =============================================================

# ── STEP 1: Install & Import Libraries ──────────────────────
# pip install ucimlrepo pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── STEP 2: Load Dataset ─────────────────────────────────────
# Dataset: Gas Turbine CO and NOx Emission (UCI ML Repository)
# 36,733 hourly sensor readings | Gas Turbine — Turkey (2011–2015)
# Columns: AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO, NOX

try:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=551)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    print("✅ Dataset loaded from UCI repository")
except:
    # Fallback: simulate realistic data if no internet
    np.random.seed(42)
    n = 36733
    df = pd.DataFrame({
        'AT':   np.random.normal(17.7, 8.0, n),      # Ambient Temperature (°C)
        'AP':   np.random.normal(1013, 5.0, n),       # Ambient Pressure (mbar)
        'AH':   np.random.uniform(24, 100, n),        # Ambient Humidity (%)
        'AFDP': np.random.normal(3.5, 1.0, n),        # Air Filter Diff. Pressure (mbar)
        'GTEP': np.random.normal(25, 5.0, n),         # GT Exhaust Pressure (mbar)
        'TIT':  np.random.normal(1081, 30, n),        # Turbine Inlet Temp (°C)
        'TAT':  np.random.normal(546, 15, n),         # Turbine After Temp (°C)
        'CDP':  np.random.normal(9.2, 1.0, n),        # Compressor Discharge Pressure (bar)
        'TEY':  np.random.normal(133, 15, n),         # Turbine Energy Yield (MWh)
        'CO':   np.random.exponential(2.0, n),        # CO Emissions (mg/m³)
        'NOX':  np.random.normal(65, 20, n),          # NOx Emissions (mg/m³)
    })
    print("⚠️  Using simulated data (install ucimlrepo for real data)")

# ── STEP 3: Basic Exploration ────────────────────────────────
print("\n" + "="*55)
print("📊 DATASET OVERVIEW")
print("="*55)
print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Missing Values : {df.isnull().sum().sum()}")
print(f"\nKey Statistics:")
print(df[['AT','TIT','TEY','CO','NOX']].describe().round(2))

# ── STEP 4: Data Cleaning ────────────────────────────────────
df.dropna(inplace=True)
df = df[(df['TEY'] > 50) & (df['TEY'] < 200)]   # realistic MW range
df = df[(df['TIT'] > 900) & (df['TIT'] < 1200)] # realistic TIT range
df = df[df['CO']  >= 0]
df = df[df['NOX'] >= 0]
df.reset_index(drop=True, inplace=True)
print(f"\n✅ After cleaning: {df.shape[0]:,} rows")

# Derive Thermal Efficiency (simplified) & Heat Rate
df['heat_rate']   = 3412 / (df['TEY'] / df['TEY'].max() * 0.6)   # BTU/kWh proxy
df['efficiency']  = 100 * (df['TEY'] / df['TEY'].max())           # relative %
df['temp_effect'] = df['TIT'] - df['AT']                          # delta T

# Add year column (2011–2015, 5 equal parts)
df['year'] = pd.cut(df.index, bins=5,
                    labels=[2011, 2012, 2013, 2014, 2015]).astype(int)

# ── STEP 5: VISUALISATIONS ───────────────────────────────────
# Color palette — engineering/industrial feel
COLORS = ['#1A6B8A', '#E07B39', '#2ECC71', '#E74C3C', '#9B59B6']
BG     = '#F7F9FC'
GRID   = '#D5DCE8'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG,
    'axes.grid':        True,
    'grid.color':       GRID,
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'font.size':        10,
})

# ════════════════════════════════════════════════════════
# FIGURE 1 — Main Dashboard (2×3)
# ════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "⚙️  Gas Turbine Performance & Emissions Dashboard\n"
    "UCI Dataset — Combined Cycle Power Plant (2011–2015)",
    fontsize=16, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# — Plot 1: Energy Yield Distribution ——————————————
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['TEY'], bins=60, color=COLORS[0], edgecolor='white', linewidth=0.4)
ax1.axvline(df['TEY'].mean(), color=COLORS[1], linewidth=2,
            linestyle='--', label=f"Mean = {df['TEY'].mean():.1f} MWh")
ax1.set_title("Turbine Energy Yield Distribution", fontweight='bold')
ax1.set_xlabel("Energy Yield (MWh)")
ax1.set_ylabel("Frequency")
ax1.legend()

# — Plot 2: TEY vs Ambient Temperature ————————————
ax2 = fig.add_subplot(gs[0, 1])
sc = ax2.scatter(df['AT'], df['TEY'], c=df['TIT'],
                 cmap='plasma', alpha=0.3, s=3)
plt.colorbar(sc, ax=ax2, label='Turbine Inlet Temp (°C)')
ax2.set_title("Energy Yield vs Ambient Temperature", fontweight='bold')
ax2.set_xlabel("Ambient Temperature (°C)")
ax2.set_ylabel("Energy Yield (MWh)")

# — Plot 3: NOx vs CO Emissions ————————————————
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(df['CO'], df['NOX'], alpha=0.15, s=3, color=COLORS[3])
ax3.set_title("CO vs NOx Emissions", fontweight='bold')
ax3.set_xlabel("CO Emissions (mg/m³)")
ax3.set_ylabel("NOx Emissions (mg/m³)")
# Add operational limits
ax3.axhline(70, color=COLORS[1], linestyle='--', linewidth=1.2,
            label='NOx limit 70 mg/m³')
ax3.legend(fontsize=8)

# — Plot 4: Yearly Avg Energy Yield ————————————————
ax4 = fig.add_subplot(gs[1, 0])
yearly = df.groupby('year')['TEY'].agg(['mean','std']).reset_index()
bars = ax4.bar(yearly['year'], yearly['mean'],
               color=COLORS[0], width=0.6, edgecolor='white')
ax4.errorbar(yearly['year'], yearly['mean'], yerr=yearly['std'],
             fmt='none', color='#333', capsize=5, linewidth=1.5)
ax4.set_title("Average Energy Yield per Year", fontweight='bold')
ax4.set_xlabel("Year")
ax4.set_ylabel("Avg TEY (MWh)")
for bar, val in zip(bars, yearly['mean']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

# — Plot 5: Correlation Heatmap ————————————————
ax5 = fig.add_subplot(gs[1, 1])
cols_corr = ['AT','TIT','TAT','CDP','TEY','CO','NOX']
corr = df[cols_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax5, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax5.set_title("Parameter Correlation Matrix", fontweight='bold')
ax5.tick_params(axis='x', rotation=30)

# — Plot 6: Heat Rate Trend ————————————————————
ax6 = fig.add_subplot(gs[1, 2])
hr_year = df.groupby('year')['heat_rate'].mean()
ax6.plot(hr_year.index, hr_year.values, 'o-',
         color=COLORS[1], linewidth=2.5, markersize=8)
ax6.fill_between(hr_year.index, hr_year.values,
                 alpha=0.2, color=COLORS[1])
ax6.set_title("Average Heat Rate Trend (BTU/kWh)", fontweight='bold')
ax6.set_xlabel("Year")
ax6.set_ylabel("Heat Rate (BTU/kWh)")
ax6.invert_yaxis()  # lower heat rate = better efficiency

plt.savefig('/mnt/user-data/outputs/dashboard.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Dashboard saved → dashboard.png")

# ════════════════════════════════════════════════════════
# FIGURE 2 — ML Model: Predict Energy Yield
# ════════════════════════════════════════════════════════
features = ['AT', 'AP', 'AH', 'TIT', 'TAT', 'CDP', 'AFDP', 'GTEP']
target   = 'TEY'

X_data = df[features]
y_data = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*55)
print("🤖 ML MODEL RESULTS — Linear Regression")
print("="*55)
print(f"R² Score (Accuracy)  : {r2:.4f}  ({r2*100:.1f}%)")
print(f"Mean Absolute Error  : {mae:.2f} MWh")
print(f"\nFeature Importance (Coefficients):")
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef_df.to_string(index=False))

# Plot: Actual vs Predicted
fig2, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig2.suptitle("🤖 ML Model — Turbine Energy Yield Prediction",
              fontsize=14, fontweight='bold')

ax_a = axes[0]
sample = min(500, len(y_test))
idx   = np.random.choice(len(y_test), sample, replace=False)
ax_a.scatter(y_test.iloc[idx], y_pred[idx],
             alpha=0.5, s=20, color=COLORS[0])
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax_a.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
ax_a.set_xlabel("Actual TEY (MWh)")
ax_a.set_ylabel("Predicted TEY (MWh)")
ax_a.set_title(f"Actual vs Predicted  |  R² = {r2:.3f}", fontweight='bold')
ax_a.legend()
ax_a.set_facecolor(BG)
ax_a.grid(True, color=GRID)

ax_b = axes[1]
residuals = y_test.values - y_pred
ax_b.hist(residuals, bins=50, color=COLORS[2], edgecolor='white', linewidth=0.4)
ax_b.axvline(0, color='red', linewidth=2, linestyle='--')
ax_b.set_xlabel("Residual (Actual − Predicted) MWh")
ax_b.set_ylabel("Frequency")
ax_b.set_title("Residual Distribution", fontweight='bold')
ax_b.set_facecolor(BG)
ax_b.grid(True, color=GRID)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/ml_model.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ ML Model chart saved → ml_model.png")

# ── STEP 6: Key Insights Summary ─────────────────────────────
print("\n" + "="*55)
print("💡 KEY ENGINEERING INSIGHTS")
print("="*55)
print(f"• Avg Energy Output  : {df['TEY'].mean():.1f} MWh (σ = {df['TEY'].std():.1f})")
print(f"• Peak Output        : {df['TEY'].max():.1f} MWh")
print(f"• Low Output         : {df['TEY'].min():.1f} MWh")
print(f"• Avg NOx Emissions  : {df['NOX'].mean():.1f} mg/m³")
print(f"• Avg CO  Emissions  : {df['CO'].mean():.2f} mg/m³")
print(f"• Temp impact on TEY : {df[['AT','TEY']].corr().iloc[0,1]:.3f} correlation")
print(f"• TIT impact on TEY  : {df[['TIT','TEY']].corr().iloc[0,1]:.3f} correlation")
best_year = df.groupby('year')['TEY'].mean().idxmax()
print(f"• Best Efficiency Yr : {best_year}")
print(f"\n✅ Analysis Complete! Check output charts.")
print("="*55)
