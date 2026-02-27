import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Phase Noise Model Function
# -----------------------------
def phase_noise_model(f, h0, h1, h2, h3):
    """
    Computes phase noise in dBc/Hz given coefficients h0..h3
    and frequency array f (Hz)
    """
    return 10 * np.log10(h0 + h1/f + h2/f**2 + h3/f**3)

# =============================================================================
# Global Settings
# =============================================================================
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
save_path = 'C:/Users/16474/Desktop/ece412assignment2/'

# =============================================================================
# Given Specs
# =============================================================================
fosc1 = 1.9e9
fosc2 = 2.1e9
Kosc = 250e6
LVCO_1MHz = -125
lock_time_max = 20e-6
fREF = 20e6

N_min = int(np.ceil(fosc1 / fREF))
N_max = int(np.floor(fosc2 / fREF))
N = N_max
fosc = N * fREF

print(f"Division Ratio N: {N}")

# =============================================================================
# Part 2 & 3: Reference Oscillator and VCO Phase Noise Models
# =============================================================================
# Reference: Lref = -160 dBc/Hz floor, -140 dBc/Hz at 10 kHz, 1/f^2 dependence
h0_ref = 10**(-160/10)
h2_ref = (10**(-140/10) - h0_ref) * (10e3)**2
h1_ref = 0
h3_ref = 0

print("\nReference Oscillator Phase Noise Coefficients:")
print(f"h0 = {h0_ref:.4e}, h1 = {h1_ref:.4e}, h2 = {h2_ref:.4e}, h3 = {h3_ref:.4e}")

# VCO: L_vco floor = -140 dBc/Hz, 1/f^3 dependence, -125 dBc/Hz at 1 MHz
h0_vco = 10**(-140/10)
h3_vco = (10**(-125/10) - h0_vco) * (1e6)**3
h1_vco = 0
h2_vco = 0

print("\nVCO Phase Noise Coefficients:")
print(f"h0 = {h0_vco:.4e}, h1 = {h1_vco:.4e}, h2 = {h2_vco:.4e}, h3 = {h3_vco:.4e}")

f_range = np.logspace(2, 8, 2000)
L_ref = 10*np.log10(h0_ref + h1_ref/f_range + h2_ref/f_range**2 + h3_ref/f_range**3)
L_vco = 10*np.log10(h0_vco + h1_vco/f_range + h2_vco/f_range**2 + h3_vco/f_range**3)

plt.figure()
plt.semilogx(f_range, L_ref, linewidth=2, label='Reference Oscillator')
plt.semilogx(f_range, L_vco, linewidth=2, color='r', label='VCO')
plt.grid(True, which='both', alpha=0.3)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase Noise [dBc/Hz]")
plt.title("Reference Oscillator and VCO Phase Noise")
plt.ylim([-170, -90])
plt.legend()
plt.tight_layout()
plt.savefig(save_path + 'part2_3_phase_noise.png', dpi=300)
plt.close()

# =============================================================================
# Part 4: Optimal PLL Bandwidth
# =============================================================================
N_squared_L_ref = 20*np.log10(N) + L_ref
diff = N_squared_L_ref - L_vco
idx = np.where(np.diff(np.sign(diff)))[0][0]

# Linear interpolation for crossover
f1, f2 = f_range[idx], f_range[idx+1]
d1, d2 = diff[idx], diff[idx+1]
f_cross = f1 - d1*(f2 - f1)/(d2 - d1)
omega_3dB = 2*np.pi*f_cross
print(f"\nOptimal PLL Bandwidth: {f_cross:.3e} Hz")

# =============================================================================
# Part 5: Type-II PLL Design
# =============================================================================
def design_pll(Q):

    print(f"\n===== PLL Design for Q = {Q} =====")

    if Q == 0.1:
        omega_pll = Q * omega_3dB
        C2_ratio = 250
    else:
        omega_pll = 0.4 * omega_3dB
        C2_ratio = 10

    Ich = 100e-6
    Ich_over_C1 = omega_pll**2 * 2*np.pi*N / Kosc
    C1 = Ich / Ich_over_C1
    omega_z = Q * omega_pll
    R = 1/(omega_z * C1)
    C2 = C1 / C2_ratio

    tau = Q / omega_pll
    assert tau < lock_time_max, "Lock time exceeds max lock time"

    print(f"Charge pump current: {Ich:.4e} A")
    print(f"Loop filter R: {R:.4e} Ω, C1: {C1:.4e} F, C2: {C2:.4e} F")
    print(f"Settling time constant: {tau:.4e} s")

    # -----------------------------
    # Part 5b: PLL Loop Gain
    # -----------------------------
    f = np.logspace(2, 8, 3000)
    s = 1j*2*np.pi*f
    Kpd = Ich/(2*np.pi)
    Hlp = (1 + s*R*C1) / (s*(C1+C2)*(1 + s*R*(C1*C2)/(C1+C2)))
    L = Kpd * Hlp * Kosc / (N*s)
    magL = 20*np.log10(np.abs(L))
    phaseL = np.angle(L, deg=True)

    # Unity gain frequency (0 dB crossing)
    idx = np.where(np.diff(np.sign(magL)))[0][0]
    f_ug = f[idx] - magL[idx]*(f[idx+1]-f[idx])/(magL[idx+1]-magL[idx])
    phase_at_ug = np.interp(f_ug, f, phaseL)
    phase_margin = 180 + phase_at_ug

    plt.figure()
    plt.subplot(2,1,1)
    plt.semilogx(f, magL, linewidth=2)
    plt.axvline(f_ug, linestyle='--', color='r')
    plt.annotate(f"Unity Gain\n{f_ug:.2e} Hz", xy=(f_ug,0), xytext=(f_ug*1.3,-25),
                 arrowprops=dict(arrowstyle="->"), bbox=dict(boxstyle="round", fc="white"))
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which='both', alpha=0.3)

    plt.subplot(2,1,2)
    plt.semilogx(f, phaseL, linewidth=2)
    plt.annotate(f"Phase Margin ≈ {phase_margin:.1f}°", xy=(f_ug, phase_at_ug),
                 xytext=(f_ug*0.5, phase_at_ug+40),
                 arrowprops=dict(arrowstyle="->"), bbox=dict(boxstyle="round", fc="white"))
    plt.ylabel("Phase (deg)")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path + f"L_Q={Q}.png", dpi=300)
    plt.close()

    # -----------------------------
    # Part 5c: Jitter Transfer Function
    # -----------------------------
    H = (Kpd * Hlp * Kosc / s) / (1 + L)
    magH = 20*np.log10(np.abs(H))
    idx = np.where(np.diff(np.sign(magH + 3)))[0][0]
    f_3db = f[idx] - (magH[idx]+3)*(f[idx+1]-f[idx])/(magH[idx+1]-magH[idx])

    plt.figure()
    plt.semilogx(f, magH, linewidth=2)
    plt.axvline(f_3db, linestyle='--', color='r')
    plt.annotate(f"3dB Bandwidth\n{f_3db:.2e} Hz", xy=(f_3db, -3), xytext=(f_3db*1.5, -15),
                 arrowprops=dict(arrowstyle="->"), bbox=dict(boxstyle="round", fc="white"))
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(save_path + f"H_Q={Q}.png", dpi=300)
    plt.close()

    # -----------------------------
    # Part 5d: Output Phase Noise
    # -----------------------------
    S_ref = 10**(phase_noise_model(f, h0_ref, 0, h2_ref, 0)/10)
    S_vco = 10**(phase_noise_model(f, h0_vco, 0, 0, h3_vco)/10)
    L_out = 10*np.log10(np.abs(H)**2*S_ref + np.abs(1 / (1 + L))**2*S_vco)

    plt.figure()
    plt.semilogx(f, 10*np.log10(S_ref), linewidth=2, label='S_ref')
    plt.semilogx(f, 10*np.log10(S_vco), linewidth=2, label='S_vco')
    plt.semilogx(f, L_out, linewidth=2, label='Combined Output')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase Noise [dBc/Hz]")
    plt.tight_layout()
    plt.savefig(save_path + f"L_out_Q={Q}.png", dpi=300)
    plt.close()

    # -----------------------------
    # Part 5e: RMS Random Jitter
    # -----------------------------
    mask = (f >= 1e3) & (f <= 100e6)
    phase_variance = np.trapz(10**(L_out[mask]/10), f[mask])

    rms_phase_rad = np.sqrt(phase_variance)
    rms_phase_deg = rms_phase_rad * 180 / np.pi
    rms_jitter_ps = rms_phase_rad / (2*np.pi*fosc) * 1e12

    print(f"Unity Gain Frequency: {f_ug:.3e} Hz")
    print(f"Phase Margin: {phase_margin:.2f}°")
    print(f"3 dB Bandwidth: {f_3db:.3e} Hz")
    print(f"RMS Phase Error: {rms_phase_deg:.3f}°")
    print(f"RMS Random Jitter: {rms_jitter_ps:.3f} ps")

design_pll(0.1)
