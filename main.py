import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

fosc1 = 1.9e9 # Desired output frequency [Hz]
fosc2 = 2.1e9 # Desired output frequency [Hz]
Kosc = 250e6  # VCO gain [Hz/V]
LVCO_1MHz = -125  # VCO phase noise at 1MHz offset [dBc/Hz]
lock_time_max = 20e-6  # Maximum lock time [s]
fREF = 20e6  # Reference frequency [Hz]

N_min = int(np.ceil(fosc1/fREF))
N_max = int(np.floor(fosc2/fREF))
N = N_max
fosc = N * fREF

print(f"Division Ratio N: {N}")

# ============================================================================
# Problem 2: Reference Oscillator Phase Noise Model
# ============================================================================

h0_ref = 10**(-160/10)
h1_ref = 0
h2_ref = (10**(-140/10) - h0_ref) * (10e3)**2
h3_ref = 0

print("\nReference Oscillator Phase Noise Coefficients:")
print(f"h0 = {h0_ref:.4e}")
print(f"h1 = {h1_ref:.4e}")
print(f"h2 = {h2_ref:.4e}")
print(f"h3 = {h3_ref:.4e}")

# ============================================================================
# Problem 3: VCO Phase Noise Model
# ============================================================================

h0_vco = 10**(-140/10)
h1_vco = 0
h2_vco = 0
h3_vco = (10**(-125/10) - h0_vco) * (1e6)**3

print("\nVCO Phase Noise Coefficients:")
print(f"h0 = {h0_vco:.4e}")
print(f"h1 = {h1_vco:.4e}")
print(f"h2 = {h2_vco:.4e}")
print(f"h3 = {h3_vco:.4e}")

f_range = np.logspace(2, 8, 1000)
L_ref = 10*np.log10(h0_ref + h1_ref/f_range + h2_ref/f_range**2 + h3_ref/f_range**3)
L_vco = 10*np.log10(h0_vco + h1_vco/f_range + h2_vco/f_range**2 + h3_vco/f_range**3)

plt.figure(figsize=(10, 6))
plt.semilogx(f_range, L_ref, 'b-', linewidth=2, label='Reference Oscillator')
plt.semilogx(f_range, L_vco, 'r-', linewidth=2, label='VCO')
plt.grid(True, which='both', alpha=0.3)
plt.xlabel('Offset Frequency (Hz)')
plt.ylabel('Phase Noise L(f) [dBc/Hz]')
plt.title('Phase Noise Characteristics')
plt.legend(loc='upper right')
plt.xlim([1e2, 1e8])
plt.ylim([-170, -90])
plt.tight_layout()
plt.savefig('C:/Users/16474/Desktop/ece412assignment2/phase_noise_characteristics.png', dpi=300, bbox_inches='tight')
print("\nFigure saved: phase_noise_characteristics.png")

# ============================================================================
# Problem 4: Optimal PLL Bandwidth
# ============================================================================

# deviation
N_squared_L_ref = 20*np.log10(N) + L_ref
diff = N_squared_L_ref - L_vco
# find where diff crosses zero
idx = np.where(np.diff(np.sign(diff)))[0]
i = idx[0]
# interpolate
f1, f2 = f_range[i], f_range[i+1]
d1, d2 = diff[i], diff[i+1]
omega_3dB = 2 * np.pi * (f1 - d1 * (f2 - f1) / (d2 - d1))

print(f"Optimal bandwidth: {omega_3dB:.4e} rad/s")

# ============================================================================
# Problem 5: Type-II PLL Design
# ============================================================================

def problem5(Q):
    # part a
    if Q == 0.1:
        omega_pll = Q * omega_3dB
        C2_ratio = 250
    elif Q == 0.5:
        omega_pll = 0.4 * omega_3dB
        C2_ratio = 10

    Ich_over_C1 = omega_pll**2 * 2 * np.pi * N / Kosc
    Ich = 100e-6
    C1 = Ich / Ich_over_C1
    omega_z = Q * omega_pll
    R = 1 / (omega_z * C1)
    C2 = C1 / C2_ratio

    # part b
    f = np.logspace(2, 8, 1000)
    s = 1j * 2 * np.pi * f
    Kpd = Ich / (2 * np.pi)
    Klp_times_Hlp = (1 + s * R * C1) / (s * (C1 + C2) * (1 + s * R * (C1 * C2) / (C1 + C2)))
    L = Kpd * Klp_times_Hlp * Kosc / (N * s)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.semilogx(f, 20*np.log10(np.abs(L)), 'b-', linewidth=2)
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('PLL Loop Gain Magnitude')
    plt.subplot(2, 1, 2)
    plt.semilogx(f, np.angle(L, deg=True), 'r-', linewidth=2)
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.title('PLL Loop Gain Phase')
    plt.tight_layout()
    plt.savefig(f'C:/Users/16474/Desktop/ece412assignment2/PLL_Q_{Q}.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: PLL_Q_{Q}.png")

    # part c
    H = 1 / (1 + L) # jitter transfer function
    three_dB_freq = f[np.argmin(np.abs(20*np.log10(np.abs(H)) + 3))]
    unity_gain_freq = f[np.argmin(np.abs(20*np.log10(np.abs(L))))]
    loop_gain_phase_margin = 180 + np.angle(L[np.argmin(np.abs(20*np.log10(np.abs(L)) + 3))], deg=True)

    print(f"\nFor Q = {Q}:")
    print(f"Three dB frequency: {three_dB_freq:.4e} Hz")
    print(f"Unity gain frequency: {unity_gain_freq:.4e} Hz")
    print(f"Loop gain phase margin: {loop_gain_phase_margin} degrees")

    # part d
    L_out = 10*np.log10(np.abs(H)**2 * (h0_ref + h1_ref/f**2 + h2_ref/f**4 + h3_ref/f**6) + np.abs(1 - H)**2 * (h0_vco + h1_vco/f**2 + h2_vco/f**4 + h3_vco/f**6))
    plt.figure(figsize=(10, 6))
    plt.semilogx(f, L_out, 'm-', linewidth=2)
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Output Phase Noise L_out(f) [dBc/Hz]')
    plt.title(f'Output Phase Noise for Q = {Q}')
    plt.xlim([1e2, 1e8])
    plt.ylim([-170, -90])
    plt.tight_layout()
    plt.savefig(f'C:/Users/16474/Desktop/ece412assignment2/L_out_Q_{Q}.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: L_out_Q_{Q}.png")

    # part e: calculate the rms random jitter in ps and the rms phase error in degrees when the phase noise at the output of the PLL is integrated from 1 kHz to 100 MHz
    f_low = 1e3
    f_high = 100e6
    idx_low = np.argmin(np.abs(f - f_low))
    idx_high = np.argmin(np.abs(f - f_high))
    phase_noise_integral = np.trapz(10**(L_out[idx_low:idx_high]/10), f[idx_low:idx_high])
    rms_phase_error = np.sqrt(phase_noise_integral)
    rms_random_jitter = rms_phase_error / (2 * np.pi * fosc) * 1e12 # convert to ps
    print(f"\nRMS random jitter: {rms_random_jitter} ps")
    print(f"RMS phase error: {rms_phase_error} degrees")

problem5(0.1)
problem5(0.5)
