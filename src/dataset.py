import mne

# Load the EDF file
edf_file = "/Users/carlitos/Desktop/ECG-Project/data/raw/Raw ECG project/30100_LAB_Conditions_ECG.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

# Plot the raw signals
raw.plot(scalings="auto", title="EDF Data", show=True, block=True)

# Plot the power spectral density (optional)
raw.plot_psd()
