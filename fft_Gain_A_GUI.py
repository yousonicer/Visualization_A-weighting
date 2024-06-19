import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy.fft import fft
import math
import librosa
import pyaudio
import wave
import threading


def fft_custom(sig, Fs, zero_padding=False, window=True):
    if zero_padding:
        fft_num = np.power(2, math.ceil(np.log2(len(sig))))
        if window:
            mag = np.fft.fft(sig * np.hanning(len(sig)), fft_num) * 2 / fft_num
            f = np.arange(fft_num) * (Fs / fft_num)
        else:
            mag = np.fft.fft(sig, fft_num) * 2 / fft_num
            f = np.arange(fft_num) * (Fs / fft_num)
    else:
        fft_num = len(sig)
        if window:
            mag = np.fft.fft(sig * np.hanning(len(sig)), fft_num) * 2 / fft_num
            f = np.arange(fft_num) * (Fs / fft_num)
        else:
            mag = np.fft.fft(sig, fft_num) * 2 / fft_num
            f = np.arange(fft_num) * (Fs / fft_num)
    mag[0] /= 2
    return f[:int(fft_num / 2)], abs(mag[:int(fft_num / 2)])


def get_octave_band_base_2(start, multi, oc):
    bands = list()
    for i in range(0, multi * 3 - 2 if oc else multi):
        central_frequency = start * np.power(2, i / 3 if oc else i)
        bands.append(
            [central_frequency, central_frequency / np.power(2, 1 / 6), central_frequency * np.power(2, 1 / 6)])
    return np.asarray(bands)


def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    p.terminate()


def record_audio(file_path, duration, sr):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = sr
    RECORD_SECONDS = duration

    try:
        p = pyaudio.PyAudio()
        list_audio_devices()  # 列出可用的音频设备

        # 默认选择设备ID为0的设备
        device_index = 0

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Error opening stream: {e}")
        return

    frames = []
    print("Recording...")

    try:
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            # 检查数据是否为全零
            if np.frombuffer(data, dtype=np.int16).max() == 0:
                print(f"Frame {i}: data is all zeros")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        print("Recording finished.")
        stream.stop_stream()
        stream.close()
        p.terminate()

    try:
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    except Exception as e:
        print(f"Error saving file: {e}")


def a_weighting(frequencies):
    """
    Apply A-weighting to the given frequency array based on standard values.

    :param frequencies: array of frequencies to apply A-weighting
    :return: array of A-weighted values in dB
    """
    a_weighting_values = {
        10: -70.4, 12.5: -63.4, 16: -56.7, 20: -50.5, 25: -44.7,
        31.5: -39.4, 40: -34.6, 50: -30.2, 63: -26.2, 80: -22.5,
        100: -19.1, 125: -16.1, 160: -13.4, 200: -10.9, 250: -8.6,
        315: -6.6, 400: -4.8, 500: -3.2, 630: -1.9, 800: -0.8,
        1000: 0.0, 1250: 0.6, 1600: 1.0, 2000: 1.2, 2500: 1.3,
        3150: 1.2, 4000: 1.0, 5000: 0.5, 6300: -0.1, 8000: -1.1,
        10000: -2.5, 12500: -4.3, 16000: -6.6, 20000: -9.3
    }

    # Interpolate A-weighting values for given frequencies
    freqs = np.array(list(a_weighting_values.keys()))
    a_weights = np.array(list(a_weighting_values.values()))
    a_weight_db = np.interp(frequencies, freqs, a_weights)

    return a_weight_db


def measure_spl(file_path, record_time=0.5, sr=44100, start_f=31.25, multi_f=8, oc_f=0, gain=0, A_weighting=False,
                plot=None):
    record_audio(file_path, duration=record_time, sr=sr)
    print("Audio recorded and saved as 'recorded_audio.wav'.")

    sig, Fs = librosa.load(file_path)
    print(sig)

    sig = sig * 10 ** (gain / 20)

    N = len(sig)
    n = list(np.arange(N))
    t = np.asarray([temp / Fs for temp in n])

    temp = [temp * temp for temp in sig]
    p_square_time = np.sum(temp) / len(temp)
    print("p_square_time: ", p_square_time)

    f, mag_fft = fft_custom(sig, Fs)

    if A_weighting:
        a_weight = a_weighting(f)
        mag_fft = mag_fft * 10 ** (a_weight / 20)

    temp = [(temp / np.sqrt(2)) * (temp / np.sqrt(2)) for temp in mag_fft]
    p_square_frequency = np.sum(temp)
    print("p_square_frequency: ", p_square_frequency)

    spl_overall = 20 * np.log10(p_square_time / 0.0000000004)
    print("声压级(DB):", spl_overall)

    bands = get_octave_band_base_2(start=start_f, multi=multi_f, oc=oc_f)
    spl_of_bands = list()
    f = list(f)
    index_start = 0
    for band in bands:
        index_stop = np.where(f < band[2])[0][-1]
        band_frequencies_mag = mag_fft[index_start:index_stop]
        temp = [temp ** 2 / 2 for temp in band_frequencies_mag]
        p_square_frequency_band = np.sum(temp)
        spl_ = 20 * np.log10(p_square_frequency_band / 0.0000000004)
        if spl_ < 0:
            spl_ = 0.0000000004
        if band[0] <= Fs / 2:
            spl_of_bands.append([band[0], spl_])
        else:
            break
        index_start = index_stop

    spl_values = list()
    xticks = list()

    for temp in spl_of_bands:
        spl_values.append(temp[1])
        if temp[0] % start_f > 0.0:
            xticks.append(' ')
        elif temp[0] >= 1000:
            xticks.append(str(int(temp[0] / 1000)) + 'k')
        else:
            xticks.append(str(int(temp[0])))

    if plot is not None:
        plot.cla()
        plot.bar(range(len(spl_values)), spl_values, facecolor='b', width=0.8, zorder=3)
        plot.set_title("1/3 octave SPL || Overall SPL is %f " % np.round(spl_overall, 3))
        plot.set_xticks(list(range(len(spl_values))))
        plot.set_xticklabels(xticks, rotation=30)
        plot.grid(axis='x', linestyle='--', zorder=0)
        plot.set_xlabel('Freq/Hz')
        plot.set_ylabel('SPL')
        plot.figure.canvas.draw()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder and SPL Meter")

        self.record_time = 0.5
        self.sr = 44100
        self.start_f = 31.25
        self.multi_f = 8
        self.oc_f = 0
        self.gain = 10
        self.A_weighting = False

        self.gain_label = tk.Label(root, text="Gain (dB):")
        self.gain_label.pack()

        # 增加一个框架用于存放增大和减小按钮
        self.gain_frame = tk.Frame(root)
        self.gain_frame.pack(pady=5)

        self.gain_decrease_button = tk.Button(self.gain_frame, text="-", command=self.decrease_gain, width=3)
        self.gain_decrease_button.grid(row=0, column=0)

        self.gain_entry = tk.Entry(self.gain_frame, width=5, justify='center')
        self.gain_entry.grid(row=0, column=1, padx=5)
        self.gain_entry.insert(0, str(self.gain))

        self.gain_increase_button = tk.Button(self.gain_frame, text="+", command=self.increase_gain, width=3)
        self.gain_increase_button.grid(row=0, column=2)

        self.a_weighting_var = tk.IntVar()
        self.a_weighting_check = tk.Checkbutton(root, text="A-Weighting", variable=self.a_weighting_var)
        self.a_weighting_check.pack()

        self.start_button = tk.Button(root, text="Start Measurement", command=self.toggle_measurement)
        self.start_button.pack()

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.is_running = False

    def increase_gain(self):
        self.gain += 1
        self.gain_entry.delete(0, tk.END)
        self.gain_entry.insert(0, str(self.gain))

    def decrease_gain(self):
        self.gain -= 1
        self.gain_entry.delete(0, tk.END)
        self.gain_entry.insert(0, str(self.gain))

    def toggle_measurement(self):
        if self.is_running:
            self.stop_measurement()
        else:
            self.start_measurement()

    def start_measurement(self):
        self.is_running = True
        self.start_button.config(text="End Measurement")
        self.update_measurement()

    def stop_measurement(self):
        self.is_running = False
        self.start_button.config(text="Start Measurement")

    def update_measurement(self):
        if self.is_running:
            self.gain = int(self.gain_entry.get())
            self.A_weighting = bool(self.a_weighting_var.get())
            threading.Thread(target=measure_spl, args=('recorded_audio.wav', self.record_time, self.sr, self.start_f, self.multi_f, self.oc_f, self.gain, self.A_weighting, self.ax)).start()
            self.root.after(1000, self.update_measurement)  # 更新间隔



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
