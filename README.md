# 🎶 Music Genre Classification System

Welcome to the **Music Genre Classification System**!  
This project leverages deep learning and audio signal processing to automatically identify the genre of a music track from an uploaded audio file.  
Experience the power of AI in music analysis, all through a beautiful and interactive web interface.

---

## 🚀 Features

- **Easy-to-use Web App**: Built with [Streamlit](https://streamlit.io/), no coding required!
- **Deep Learning Model**: Utilizes a trained TensorFlow Keras model for high-accuracy predictions.
- **Audio Preprocessing**: Converts audio into Mel-Spectrograms, mimicking human hearing.
- **Supports Multiple Formats**: Upload `.mp3` or `.wav` files.
- **Instant Results**: Get the predicted genre in seconds, with a celebratory animation!

---

## 🖼️ Demo

![App Screenshot](A1.jpg)

---

## 🛠️ How It Works

1. **Upload Audio**: Go to the "Prediction" page and upload your music file.
2. **Processing**: The system splits the audio, generates Mel-Spectrograms, and feeds them to a CNN.
3. **Prediction**: The model predicts the most likely genre from 10 possible categories.
4. **Result**: See the genre and enjoy the interactive UI!

---

## 🎵 Supported Genres

- Blues
- Classical
- Country
- Disco
- Hiphop
- Jazz
- Metal
- Pop
- Reggae
- Rock

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Music
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the model and image**
   - `Trained_model.keras` (pre-trained model, included)
   - `A1.jpg` (for the homepage)

---

## ▶️ Usage

Launch the Streamlit app:

```bash
streamlit run a.py
```

Open the provided local URL in your browser and enjoy!

---

## 📁 Project Structure

```
.
├── a.py                      # Main Streamlit app
├── requirements.txt          # Python dependencies
├── Trained_model.keras       # Pre-trained Keras model
├── A1.jpg                    # App banner image
├── Music_Genre_Classification_System/ # (Optional: source/data)
└── star/                     # (Virtual environment and packages)
```

---

## 🧠 How Does It Work?

- **Audio Loading**: Uses [Librosa](https://librosa.org/) to load and process audio.
- **Mel-Spectrograms**: Converts audio chunks into Mel-Spectrogram images.
- **CNN Model**: A Convolutional Neural Network classifies the spectrograms.
- **Majority Voting**: For longer tracks, the most frequent prediction is chosen.

---

## 📊 Example: Fourier Transform Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

fs = 500
t = np.linspace(0, 1, fs, endpoint=False)
freqs = [5, 50, 120]
signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)

fft_values = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(t), 1/fs)

plt.plot(frequencies[:fs//2], np.abs(fft_values[:fs//2]))
plt.title("Frequency Spectrum of a Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()
```

---

## 📚 Dataset

- **Source**: [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html)
- **Classes**: 10 genres, 100 audio files each, 30 seconds per file.
- **Preprocessing**: Mel-Spectrograms, data augmentation with 3-second chunks.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgements

- [Librosa](https://librosa.org/) for audio processing
- [Streamlit](https://streamlit.io/) for the web interface
- [TensorFlow](https://www.tensorflow.org/) for deep learning
- [GTZAN Dataset](http://marsyas.info/downloads/datasets.html)

---

Feel free to further customize this README with your contact info, links, or additional instructions! 