import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
import numpy as np
from tensorflow.image import resize

#Function
@st.cache_resource()
def load_model():
 model = tf.keras.models.load_model("Trained_model.keras")
 return model


# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
                    # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
                    # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
                    # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
                #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)



#Tensorflow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    #print(unique_elements, counts)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]



#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

## Main Page
if(app_mode=="Home"):
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #181646;  /* Blue background */
        color: white;
    }
    h2, h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown(''' ## Welcome to the,\n
    ## Music Genre Classification System! 🎶🎧''')
    image_path = "A1.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
**Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. **Results:** View the predicted genre along with related information.
""")



#About Project
elif app_mode == "About Project":
    st.markdown("""
        <h2 style="color:#4A90E2;">🎵 Music Genre Classification System</h2>

        <p>
            Understanding sound and what differentiates one song from another has been a long-standing challenge.
            This project aims to visualize and classify audio using Mel-Spectrograms and machine learning techniques such as Convolutional Neural Networks (CNNs).
        </p>

        <h3 style="color:#50E3C2;">📁 About Dataset</h3>
        <ul>
            <li><b>Genres Original:</b> 10 genres with 100 audio files each (30 seconds long), from the GTZAN dataset.</li>
            <li><b>Genres Included:</b> blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.</li>
            <li><b>Images:</b> Mel-Spectrograms generated from each audio file to feed into CNN models.</li>
            <li><b>CSV Files:</b> One with 30-second audio features, another with 3-second chunks for data augmentation.</li>
        </ul>

        <h3 style="color:#50E3C2;">🎼 What is a Mel-Spectrogram?</h3>
        <p>
            A Mel-Spectrogram is a time-frequency representation of audio using the Mel scale – which aligns with human hearing perception.
        </p>
        <ul>
            <li><b>X-axis:</b> Time</li>
            <li><b>Y-axis:</b> Frequency (Mel scale)</li>
            <li><b>Color Intensity:</b> Amplitude of signal</li>
        </ul>

        

        <h3 style="color:#50E3C2;">⚙️ Steps to Generate Mel-Spectrogram</h3>
        <ol>
            <li><b>Recording Sound:</b> Capture air pressure over time (digital sound).</li>
            <li><b>FFT:</b> Convert time signal to frequency domain using Fast Fourier Transform.</li>
            <li><b>Log Scale Conversion:</b> Apply logarithmic scale to frequencies.</li>
            <li><b>Mel Scale:</b> Adjust frequencies to match human perception.</li>
        </ol>

        <h3 style="color:#50E3C2;">🔍 Why Use Fourier Transform?</h3>
        <ul>
            <li><b>Pattern Detection:</b> Reveals frequency patterns.</li>
            <li><b>Human Perception:</b> Captures pitch better than amplitude.</li>
            <li><b>Feature Extraction:</b> Used in speech/music recognition.</li>
            <li><b>Noise Reduction:</b> Helps remove unwanted frequencies.</li>
            <li><b>Compression:</b> Removes inaudible parts (e.g., in MP3).</li>
        </ul>

        
        <h3 style="color:#50E3C2;">📊 Fourier Transform Visualization (Code)</h3>

    """, unsafe_allow_html=True)

    st.code("""
import numpy as np
import matplotlib.pyplot as plt

fs = 500
t = np.linspace(0, 1, fs, endpoint=False)
freqs = [5, 50, 120]
signal = np.sin(2 * np.pi * freqs[0] * t) + np.sin(2 * np.pi * freqs[1] * t) + np.sin(2 * np.pi * freqs[2] * t)

fft_values = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(t), 1/fs)

plt.figure(figsize=(8, 4))
plt.plot(frequencies[:fs//2], np.abs(fft_values[:fs//2]), 'b')
plt.title("Frequency Spectrum of a Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
    """, language='python')

    st.markdown("""
        <h3 style="color:#50E3C2;">👂 Human Hearing & Mel Scale</h3>
        <p>
            The <b>Mel Scale</b> mimics how the human ear perceives pitch.
        </p>
        <ul>
            <li>Low frequencies are expanded.</li>
            <li>High frequencies are compressed.</li>
        </ul>
        <p>
            <b>Spectrogram:</b> A 2D image showing how sound frequency content evolves over time.
        </p>

        

        <h3 style="color:#50E3C2;">📚 Librosa Library</h3>
        <p><b>Librosa</b> is a Python library for audio analysis.</p>
        <ul>
            <li>Audio Loading</li>
            <li>Feature Extraction (Spectrograms, Mel-spectrograms, MFCCs)</li>
            <li>Visualization tools</li>
        </ul>
        <p><b>Usage:</b> Extract audio features and feed them to ML models.</p>
    """, unsafe_allow_html=True)


    

#Prediction Page
# Prediction Page
elif(app_mode == "Prediction"):
    st.header("🎵 Music Genre Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if test_mp3 is not None:
        # Show Audio Player
        st.audio(test_mp3)

        # Predict Button
        if st.button("Predict"):
            with st.spinner("Analyzing the audio file... 🔍"):
                # Read bytes from uploaded file
                import tempfile

                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(test_mp3.read())
                    tmp_path = tmp_file.name

                # Preprocess and Predict
                X_test = load_and_preprocess_data(tmp_path)
                result_index = model_prediction(X_test)

                st.balloons()
                label = ['blues', 'classical', 'country', 'disco', 'hiphop',
                         'jazz', 'metal', 'pop', 'reggae', 'rock']
                st.markdown(
                    f"**🎧 Model Prediction:** It's a :red[{label[result_index]}] music genre!")


       
