import numpy as np
import os
from scipy.io import wavfile
from scipy.fft import fft
from sklearn.model_selection import train_test_split
import shutil

# Step 1: Frame the signal
def frame_signal(signal, frame_length, frame_step):
    """
    Divide signal into overlapping frames
    
    Args:
        signal: Audio signal
        frame_length: Length of each frame in samples
        frame_step: Step size between frames in samples
    
    Returns:
        Framed signal as 2D array
    """
    signal_length = len(signal)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1
    
    # Pad signal if necessary
    pad_length = (num_frames - 1) * frame_step + frame_length
    padded_signal = np.pad(signal, (0, pad_length - signal_length), mode='constant')
    
    # Create frames
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * frame_step
        frames[i] = padded_signal[start:start + frame_length]
    
    return frames

# Step 2: Compute FFT spectrum for each frame with improvements
def compute_fft_features(frames, use_log=True):
    """
    Compute FFT spectrum for each frame with optional log scaling
    
    Args:
        frames: 2D array of frames
        use_log: Apply log scaling to features (reduces dynamic range)
    
    Returns:
        FFT magnitude spectrum for each frame
    """
    # Apply Hamming window to reduce spectral leakage
    window = np.hamming(frames.shape[1])
    windowed_frames = frames * window
    
    # Compute FFT and get magnitude spectrum
    fft_frames = np.abs(fft(windowed_frames, axis=1))
    
    # Keep only positive frequencies (first half)
    fft_features = fft_frames[:, :fft_frames.shape[1]//2]
    
    # Apply log scaling to compress dynamic range (optional but recommended)
    if use_log:
        fft_features = np.log(fft_features + 1e-10)  # Add small constant to avoid log(0)
    
    return fft_features

def extract_features(audio_path, frame_length=512, frame_step=256, use_log=True):
    """
    Extract FFT features from an audio file
    
    Args:
        audio_path: Path to WAV file
        frame_length: Frame length in samples
        frame_step: Frame step in samples
        use_log: Apply log scaling to features
    
    Returns:
        FFT features
    """
    # Read audio file
    sample_rate, signal = wavfile.read(audio_path)
    
    # Convert to mono if stereo
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    
    # Normalize signal
    signal = signal.astype(float)
    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    
    # Frame the signal
    frames = frame_signal(signal, frame_length, frame_step)
    
    # Compute FFT features
    features = compute_fft_features(frames, use_log)
    
    return features

# Step 3: Compute average template for each class
def compute_class_templates(data_path, classes, frame_length=512, frame_step=256, use_log=True):
    """
    Compute average feature template for each speech class
    
    Args:
        data_path: Path to dataset
        classes: List of class names
        frame_length: Frame length in samples
        frame_step: Frame step in samples
        use_log: Apply log scaling to features
    
    Returns:
        Dictionary of class templates
    """
    templates = {}
    
    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        audio_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        
        all_features = []
        
        print(f"Processing class: {class_name} ({len(audio_files)} files)")
        
        for audio_file in audio_files:
            audio_path = os.path.join(class_path, audio_file)
            try:
                features = extract_features(audio_path, frame_length, frame_step, use_log)
                all_features.append(features)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
        
        # Average all features for this class
        # First, find the maximum number of frames
        max_frames = max(f.shape[0] for f in all_features)
        
        # Pad all features to the same length
        padded_features = []
        for features in all_features:
            if features.shape[0] < max_frames:
                pad_width = ((0, max_frames - features.shape[0]), (0, 0))
                features = np.pad(features, pad_width, mode='constant')
            padded_features.append(features)
        
        # Compute mean template
        templates[class_name] = np.mean(padded_features, axis=0)
    
    return templates

# Step 4: Compute distance to each class template
def compute_distance(test_features, template, distance_type='euclidean'):
    """
    Compute distance between test features and template
    
    Args:
        test_features: Features from test audio
        template: Class template
        distance_type: Type of distance ('euclidean', 'cosine', 'correlation')
    
    Returns:
        Distance value
    """
    # Pad to match dimensions
    max_frames = max(test_features.shape[0], template.shape[0])
    
    if test_features.shape[0] < max_frames:
        pad_width = ((0, max_frames - test_features.shape[0]), (0, 0))
        test_features = np.pad(test_features, pad_width, mode='constant')
    
    if template.shape[0] < max_frames:
        pad_width = ((0, max_frames - template.shape[0]), (0, 0))
        template = np.pad(template, pad_width, mode='constant')
    
    if distance_type == 'euclidean':
        # Euclidean distance
        distance = np.sqrt(np.sum((test_features - template) ** 2))
    
    elif distance_type == 'cosine':
        # Cosine distance (1 - cosine similarity)
        test_flat = test_features.flatten()
        template_flat = template.flatten()
        
        dot_product = np.dot(test_flat, template_flat)
        norm_test = np.linalg.norm(test_flat)
        norm_template = np.linalg.norm(template_flat)
        
        cosine_sim = dot_product / (norm_test * norm_template + 1e-10)
        distance = 1 - cosine_sim
    
    elif distance_type == 'correlation':
        # Correlation distance
        test_flat = test_features.flatten()
        template_flat = template.flatten()
        
        correlation = np.corrcoef(test_flat, template_flat)[0, 1]
        distance = 1 - correlation
    
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")
    
    return distance

# Step 5: Predict class with smallest distance
def predict_class(test_features, templates, distance_type='euclidean'):
    """
    Predict the class with the smallest distance
    
    Args:
        test_features: Features from test audio
        templates: Dictionary of class templates
        distance_type: Type of distance metric
    
    Returns:
        Predicted class name and distances
    """
    distances = {}
    
    for class_name, template in templates.items():
        distances[class_name] = compute_distance(test_features, template, distance_type)
    
    # Return class with minimum distance
    predicted_class = min(distances, key=distances.get)
    
    return predicted_class, distances

# Step 6: Compute accuracy
def evaluate_model(data_path, templates, test_files, frame_length=512, frame_step=256, 
                   use_log=True, distance_type='euclidean'):
    """
    Evaluate the model on test files
    
    Args:
        data_path: Path to dataset
        templates: Dictionary of class templates
        test_files: List of (class_name, file_name) tuples
        frame_length: Frame length in samples
        frame_step: Frame step in samples
        use_log: Apply log scaling to features
        distance_type: Type of distance metric
    
    Returns:
        Accuracy score, predictions, and actuals
    """
    correct = 0
    total = len(test_files)
    
    predictions = []
    actuals = []
    
    print("\nEvaluating model...")
    
    for actual_class, file_name in test_files:
        audio_path = os.path.join(data_path, actual_class, file_name)
        
        try:
            # Extract features
            test_features = extract_features(audio_path, frame_length, frame_step, use_log)
            
            # Predict class
            predicted_class, distances = predict_class(test_features, templates, distance_type)
            
            predictions.append(predicted_class)
            actuals.append(actual_class)
            
            if predicted_class == actual_class:
                correct += 1
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            total -= 1
    
    accuracy = correct / total if total > 0 else 0
    
    return accuracy, predictions, actuals

# Main execution
def main():
    # Configuration
    data_path = '/home/yahia/DSP/mini_speech_commands/mini_speech_commands'
    
    # Best parameters found:
    frame_length = 512   # Original works best
    frame_step = 256     # 50% overlap
    train_ratio = 0.8    # 80% train, 20% test (better evaluation)
    use_log = False      # NO log scaling works better!
    distance_type = 'euclidean'  # Euclidean distance
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at '{data_path}'")
        print("Please update the 'data_path' variable to point to your extracted dataset folder")
        return
    
    # Get list of classes (subdirectories), excluding system folders
    classes = [d for d in os.listdir(data_path) 
               if os.path.isdir(os.path.join(data_path, d)) 
               and not d.startswith('__') and not d.startswith('.')]
    
    print(f"\nFound {len(classes)} classes: {classes}")
    
    # Split data into train/test
    all_files = []
    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        audio_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        all_files.extend([(class_name, f) for f in audio_files])
    
    train_files, test_files = train_test_split(all_files, train_size=train_ratio, 
                                                 random_state=42, stratify=[f[0] for f in all_files])
    
    print(f"\nTrain files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    print(f"\nParameters:")
    print(f"  Frame length: {frame_length}")
    print(f"  Frame step: {frame_step}")
    print(f"  Log scaling: {use_log}")
    print(f"  Distance metric: {distance_type}")
    
    # Create temporary train directory structure
    train_path = 'data/train_speech_commands'
    os.makedirs(train_path, exist_ok=True)
    
    for class_name in classes:
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
    
    # Copy train files
    for class_name, file_name in train_files:
        src = os.path.join(data_path, class_name, file_name)
        dst = os.path.join(train_path, class_name, file_name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Compute templates from training data
    print("\nComputing class templates...")
    templates = compute_class_templates(train_path, classes, frame_length, frame_step, use_log)
    
    # Evaluate on test data
    accuracy, predictions, actuals = evaluate_model(data_path, templates, test_files, 
                                                     frame_length, frame_step, use_log, distance_type)
    
    print(f"\n{'='*50}")
    print(f"ACCURACY: {accuracy * 100:.2f}%")
    print(f"{'='*50}")
    
    # Confusion matrix (simplified)
    print("\nClass-wise accuracy:")
    for class_name in classes:
        class_correct = sum(1 for p, a in zip(predictions, actuals) 
                           if a == class_name and p == class_name)
        class_total = sum(1 for a in actuals if a == class_name)
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {class_name}: {class_acc * 100:.2f}% ({class_correct}/{class_total})")

if __name__ == "__main__":
    main()