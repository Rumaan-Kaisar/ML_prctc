
# mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# change directories:
%cd /content/drive/MyDrive/Colab\ Notebooks
!pwd    # current working directory now changed
!ls



'''
# raw files import edit

output_file = f"data{index}_p{patient_id}_{ground_truth}.pkl"
print(f"total data points at {output_file} is {len(au_list)}")

# Save as a pickle file for efficient storage
with open(output_file, 'wb') as f:
    pickle.dump(extructed_data, f)

print(f"Processed data saved to {output_file}")

# ================== Processing Loop for All Patients ==================
import json

for i in range(len(patient_id)):
    with open(f'{folder_path}data/{patient_id[i]}.json', 'r') as f:
        data_pid = json.load(f)

    data_extractor(data_pid, start_ts[i], end_ts[[i][0]], ground_truth[i], i, patient_id[i])


extructed_data = {
    'feature': (contours_list, au_list, landmarks_list, headEulerAngle_list, classification_list),
    'ground_truth': ground_truth
}

'''



# =====================================================================================================
# ------------------------    Accessing Data    ------------------------
# =====================================================================================================


#@title ----  import sample file  ----
import pandas as pd

main_data_folder_path = '../64_facePsy/dataset/dataset/'
# main_data_path: f'{main_data_folder_path}data/{patient_id[i]}.json'

extracted_data_path = './extracted_face_data/'

patient_id = ['P08', 'P08', 'P10', 'P10', 'P12', 'P12', 'P13', 'P13', 'P14', 'P15', 'P15', 'P16', 'P16', 'P17', 'P17', 'P18', 'P18', 'P19', 'P19', 'P20', 'P20', 'P21', 'P21', 'P23', 'P23', 'P24', 'P24', 'P25', 'P27', 'P28', 'P29', 'P29', 'P30', 'P30', 'P31', 'P31', 'P33', 'P33', 'P34', 'P35', 'P35', 'P36', 'P38', 'P38']
ground_truth = [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# Load the CSV file
# csv_file_path = folder_path+'groundtruth/phq9.csv'  # Replace with the path to your CSV file
# data_1 = pd.read_csv(csv_file_path)




# @title load all extracted data
import pickle

for i in range(len(patient_id)):
    # Generate the filename
    file_path = extracted_data_path + f"data{i}_p{patient_id[i]}_{ground_truth[i]}.pkl"

    # Load the file
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)



# load a single extracted data
import pickle

fp = extracted_data_path + f"data{0}_p{patient_id[0]}_{ground_truth[0]}.pkl"
# Load the file
with open(fp, 'rb') as f:
    data_P08_1 = pickle.load(f)

print(data_P08_1['ground_truth'])

print(len(data_P08_1['feature'][0][0]))




#@title ----  Arrange Features in Dictionary Form  ----

# Function for Dictionary Representation of Data-features

def convert_to_named_features(old_data):
    """Convert tuple-based feature structure to named dictionary structure"""
    feature_tuple = old_data['feature']

    new_data = {
        'feature': {
            'contours': feature_tuple[0],  # contours_list
            'au': feature_tuple[1],       # au_list
            'landmarks': feature_tuple[2], # landmarks_list
            'headEulerAngle': feature_tuple[3], # headEulerAngle_list
            'classification': feature_tuple[4]   # classification_list
        },
        'ground_truth': old_data['ground_truth']
    }
    return new_data



# Loading and Converting
import pickle
import os

def load_and_convert(filepath):
    """Load a pickle file and convert its structure"""
    with open(filepath, 'rb') as f:
        old_data = pickle.load(f)

    # Convert the data structure
    new_data = convert_to_named_features(old_data)

    # Optional: Save the converted version back
    # new_filename = f"converted_{os.path.basename(filepath)}"
    # with open(new_filename, 'wb') as f:
    #     pickle.dump(new_data, f)

    return new_data

"""___
## data loading test
"""

# Example usage:
converted_data = load_and_convert('data0_pP08_0.pkl')
print(converted_data['feature']['au'][0])  # Now accessible by name

data_P08_1_dict = convert_to_named_features(data_P08_1)

print(data_P08_1_dict['ground_truth'])

print(len(data_P08_1_dict['feature']['contours'][0]))

data_P08_1_dict['feature']['contours'][0][0]

print(len(data_P08_1_dict['feature']['contours']))
print(len(data_P08_1_dict['feature']['au']))
print(len(data_P08_1_dict['feature']['landmarks']))
print(len(data_P08_1_dict['feature']['headEulerAngle']))
print(len(data_P08_1_dict['feature']['classification']))





# =====================================================================================================
# ------------------------    sampling    ------------------------
# =====================================================================================================


# Now our goal is to choose representative samples randomly and uniformly 
    # from the 9,875 contours along with their corresponding features.
        # How can we achieve this?
        # What will be the code?


# ------------    Notable Points    ------------

# Uniform Sampling: 
    # Ensures a fair selection from all 9,875 samples.

# Reproducibility (using seed ?):  
    # Uses np.random.seed() to get the same results every run.
    # If you don't need consistency across runs, skipping the seed ensures a truly random selection each time.
    # in our case we won't use seed

# No Replacement:   
    # Prevents duplicate selections.
    # Using without replacement ensures a diverse and representative subset of the data.


# To choose representative samples randomly and uniformly 
    # from the 9,875 contours (and their corresponding features)
        # Determine the number of samples (e.g., N = 500).
        # Use uniform random sampling (without replacement) to pick N indices.
        # Extract the selected samples from all feature lists.


# avoid a seed?
    # If you need diverse samples across multiple runs (e.g., training models with different data splits).
    # If you're conducting real-world experiments where randomness should be truly unpredictable.

# ======== with seed ========
import numpy as np

def sample_uniformly(data_dict, num_samples=500, seed=42):
    """
    Randomly selects 'num_samples' indices from the dataset in a uniform manner.
    Extracts corresponding feature values.

    Args:
        data_dict (dict): Dictionary containing 'feature' and its sub-keys.
        num_samples (int): Number of samples to select (default: 500).
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A new dictionary containing only the sampled data.
    """
    np.random.seed(seed)  # Ensure reproducibility
    
    total_samples = len(data_dict['feature']['contours'])  # 9875 in this case
    
    # Select 'num_samples' unique indices uniformly
    sampled_indices = np.random.choice(total_samples, num_samples, replace=False)

    # Extract the corresponding samples
    sampled_data = {
        'sample': sampled_indices,
        'feature': {
            'contours': [data_dict['feature']['contours'][i] for i in sampled_indices],
            'au': [data_dict['feature']['au'][i] for i in sampled_indices],
            'landmarks': [data_dict['feature']['landmarks'][i] for i in sampled_indices],
            'headEulerAngle': [data_dict['feature']['headEulerAngle'][i] for i in sampled_indices],
            'classification': [data_dict['feature']['classification'][i] for i in sampled_indices],
        },
        'ground_truth': data_dict['ground_truth']
    }

    return sampled_data

# Example Usage
sampled_data_P08_1 = sample_uniformly(data_P08_1_dict, num_samples=500)




# ======== NO seed (TESTED) ========

import numpy as np

def sample_uniformly(data_dict, num_samples=500):
    """
    Randomly selects 'num_samples' indices from the dataset in a uniform manner.
    Extracts corresponding feature values.

    Args:
        data_dict (dict): Dictionary containing 'feature' and its sub-keys.
        num_samples (int): Number of samples to select (default: 500).

    Returns:
        dict: A new dictionary containing only the sampled data.
    """
    total_samples = len(data_dict['feature']['contours'])  # 9875 in this case
    
    # Select 'num_samples' unique indices uniformly without a fixed seed
    sampled_indices = np.random.choice(total_samples, num_samples, replace=False)

    # Extract the corresponding samples
    sampled_data = {
        'sample': sampled_indices,
        'feature': {
            'contours': [data_dict['feature']['contours'][i] for i in sampled_indices],
            'au': [data_dict['feature']['au'][i] for i in sampled_indices],
            'landmarks': [data_dict['feature']['landmarks'][i] for i in sampled_indices],
            'headEulerAngle': [data_dict['feature']['headEulerAngle'][i] for i in sampled_indices],
            'classification': [data_dict['feature']['classification'][i] for i in sampled_indices],
        },
        'ground_truth': data_dict['ground_truth']
    }

    return sampled_data


# Example Usage
sampled_data_P08_1 = sample_uniformly(data_P08_1_dict, num_samples=500)


# --------  examine sample vs population  --------
sampled_data_P08_1['sample'][:10]
print(sampled_data_P08_1['ground_truth'])

# in sample
print(sampled_data_P08_1['feature']['contours'][0][:3])
print(sampled_data_P08_1['feature']['au'][0][:3])
print(sampled_data_P08_1['feature']['landmarks'][0][:1])
print(sampled_data_P08_1['feature']['headEulerAngle'][0])
print(sampled_data_P08_1['feature']['classification'][0])


data_point = sampled_data_P08_1['sample'][0]
print(f"data point to check: {data_point}")

# in population
population_data = data_P08_1

print(population_data['feature'][0][data_point][:3])
print(population_data['feature'][1][data_point][:3])
print(population_data['feature'][2][data_point][:1])
print(population_data['feature'][3][data_point])
print(population_data['feature'][4][data_point])






# ---------------------------------------------------------------------------------
# selecting proper functions for uniform-random sampling

# METHOD 1: 
# np.round(np.random.uniform(low=0, high=len(feature_1[i])-1, size=num_images)).astype(int)
    # Duplicates possible:
        # Even with replace=False logic, rounding can cause the same integer to be chosen multiple times, especially for small ranges.
    # Not guaranteed to produce uniformly spaced integers (depends on rounding)
    # Can lead to index bias at edges due to rounding
    # More useful for approximate sampling or when you allow repeats
    # Wasted Computation:
        # need more computation for "Float -> Rounded to Integer" and 
        # You're doing:     Float generation → Rounding → Type casting to int

    # Bias Due to Rounding: 
        # "Approximate, not perfect due to rounding"
        # eg: 2.6, 2.7 both gives 3 make a seslection of index 3 multiple time
        # any float between 2.5 and 3.5 rounds to 3.  introducing a non-uniform distribution.


# METHOD 2 (Better Alternative):
# np.random.choice(total_samples, num_samples, replace=False)
    # No duplicates (since replace=False)
        # replace=False is critical to ensure no duplicates.
        # Reproducibility: Add a random seed for consistent results:
    # Uniform probability for each number
    # Returns integers directly
    # Guaranteed to be valid indices
    # Best when you need unique random selections from a set


# METHOD 3 (Alternative 2, Clean & faster than METHOD 2):
# indices = np.random.permutation(len(data))[:500]
    # is more efficient for large datasets
    # np.random.permutation is uniform (n = total population)
    # It returns a uniformly random permutation of the integers [0, 1, ..., n-1]
    # we can use it during testing phase by slicing it by 500
    # Uniformly random: Every possible permutation has an equal chance of appearing.
    # Non-repeating: You’ll never get duplicates.
    # Shuffled: It randomly rearranges the entire list of indices.
    # Use np.random.permutation when:
        # You want to shuffle data.
        # You need a random, non-repeating sample (e.g., first 500 from permuted list).
        # You're avoiding bias in sample order.
import numpy as np

# Generate shuffled indices
    # np.random.permutation shuffles indices without replacement.
    # Slicing [:500] directly gives unique samples.
n_total = len(your_dataset)  # Total number of data points
shuffled_indices = np.random.permutation(n_total)

# Select first 500 shuffled indices
sample_indices = shuffled_indices[:500]

# Get the actual samples
sampled_data = your_dataset[sample_indices]




# Note: 
    # Use either one of METHOD 2 or METHOD 3, not both.
    # You're shuffling then randomly selecting again — redundant.
    # Adds complexity without any extra value.
    # METHOD 2 or METHOD 3 gives almost same result
    
    # Use METHOD 3 (np.random.permutation) if:
        # You need the entire dataset shuffled (e.g., for train/val/test splits later).
        # You want explicit control over shuffling.
    
    # Use METHOD 2 (np.random.choice) if:
        # You only need a single sample of 500 and want concise code.





# ---------------------------------------------------------------------------------
# sample from SEQUENTIAL FRAMES:

# If your contour data (or any facial feature data) contains consecutive frames from the same video segment, you're at risk of having:
    # Highly redundant samples
    # Biased SOM training (since similar-looking frames cluster easily)
    # Lower data diversity

# you should break or reduce this frame sequence bias to make your sampling more representative and diverse.

# why?
    # Consecutive frames are very similar, especially if facial expression or head position changes slowly.
    # Sampling such sequences results in redundant information.
    # SOM may learn local redundancy instead of global structure.

# METHODS:
    # 1. Sample with spacing (Frame Skipping)

    # 2. Random sampling with deduplication (optional entropy filtering)
            # You randomly pick, but optionally compare distance (e.g., using Euclidean or cosine) 
            # between samples to avoid selecting near-duplicates.
            # This is slower, but ensures diversity.

    # 3. Segment-aware sampling (If segment info is available)
            # If your data includes metadata about the video ID or timestamps, you can:
            # Sample only 1-2 frames from each segment
            # Prevent oversampling from a single video






# ---------------------------------------------------------------------------------

# near-duplicate samples filtering:
    # How can we combine random sampling with duplicate filtering?

    # How to randomly select samples (like with np.random.choice(...)), 
        # but also check that the selected samples aren't too similar to each other — for example,
        # by measuring distance between them (using cosine or Euclidean). 
    # This helps make the selection more diverse, even though it's a bit slower.


# combine "random sampling" with a diversity check to avoid selecting "near-duplicate samples"

# use something like:
        # sampled_indices = np.random.choice(total_samples, num_samples, replace=False)
    # but filter it further to ensure selected samples are not too similar (e.g., based on cosine or Euclidean distance).


#@title --------  hybrid sampling with distance check  --------

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def diverse_random_sample(features, num_samples=500, distance_threshold=0.1):
    """
    Randomly sample `num_samples` feature vectors from `features`,
    avoiding near-duplicates based on cosine distance.
    
    Args:
        features (List or np.ndarray): List of feature vectors (N x D).
        num_samples (int): Number of samples to draw.
        distance_threshold (float): Min cosine distance between any two samples.

    Returns:
        List[int]: Indices of selected diverse samples.
    """
    total_samples = len(features)
    selected_indices = []
    all_indices = np.random.permutation(total_samples)

    for idx in all_indices:
        if len(selected_indices) == 0:
            selected_indices.append(idx)
        else:
            candidate_vec = features[idx].reshape(1, -1)
            selected_vecs = np.array([features[i] for i in selected_indices])
            distances = cosine_distances(candidate_vec, selected_vecs)

            # Only add if it's not too close to any selected one
            if np.all(distances > distance_threshold):
                selected_indices.append(idx)

        if len(selected_indices) >= num_samples:
            break

    return selected_indices


# How to use it:
features = np.array(data_dict['feature']['au'])  # or flattened face images, or embeddings
selected_indices = diverse_random_sample(features, num_samples=500, distance_threshold=0.1)


# Now you can use selected_indices to extract from your dataset
sampled_data = {
    'feature': {
        k: [data_dict['feature'][k][i] for i in selected_indices]
        for k in data_dict['feature']
    },
    'ground_truth': [data_dict['ground_truth'][i] for i in selected_indices]
}

# NOTE:
    # Try both cosine distance and Euclidean distance — "cosine" works well for "normalized features".
    # You can tweak the "distance_threshold" to control "diversity level".
    # It's slower than pure random sampling — but worth it if you're optimizing for entropy/diversity.


# Is "near-duplicate sample filtering" good or bad for representative sampling?
    # For example, if the population contains specific proportions of certain facial expressions in a particular sequence, 
    # does uniform random sampling preserve these proportions? 
    # And if we apply SOM on such a sample, will those proportions still be represented?


""" 
    # ----------  Use filtering with random sampling?   -----------

    Near-duplicate filtering can be both helpful and risky depending on your goal:

    Good for diversity-oriented tasks: 
        If your objective is to Capture the broadest range of variation in facial expressions,
        Avoid over-representing repetitive frames (e.g., many similar smiles from the same sequence),
        Or train SOM to learn distinct facial patterns,
        May Improve SOM: By reducing noise (e.g., removing redundant "neutral" expressions).
        SOM shows the overall structure well but might miss details in categories with many similar samples.

    Then filtering out near-duplicates helps reduce redundancy and increases entropy (diversity of information).
    This is especially valuable when you have limited capacity (e.g., small sample size) and want to maximize meaningful variation.



    Bad for statistical representativeness:
        Maintain natural distribution (e.g., a person smiling 60% of the time in a sequence),
        Reflect true frequency and transitions in real-world data,
        Or model behavior patterns and durations,

    Then removing near-duplicates alters the true proportions.
    "Uniform random sampling without filtering" is BETTER here—
        it ensures each frame has an equal chance of being picked, preserving natural occurrence.



    Regarding SOM:
        SOM is unsupervised and density-sensitive—it clusters based on feature distribution:
        If your sample is balanced, SOM reflects population structure well.
        If sampling preserves proportions, SOM will reflect them.
        If you aggressively filter near-duplicates, SOM may emphasize rare expressions and "under-represent common patterns".
        For Unfiltered Data: SOM might show clusters dominated by duplicates (e.g., many nearly identical "neutral" faces).



    REMARKS:
        In our case "uniform random sampling without filtering" is likely the best choice. Here's why:

        Why it works well:
            Goal: We want to preserve the natural distribution of facial expressions.

            Uniform sampling without filtering ensures that:
                The proportions of different face types (happy, neutral, sad, etc.) in your population are retained.
                SOM will then map this distribution into clusters, giving you a probability-like representation of expression diversity.


        If you apply filtering (e.g., remove near-duplicates):
            You may unintentionally distort the original distribution.
            Expressions that occur frequently (like neutral faces) may be underrepresented, and rare ones overemphasized.



    SO USE:
            sampled_indices = np.random.choice(total_samples, num_samples, replace=False)
        
        and use this sample directly for training SOM, to capture realistic population proportions.

"""




# =====================================================================================================
# ------------------------    Image: contour and enocoding features    ------------------------
# =====================================================================================================

# image from contour + probability vectors

# -----------------------------    Tr1 1: NO encoded AU & Prob    -----------------------------

# ========  FIG 1: only eye-lip  ========
from PIL import Image, ImageDraw

# Define image size
image_size = 64

def convert_to_img(X):
    x_coords = np.array([point['x'] for point in X])
    y_coords = np.array([point['y'] for point in X])

    # Min-max scaling for x and y independently
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_normalized = (x_coords - x_min) / (x_max - x_min)
    y_normalized = (y_coords - y_min) / (y_max - y_min)

    landmarks = np.column_stack((x_normalized, y_normalized))

    # Scale normalized landmarks to fit within the 64x64 image
    landmarks_scaled = np.round(landmarks * (image_size - 1)).astype(int)

    face_parts = [
        # ((0, 35), 'Face oval', (255, 255, 0, 128)),  # Yellow
        ((36, 40), 'Left eyebrow (top)', (0, 0, 255, 128)),  # Blue
        ((41, 45), 'Left eyebrow (bottom)', (0, 0, 255, 128)),
        ((46, 50), 'Right eyebrow (top)', (128, 0, 128, 128)),  # Purple
        ((51, 55), 'Right eyebrow (bottom)', (128, 0, 128, 128)),
        ((56, 71), 'Left eye', (255, 0, 0, 128)),  # Red
        ((72, 87), 'Right eye', (255, 0, 0, 128)),
        ((88, 96), 'Upper lip (bottom)', (255, 165, 0, 128)),  # Orange
        ((97, 105), 'Lower lip (top)', (255, 165, 0, 128)),
        ((106, 116), 'Upper lip (top)', (0, 255, 255, 128)),  # Cyan
        ((117, 125), 'Lower lip (bottom)', (0, 255, 255, 128)),
        ((131, 131), 'Left cheek (center)', (128, 128, 128, 128)),  # Gray
        ((132, 132), 'Right cheek (center)', (255, 192, 203, 128)),  # Pink
    ]

    # Create a blank RGBA image
    img = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for indices, label, color in face_parts:
        start, end = indices
        region_points = [(x, y) for x, y in landmarks_scaled[start:end + 1]]

        # Fill the polygon formed by the region points
        if len(region_points) > 2:  # At least 3 points required to form a polygon
            draw.polygon(region_points, fill=color)

    # Convert to numpy array and normalize to 0-1 range (RGB only)
    img_array = np.array(img)[:, :, :3] / 255.0  # Exclude alpha channel, normalize RGB
    return img_array


imme = convert_to_img(data_3[0]['contours'] )

# Display the image
plt.imshow(imme)
plt.axis("off")
plt.title("Facial Landmarks & Contours for data_3[0]")
plt.show()



# ========  FIG 2: contours and landmarks  ========

import numpy as np
import matplotlib.pyplot as plt

# ========== Process a Single Entry ==========
single_data = data_3[0]

# Extract contours & landmarks
single_contours = single_data.get('contours', [])
single_landmarks = single_data.get('landmarks', [])

if not single_contours and not single_landmarks:
    print("No contours or landmarks found.")
else:
    plt.figure(figsize=(4, 4))

    # Process contours (black small dots)
    if single_contours:
        x_contours = np.array([point['x'] for point in single_contours])
        y_contours = np.array([point['y'] for point in single_contours])

        # Normalize
        x_min, x_max = x_contours.min(), x_contours.max()
        y_min, y_max = y_contours.min(), y_contours.max()
        x_contours = ((x_contours - x_min) / (x_max - x_min) * 63).astype(int)
        y_contours = ((y_contours - y_min) / (y_max - y_min) * 63).astype(int)

        # Plot contours as black dots
        plt.scatter(x_contours, y_contours, color="black", s=5, label="Contours")

    # Process landmarks (red big dots)
    if single_landmarks:
        x_landmarks = np.array([point['x'] for point in single_landmarks])
        y_landmarks = np.array([point['y'] for point in single_landmarks])

        # Normalize
        x_landmarks = ((x_landmarks - x_min) / (x_max - x_min) * 63).astype(int)
        y_landmarks = ((y_landmarks - y_min) / (y_max - y_min) * 63).astype(int)

        # Plot landmarks as red dots
        plt.scatter(x_landmarks, y_landmarks, color="red", s=40, label="Landmarks")

    # Final plot settings
    plt.xlim(0, 63)
    plt.ylim(0, 63)
    plt.gca().invert_yaxis()  # Match image coordinate system
    plt.legend()
    plt.title("Contours (Black) & Landmarks (Red)")

    plt.legend().set_visible(False)  # Hide legend
    plt.show()







# =====================================================================================================
# ----------------  METADATA, Sample size & Split  ----------------
# =====================================================================================================

vc = [9875, 5900, 1638, 2542, 2036, 1085, 7532, 280, 10007, 3509, 2452, 11881, 10347, 1119, 0, 3201, 3412, 465, 497, 2185, 1799, 2481, 2243, 3540, 3440, 5518, 2155, 560, 4, 2890, 13038, 16197, 9036, 10052, 4108, 1423, 7039, 6073, 4153, 7687, 7214, 9576, 3963, 1940]

import numpy as np

def smart_sampling(vc_list, sample_size_dict):
    """
    Smart sampling based on dataset size category.

    Args:
        vc_list (list): List of population sizes.
        sample_size_dict (dict): How many samples to draw for each group size.

    Returns:
        dict: Mapping of index -> sampled indices for each valid group.
    """
    samples = {}

    for idx, size in enumerate(vc_list):
        if size <= 500:
            print(f"Skipping ID {idx} (Tiny group with {size} samples).")
            continue
        
        # Determine category
        if 501 <= size <= 3000:
            category = 'small'
        elif 3001 <= size <= 7000:
            category = 'medium'
        elif 7001 <= size <= 12000:
            category = 'large'
        else:  # size > 12000
            category = 'very_large'
        
        # Determine how many samples to draw
        draw_size = min(sample_size_dict[category], size)
        
        # Draw without replacement
        sampled_indices = np.random.choice(size, draw_size, replace=False)
        
        samples[idx] = {
            'category': category,
            'size': size,
            'sampled_indices': sampled_indices.tolist()  # Save as list if needed later
        }
        
        print(f"Sampled {draw_size} from ID {idx} ({category}, total {size}).")
    
    return samples

# Example configuration:
sample_size_rules = {
    'small':  min(500, 1000),       # For small groups, sample up to 500 (but safe)
    'medium': 1000,                 # Medium groups, sample 1000
    'large': 1500,                  # Large groups, sample 1500
    'very_large': 2000              # Very large groups, sample 2000
}

# Now call it:
vc = [9875, 5900, 1638, 2542, 2036, 1085, 7532, 280, 10007, 3509, 2452, 11881, 10347, 1119, 0, 3201, 3412, 465, 497, 2185, 1799, 2481, 2243, 3540, 3440, 5518, 2155, 560, 4, 2890, 13038, 16197, 9036, 10052, 4108, 1423, 7039, 6073, 4153, 7687, 7214, 9576, 3963, 1940]

samples = smart_sampling(vc, sample_size_rules)

# samples is a dictionary with sampling info for all non-tiny groups


""" 
Class | Size range (# samples) | Example meaning
Tiny | 0-500 | Very small group
Small | 501-3000 | Manageable sample
Medium | 3001-7000 | Good for small models
Large | 7001-12000 | Solid big population
Very Large | >12000 | Extremely big group


Summary
Tiny = 5 groups
Small = 16 groups
Medium = 13 groups
Large = 9 groups
Very Large = 2 groups


Group Size | Approximate Sampling % | Why?
Small (500-3000) | ~10-30% | Small groups are sampled relatively heavily to ensure enough data.
Medium (3000-7000) | ~15-30% | Medium groups sampled moderately.
Large (7000-12000) | ~10-20% | Large groups need fewer % because they already have lots of variety.
Very Large (>12000) | ~10-15% | Very large groups sampled lightly to stay efficient.

500-3000: ~10-30% : 500
3000-5000: ~15-30% : 1000
5000-7000: ~18-30% : 1500
7000-8000: ~18-30% : 2000
8000-11000: ~23-30% : 2500
1100-17000: ~18-27% : 3000


def decide_sample_size(group_size):
    
    # Decide sample size based on updated group size classification.
    
    if 500 <= group_size < 3000:
        return 500
    elif 3000 <= group_size < 5000:
        return 1000
    elif 5000 <= group_size < 7000:
        return 1500
    elif 7000 <= group_size < 8000:
        return 2000
    elif 8000 <= group_size < 11000:
        return 2500
    elif 11000 <= group_size <= 17000:
        return 3000
    else:
        return 0  # 0 for tiny groups (like <500)


        
        
================    Selecting number of samples:    ================
make a scale: 
    500 < 1000 -> 1 time

2/3 for train 1/3 for test

SOM (Self-Organizing Map) behaves differently from traditional supervised models, 
so the intuition for training vs. testing size can shift.

In SOM (Unsupervised Learning):
    Training goal: Learn the general topological structure of the input data space.
    Projection goal: Assign or interpret new samples by projecting them onto the trained SOM map.

So, keeping the training set smaller than the projection set is acceptable and even beneficial in some scenarios:


Advantages:
    A smaller, diverse training set helps the SOM generalize better and avoid overfitting to frequent/redundant patterns.
    A larger projection set gives you better visualization and understanding of how various patterns distribute across the map.

You're not tuning weights for classification but building a topology — so representative samples matter more than volume.
    
Total Sets	Train Sets	Projection Sets
1	        1	        0
2	        1	        1
3	        1	        2
4	        2	        2
5	        2	        3
6	        3	        3

TUNING: we can tune this, and keep the training sets even smaller. 
        ->  just using 1 set per individual
        ->  also adjust the SOM "grids", "iterations" accroding to train set size

        


================    metadata    ================
Use JSON: Since you're storing hierarchical information (parameters, sample indices, training results), JSON is definitely better. 
You can track, update, and re-read it cleanly.

Where to Save the Metadata?
    Best place: Your Google Drive mounted in Colab.

    Why?
        Files persist after runtime disconnects.
        You can track history across sessions.
        You can sync/run multiple experiments and store logs in one place.

If you want to track incremental progress (e.g., training in stages or iterations), you can also store:

        "progress": {
            "iteration": 200,
            "last_saved": "2025-05-13T15:12"
        }

----------------------------------------------
        

    Make a sequence of metadata files:
    initial: TRACK first SOM -> save to a pickle file name "SOM1 - meta data"


    for each iteration a SOM generated


    SAMPLE_metadta (update at each increment + keep older entries, in dict form)
        {
            how many times patinet ids used for sampling from begginning: [{id_i: count}] (count ids been used for sample draw)
            som file: Corresponding som saved files
            sample_pre_train (index), 
            sample_post_train (index),
        }


    # separate metadata for seperate SOM, name it same as corresponding SOM
    SOM_metadata: 
        sample_pre_train (index), list of dictionary form: {patient_id: , selecte}
        sample_post_train (index),

        saved trained som file name
        saved vector file name

        cluster_pre_train (index), 
        cluster_post_train (index),

    Save data
        save trained som
        save vectors

        
----------------------------------------------
        

0. Select proper dataset:
        Add these info to metadata:
        Test:
            data2_pP10_1.pkl = 1638
            data3_pP10_0.pkl = 2542

        Train:
            load: 
                load the datasets except "Test"
            marge: 
                data6_pP13_0.pkl + data7_pP13_0.pkl both has ground truth 0
                data17_pP19_1.pkl + data18_pP19_1.pkl  both has ground truth 1
            avoid: 
                data14_pP17_0.pkl, 
                data28_pP27_0.pkl
            name the data:
                extract the ID and ground truth name the data
                eg: data2_pP10_1.pkl means patient ID: P10, ground truth (following last '_'): 1, seperated in file name by '_p' and '_'
                rename to : trainData_2_P10_1
                also rename the marged files too, since patient ID and ground truth are same

1. Draw sample: 
                fix how many times the sample can be drawn -> use the above kind of decide_sample_allocation()
                draw sample

2. split sample (use single sample to train the SOM if dataset is too small, other bigger datat for test)
   [hard code it if needed]

3. update sample metadata:
        ->  train set: {id_n : [list of sample index-set from id_n]}, i.e. [set1, set2, etc]
        ->  projection set: {id_n : [list of sample index-set from id_n]}, i.e. [set1, set2, etc]
        ->  "iteration number": "iteration number" to calculate how mnay times sample drawn (now + previously): {id_n : [times_for_train, times_for_projection]}
        ->  corresponding trained SOM file name: SOM file name

4. convert contours to image all samples

5. train the SOM from train set

6. get probabilty vector for "train set", projecting the train set via SOM

7. get probabilty vector for "projection set", projecting the train set via SOM

8. Save:    save trained som
            save vectors seperately id_wise (trained, test)

9. update SOM metadata:
    SOM_metadata: 
        ->  train set: {id_n : [list of sample index-set from id_n]}, i.e. [set1, set2, etc]
        ->  projection set: {id_n : [list of sample index-set from id_n]}, i.e. [set1, set2, etc]
    
        saved trained som file name
        saved vector file name

        cluster_trin: {id_n : list of [list of clusters "index-only" from id_n]}, i.e. [[cluster1_of_set1, cluster2_of_set1, etc], [cluster1_of_set2, cluster2_of_set2, etc], etc]
        cluster_project: {id_n : list of [list of clusters "index-only" from id_n]}, i.e. [[cluster1_of_set1, cluster2_of_set1, etc], [cluster1_of_set2, cluster2_of_set2, etc], etc]

"""


"""  

Task: Select proper dataset:
        Test:
            data2_pP10_1.pkl = 1638
            data3_pP10_0.pkl = 2542

        Train:
            load: 
                load the datasets except "Test"
            marge: 
                data6_pP13_0.pkl + data7_pP13_0.pkl both has ground truth 0
                data17_pP19_1.pkl + data18_pP19_1.pkl  both has ground truth 1
            avoid: 
                data14_pP17_0.pkl, 
                data28_pP27_0.pkl
            name the data:
                extract the ID and ground truth name the data
                eg: data2_pP10_1.pkl means patient ID: P10, ground truth (following last '_'): 1, seperated in file name by '_p' and '_'
                rename to : dt_2_P10_1
                also rename the marged files too, since patient ID and ground truth are same

        Add those info to metadata.
        make these all in a function called: data_splitter()        
"""

# =================================  Edit: 31-May-2025  ========================================

data_name = ["data0_pP08_0", "data1_pP08_0", "data2_pP10_1", "data3_pP10_0", "data4_pP12_1", "data5_pP12_1", "data6_pP13_0", "data7_pP13_0", "data8_pP14_0", "data9_pP15_0", "data10_pP15_1", "data11_pP16_0", "data12_pP16_0", "data13_pP17_1", "data14_pP17_0", "data15_pP18_1", "data16_pP18_1", "data17_pP19_1", "data18_pP19_1", "data19_pP20_0", "data20_pP20_0", "data21_pP21_0", "data22_pP21_1", "data23_pP23_0", "data24_pP23_0", "data25_pP24_1", "data26_pP24_1", "data27_pP25_0", "data28_pP27_0", "data29_pP28_0", "data30_pP29_0", "data31_pP29_0", "data32_pP30_1", "data33_pP30_1", "data34_pP31_0", "data35_pP31_0", "data36_pP33_0", "data37_pP33_0", "data38_pP34_0", "data39_pP35_0", "data40_pP35_0", "data41_pP36_0", "data42_pP38_0", "data43_pP38_0"]

data_size = [9875, 5900, 1638, 2542, 2036, 1085, 7532, 280, 10007, 3509, 2452, 11881, 10347, 1119, 0, 3201, 3412, 465, 497, 2185, 1799, 2481, 2243, 3540, 3440, 5518, 2155, 560, 4, 2890, 13038, 16197, 9036, 10052, 4108, 1423, 7039, 6073, 4153, 7687, 7214, 9576, 3963, 1940]


"""  
MARGING:
dataset = {feature, ground_truth}
feature is again a doctionary of 5 features: contours_list, au_list, landmarks_list, headEulerAngle_list, classification_list
ground truth remain same during marge, since both dataset has same ground truth
our main job is to marge the two datasets- contours_list, au_list, landmarks_list, headEulerAngle_list, classification_list

but first we need to use following functions to modify the datasets into proper dictionary form, then we rename the dataset and marge (just rename & marge, no metadata)
use following code:


"""

# =====================================================================================================
# Step 0: 
# =====================================================================================================
# -----------------------------------------------  train test split and save splitted data  -------------------------------------------------------
# mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# change directories:
%cd /content/drive/MyDrive/Colab\ Notebooks
!pwd    # current working directory now changed
!ls


extracted_data_path = './extracted_face_data/'

patient_id = ['P08', 'P08', 'P10', 'P10', 'P12', 'P12', 'P13', 'P13', 'P14', 'P15', 'P15', 'P16', 'P16', 'P17', 'P17', 'P18', 'P18', 'P19', 'P19', 'P20', 'P20', 'P21', 'P21', 'P23', 'P23', 'P24', 'P24', 'P25', 'P27', 'P28', 'P29', 'P29', 'P30', 'P30', 'P31', 'P31', 'P33', 'P33', 'P34', 'P35', 'P35', 'P36', 'P38', 'P38']
ground_truth = [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# List of file base names and sizes
data_name = [
    "data0_pP08_0", "data1_pP08_0", "data2_pP10_1", "data3_pP10_0", "data4_pP12_1",
    "data5_pP12_1", "data6_pP13_0", "data7_pP13_0", "data8_pP14_0", "data9_pP15_0",
    "data10_pP15_1", "data11_pP16_0", "data12_pP16_0", "data13_pP17_1", "data14_pP17_0",
    "data15_pP18_1", "data16_pP18_1", "data17_pP19_1", "data18_pP19_1", "data19_pP20_0",
    "data20_pP20_0", "data21_pP21_0", "data22_pP21_1", "data23_pP23_0", "data24_pP23_0",
    "data25_pP24_1", "data26_pP24_1", "data27_pP25_0", "data28_pP27_0", "data29_pP28_0",
    "data30_pP29_0", "data31_pP29_0", "data32_pP30_1", "data33_pP30_1", "data34_pP31_0",
    "data35_pP31_0", "data36_pP33_0", "data37_pP33_0", "data38_pP34_0", "data39_pP35_0",
    "data40_pP35_0", "data41_pP36_0", "data42_pP38_0", "data43_pP38_0"
]
data_size = [
    9875, 5900, 1638, 2542, 2036, 1085, 7532, 280, 10007, 3509, 2452, 11881, 10347,
    1119, 0, 3201, 3412, 465, 497, 2185, 1799, 2481, 2243, 3540, 3440, 5518, 2155,
    560, 4, 2890, 13038, 16197, 9036, 10052, 4108, 1423, 7039, 6073, 4153, 7687,
    7214, 9576, 3963, 1940
]

#@title ----  Arrange Features in Dictionary Form  ----

# Function for Dictionary Representation of Data-features

def convert_to_named_features(old_data):
    """Convert tuple-based feature structure to named dictionary structure"""
    feature_tuple = old_data['feature']

    new_data = {
        'feature': {
            'contours': feature_tuple[0],  # contours_list
            'au': feature_tuple[1],       # au_list
            'landmarks': feature_tuple[2], # landmarks_list
            'headEulerAngle': feature_tuple[3], # headEulerAngle_list
            'classification': feature_tuple[4]   # classification_list
        },
        'ground_truth': old_data['ground_truth']
    }
    return new_data



# Loading and Converting
import pickle
import os

def load_and_convert(filepath):
    """Load a pickle file and convert its structure"""
    with open(filepath, 'rb') as f:
        old_data = pickle.load(f)

    # Convert the data structure
    new_data = convert_to_named_features(old_data)

    # Optional: Save the converted version back
    # new_filename = f"converted_{os.path.basename(filepath)}"
    # with open(new_filename, 'wb') as f:
    #     pickle.dump(new_data, f)

    return new_data



# Merger
def merge_datasets(data1, data2):
    merged = {'feature': {}, 'ground_truth': data1['ground_truth']}
    for key in data1['feature']:
        merged['feature'][key] = data1['feature'][key] + data2['feature'][key]
    return merged



# @title load all extracted data
import pickle

tst = ["data2_pP10_1.pkl", "data3_pP10_0.pkl"]
        
mrg1 = ["data6_pP13_0.pkl", "data7_pP13_0.pkl"]
mrg2 = ["data17_pP19_1.pkl", "data18_pP19_1.pkl"]
                
avoid = ["data14_pP17_0.pkl", "data28_pP27_0.pkl"]

test_data = []
train_data = []

all_avoid = tst + mrg1 + mrg2 + avoid

# load test files
for fl in tst:
    idx = fl.split('.')[0].split('_')[0][4:]
    pid = fl.split('.')[0].split('_')[1][1:]
    gt = fl.split('.')[0].split('_')[2]
    new_name = f"dt_{idx}_{pid}_{gt}"

    # Generate the filepath
    file_path = extracted_data_path + fl

    # Load the file
    loaded_data = load_and_convert(file_path)

    tpl = (new_name, loaded_data)
    test_data.append(tpl)



# load and rename all other files to train data
for i in range(len(patient_id)):
    file_name = f"data{i}_p{patient_id[i]}_{ground_truth[i]}.pkl"
    new_name = f"dt_{i}_{patient_id[i]}_{ground_truth[i]}"
    if file_name in all_avoid:
        continue
    else:
        # Generate the filepath
        file_path = extracted_data_path + file_name
        # Load the file
        loaded_data = load_and_convert(file_path)
        tpl = (new_name, loaded_data)
        train_data.append(tpl)



# marge files and append to train_data
def marge_append(dt1, dt2):
    data1 = load_and_convert(extracted_data_path + dt1)
    data2 = load_and_convert(extracted_data_path + dt2)
    
    loaded_data = merge_datasets(data1, data2)
    
    idx = dt1.split('.')[0].split('_')[0][4:]
    pid = dt1.split('.')[0].split('_')[1][1:]
    gt = dt1.split('.')[0].split('_')[2]
    new_name = f"dt_{idx}_{pid}_{gt}"

    tpl = (new_name, loaded_data)
    train_data.append(tpl)

marge_append(mrg1[0], mrg1[1])
marge_append(mrg2[0], mrg2[1])



# @title save as pickle
with open('/content/drive/MyDrive/Colab Notebooks/train_data_FP.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('/content/drive/MyDrive/Colab Notebooks/test_data_FP.pkl', 'wb') as f:
    pickle.dump(test_data, f)

# re-open
with open('/content/drive/MyDrive/Colab Notebooks/train_data_FP.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('/content/drive/MyDrive/Colab Notebooks/test_data_FP.pkl', 'rb') as f:
    test_data = pickle.load(f)

# --------------------------------------------------------------------------------------------------------------------------------

# compare Train-Test with old extracted data, index check

# load a single extracted data
import pickle

fp = extracted_data_path + f"data{0}_p{patient_id[0]}_{ground_truth[0]}.pkl"
# Load the file
with open(fp, 'rb') as f:
    data_P08_1 = pickle.load(f)

data_P08_1_dict = convert_to_named_features(data_P08_1)

# old extracted data
print(data_P08_1_dict['feature']['contours'][0][3])
print(len(data_P08_1_dict['feature']['contours']))
print(data_P08_1['feature'][0][0][3])
print(len(data_P08_1['feature'][0]))

# same dasta inside "train_data"
# correction, we given the variable name wrong above
# data_P08_1 = pickle.load(f) is wrong, it should be: data_P08_0, last digit for ground truth
print(train_data[0][0])
print(type(train_data[0][0]))
print(type(train_data[0][1]))
print(len(train_data[0][1]['feature']['contours']))
print(train_data[0][1]['feature']['contours'][0][3])


# for test data "data2_pP10_1.pkl"
# load a single extracted data
import pickle

fp = extracted_data_path + "data2_pP10_1.pkl"
# Load the file
with open(fp, 'rb') as f:
    data_P10_1 = pickle.load(f)

print(data_P10_1['feature'][0][0][3])
print(len(data_P10_1['feature'][0]))
print(len(test_data[0][1]['feature']['contours']))
print(test_data[0][1]['feature']['contours'][0][3])    




# =====================================================================================================
# Step: 1, 2, 3 
# =====================================================================================================


# ----------------------------  sample -> split: train, projection  ------------------------------------

# 1. Draw sample: 
#                 fix how many times the sample can be drawn -> use the above kind of decide_sample_allocation()
#                 draw sample

# 2. split sample (use single sample to train the SOM if dataset is too small, other bigger datat for test)
#    [hard code it if needed]

# 3. update sample metadata:
#         ->  train set: {id_n : [list of sample index-set from id_n]}, i.e. [set1, set2, etc]
#         ->  projection set: {id_n : [list of sample index-set from id_n]}, i.e. [set1, set2, etc]
#         ->  "iteration number": "iteration number" to calculate how mnay times sample drawn (now + previously): {id_n : [times_for_train, times_for_projection]}
#         ->  corresponding trained SOM file name: SOM file name


# Step: 1, 2
def decide_sample_size(group_size):
    """
    Decide sample size based on group size.
    Returns (sample_size, train_split, projection_split).
    """
    if group_size < 500:
        return (0, 0, 0)
    elif group_size < 3000:
        return (500, 1, 0)
    elif group_size < 5000:
        return (1000, 1, 1)
    elif group_size < 7000:
        return (1500, 1, 2)
    elif group_size < 8000:
        return (2000, 2, 2)
    elif group_size < 11000:
        return (2500, 2, 3)
    elif group_size <= 20000:
        return (3000, 3, 3)
    else:
        n = round((0.15 * group_size) / 500)
        sample_size = n * 500
        if n % 2 == 0:
            return (sample_size, n // 2, n // 2)
        else:
            return (sample_size, (n - 1) // 2, (n + 1) // 2)



# train_data = [(name, dict)], m size list
# dict = 
        # 'feature': {
        #     'contours': contours_list
        #     'au': au_list
        #     'landmarks': landmarks_list
        #     'headEulerAngle': headEulerAngle_list
        #     'classification': classification_list
        # },
        # 'ground_truth': 0 or 1


import numpy as np

# make indices for sample
def sample_uniformly(size, num_samples=500):
    """
    Randomly selects 'num_samples' indices from the dataset in a uniform manner.

    Args:
        size (int): Total number of data points in the dataset.
        num_samples (int): Number of samples to select (default: 500).

    Returns:
        np.ndarray: An array of sampled indices.
    """
    # Ensure sample size doesn't exceed available data
    num_samples = min(size, num_samples)
    # Select 'num_samples' unique indices uniformly without a fixed seed within 'size'
    sampled_indices = np.random.choice(size, num_samples, replace=False)
    return sampled_indices


# get data from samples indexes
def get_sample_data(data_dict, sampled_indices):
    """  
    Extracts a subset of data using sampled indices.

    Args:
        data_dict (dict): Dictionary containing 'feature' and its sub-keys.
        sampled_indices (list): List of sampled indices.

    Returns: 
        dict: A new dictionary containing only the sampled data.
    """
    sampled_data = {
        'sample': sampled_indices.tolist(),
        'feature': {
            'contours': [data_dict['feature']['contours'][i] for i in sampled_indices],
            'au': [data_dict['feature']['au'][i] for i in sampled_indices],
            'landmarks': [data_dict['feature']['landmarks'][i] for i in sampled_indices],
            'headEulerAngle': [data_dict['feature']['headEulerAngle'][i] for i in sampled_indices],
            'classification': [data_dict['feature']['classification'][i] for i in sampled_indices],
        },
        'ground_truth': data_dict['ground_truth']
    }
    return sampled_data



# --- {name: idx} dictionary  ---
# Build once, after train_data is loaded
name_to_index = {name: idx for idx, (name, _) in enumerate(train_data)}

def get_index_by_name(name):
    return name_to_index.get(name, -1)  # or raise an error if you prefer



# --- Sampling Loop ---
# store these idx in metadata
som_train_idx = []
som_prj_idx = []

# lists of tuples like [(name, sampled_indices)]
for name, data_dict in train_data:
    data_size = len(data_dict['feature']['contours'])
    _, num_train, num_prj = decide_sample_size(data_size)

    for _ in range(num_train):
        indices = sample_uniformly(data_size, 500)
        som_train_idx.append((name, indices.tolist()))

    for _ in range(num_prj):
        indices = sample_uniformly(data_size, 500)
        som_prj_idx.append((name, indices.tolist()))

# Step: 3
# ----  update "sample metadata" here  ----
# Store the "som metadata" after SOM & Vector


# SAMPLE: getting samples using above som_train_idx and som_prj_idx
som_train_samples = []
som_prj_samples = []

for name, indices in som_train_idx:
    data_idx = get_index_by_name(name)
    nm, dt_dict = train_data[data_idx]
    som_train_samples.append((name, indices, get_sample_data(dt_dict, indices)))
    
for name, indices in som_prj_idx:
    data_idx = get_index_by_name(name)
    nm, dt_dict = train_data[data_idx]    
    som_prj_samples.append((name, indices, get_sample_data(dt_dict, indices)))



# =====================================================================================================
# Step: 4, 5, 6, 7, 8
# =====================================================================================================
# ----  sample contour and transform to image  ----
# do the same format using extracted som_train_samples, som_prj_samples: 
    # som_train_contour = [(name, indices, contours_list_list)]
    # som_train_img = [(name, indices, img_lis)]
    # som_prj_contour = [(name, indices, contours_list_list)]
    # som_prj_img = [(name, indices, img_lis)]

    # use  som_train_samples, som_prj_samples to extract the contours
    # use convert_to_img(contours) to image conversion
from skimage.color import rgb2gray

som_train_contour = []
som_train_img = []

som_prj_contour = []
som_prj_img = []


for name, indices, sample_dict in som_train_samples:
    contours_list = sample_dict['feature']['contours']
    
    # Store contours
    som_train_contour.append((name, indices, contours_list))
    
    # Convert contours to images
    img_list = [convert_to_img(contour) for contour in contours_list]
    # som_train_img.append((name, indices, img_list))

    # Convert RGB images to grayscale
    gray_list = [rgb2gray(np.array(img)) for img in img_list]  # convert_to_img likely returns PIL or RGB arrays
    som_train_img.append((name, indices, gray_list))



for name, indices, sample_dict in som_prj_samples:
    contours_list = sample_dict['feature']['contours']
    
    # Store contours
    som_prj_contour.append((name, indices, contours_list))
    
    # Convert contours to images
    img_list = [convert_to_img(contour) for contour in contours_list]
    # som_prj_img.append((name, indices, img_list))

    # Convert RGB images to grayscale
    gray_list = [rgb2gray(np.array(img)) for img in img_list]  # convert_to_img likely returns PIL or RGB arrays
    som_prj_img.append((name, indices, gray_list))



# ------------  try in Colab: untested [13-Jul-2025]  ------------

#@title -------- Train the Uni-Ref-SOM --------

# --------  Install minisom if not already installed  --------
!pip install minisom

# Imports
import numpy as np
from minisom import MiniSom


# ------------------------  PART 1: train a reference SOM   ------------------------

# GOAL: Extract and combine all grayscale images from som_train_img
#       Combine all images (49 items × 500 = 24,500 images) and train a SOM to define universal clusters.
# Format: som_train_img = [(name, indices, [img1, img2, ..., img500]), ...]

# Step 1: Flatten the grayscale image data (64x64) to 1D vectors (4096)
flat_all = []

for name, indices, gray_images in som_train_img:
    for img in gray_images:
        flat_all.append(img.reshape(-1))  # shape becomes (4096,)

# Convert to NumPy array
flat_all = np.array(flat_all)  # shape: (total_samples, 4096)

# Optional: Check shapes
print("Total samples:", len(flat_all))
print("Each image vector shape:", flat_all.shape)



# --------  Train universal SOM (e.g., 10x10 grid = 100 clusters)  --------
# "size" & "iteration" fixing for SOM neuron (TUNING 1):

# size: we set 11x11, considering 500 sample size for each individual
    #   i.e. 121 clusters for 500 individual, 1 node for 4 data-point
    #   because our goal was making a SOM for each sample of 500 data.
    #   5*sqrt(500) = 5*22.4 = 112 : 11*11 or 10*10

# iterations: we set initially 3000 (took 10s train), 7000(23s), 10000(36s), 60500 (240s), 70000(300s), 85000(347), 100000(400s)

som_universal = MiniSom(11, 11, input_len=flat_all.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
som_universal.train_random(flat_all, num_iteration=123000)
# we used "random_seed=42" so that random weight initialization is fixed

#@title Assign Clusters

# ------------------------  PART 2: Project samples onto the Universal SOM   ------------------------
# PART 2 GOAL:
    # 1.    Project Individual Images onto the Universal SOM
    # 2.    Create a probability vector (stochastic vector) for each individual based on the number of images assigned to each cluster.


# Project Individual Images onto the Universal SOM
    # For each individual, map their images to the universal SOM's nodes to ensure consistent group labels.

def assign_clusters(images, som):
    # Flatten images (shape: 500, 64, 64) -> (500, 4096)
    flat_img = images.reshape(500, -1)

    # Assign clusters using the universal SOM (map each image to its best-matching SOM node)
    cluster_labels = np.array([som.winner(x) for x in flat_img])
    # Unique cluster IDs: Convert node coordinates to unique integers (e.g., (row*11 + col))
        # for 11x11 som the formula is: row * 11 + col
    cluster_ids = np.array([row * 11 + col for (row, col) in cluster_labels])

    # If your SOM size changes dynamically, use:
    # grid_size_x = som.x  # Number of columns
    # cluster_ids = np.array([row * grid_size_x + col for (row, col) in cluster_labels])

    # or
    # cluster_ids = np.array([row * som.x + col for (row, col) in cluster_labels])

    return cluster_ids


# NOTE: for a single input x, som.winner(x) returns only one pair of coordinates (row, col)
    # So cluster_labels in our code is a "list of (x, y) coordinate pairs", where:
        # x represents the "row index" of the SOM grid.
        # y represents the "column index" of the SOM grid.


# @title --------  cluster assign  --------
# Task: 
    # get the clusters for each of 49 items also 
    # group the index of image according to clusters -> find the group of original indices
    # use "sampled indices"

# we'll test index 2 and 3 (i.e. items 3 & 4)
# dt_1_P08_0    [5726, 4170, 3715, 4093, 371, 4089, 4741, 3379, 3143, 3534]
# dt_4_P12_1    [1624, 1025, 474, 918, 75, 2018, 1374, 1593, 1686, 147]

# Assign clusters for item 3
print(f"{som_train_img[2][0]}\t{som_train_img[2][1][:10]}")
item3_images = som_train_img[2][2]      # Shape (500, 64, 64)
item3_clusters_ids = assign_clusters(item3_images, som_universal)

# Assign clusters for item 4
print(f"{som_train_img[3][0]}\t{som_train_img[3][1][:10]}")
item4_images = som_train_img[3][2]      # Shape (500, 64, 64)
item4_clusters_ids = assign_clusters(item4_images, som_universal)


# --------  rev[13-Jul-2025]  --------

# Analyze Cluster Consistency
    # Check if the clusters for individual 4 align with those for individual 5
    # Visualize the clusters to ensure they group similar expressions across individuals.

import matplotlib.pyplot as plt

# Visualize cluster assignments for individual 0
plt.hist(item3_clusters_ids, bins=100, alpha=0.5, label='Person 4')
plt.hist(item4_clusters_ids, bins=100, alpha=0.5, label='Person 5')
plt.xlabel('Cluster ID')
plt.ylabel('Frequency')
plt.legend()
plt.show()




# ==================    vizualization    ===============================

import numpy as np
import matplotlib.pyplot as plt

# Unique colors for each person (red tint for Person 4, blue tint for Person 5)
# person 4's index is 3,  and person 5's index is 4
color_maps = {3: 'Reds', 4: 'Blues'}  # Individual 4 → Red, Individual 5 → Blue

# Store data separately for each person
individuals = {3: item3_images, 4: item4_images}
individual_cluster_ids = {3: item3_clusters_ids, 4: item4_clusters_ids}

# Define the total number of possible clusters (e.g., 121 for an 11x11 SOM)
n_possible_clusters = 121
num_persons = len(individuals)

# Step 1: Compute Prototypes & Cluster Data for Each Person for ALL possible clusters
prototypes = {}         # keys: (person_id, cluster_id)
cluster_counts = {}     # keys: (person_id, cluster_id)
cluster_images_dict = {}  # keys: (person_id, cluster_id)

for person_id in individuals:
    images = individuals[person_id]
    cluster_ids = individual_cluster_ids[person_id]
    # Determine image shape from the first image (assumes at least one image exists)
    image_shape = images[0].shape

    for cluster_id in range(n_possible_clusters):
        key = (person_id, cluster_id)
        # Check if this cluster exists in the current person's data
        if cluster_id in np.unique(cluster_ids):
            cluster_imgs = images[cluster_ids == cluster_id]
            prototype = np.mean(cluster_imgs, axis=0)
            prototypes[key] = prototype
            cluster_counts[key] = len(cluster_imgs)
            cluster_images_dict[key] = cluster_imgs
        else:
            # For empty clusters, assign an empty (all-zero) image
            prototypes[key] = np.zeros(image_shape)
            cluster_counts[key] = 0
            cluster_images_dict[key] = np.empty((0, *image_shape))

# Determine maximum number of images in any cluster across all individuals
max_n = max(cluster_counts.values()) if cluster_counts else 0

# Total columns: each cluster has one column per person
total_columns = n_possible_clusters * num_persons
total_rows = max_n + 1  # First row for prototypes

# Create a grid for visualization (adjust figsize as needed)
# To increase the size of the images in the visualization, modify the "figsize" parameter in the plt.subplots() function.
# currently 0.5 -> 2
fig, axes = plt.subplots(total_rows, total_columns, figsize=(total_columns * 2, total_rows * 2))

# Step 2: Visualize: For each possible cluster, show each person's prototype and images side by side
for cluster_id in range(n_possible_clusters):
    for person_id in range(num_persons):
        col_idx = cluster_id * num_persons + person_id
        key = (person_id, cluster_id)
        cmap = color_maps[person_id]

        # Plot prototype in the first row
        axes[0, col_idx].imshow(prototypes[key], cmap=cmap)
        count = cluster_counts[key]
        axes[0, col_idx].set_title(f'P{person_id} C{cluster_id}\n(n={count})', fontsize=6)
        axes[0, col_idx].axis('off')

        # Plot each image below the prototype (or an empty image if none available)
        cluster_imgs = cluster_images_dict[key]
        for row_idx in range(max_n):
            ax = axes[row_idx + 1, col_idx]
            if row_idx < count:
                ax.imshow(cluster_imgs[row_idx], cmap=cmap)
            else:
                ax.imshow(np.zeros_like(prototypes[key]), cmap=cmap)
            ax.axis('off')

plt.tight_layout()
plt.show()




# ===============    probability (stochastic) vector    ===============

# probability_vectors contains the probability vector for each individual.
# Note: cluster_ids and person_clusters represent the same data: the cluster assignment for each image of a person.


import numpy as np

def compute_probability_vector(cluster_ids, n_clusters):
    """
    Compute both the normalized probability vector and the raw counts vector for an individual based on 
    the number of images assigned to each cluster (from 0 to n_clusters-1).
    
    Parameters:
    - cluster_ids: a NumPy array of cluster IDs assigned to each image.
    - n_clusters: total number of possible clusters (e.g., 121 for an 11x11 SOM).
    
    Returns:
    - probability_vector: normalized counts (stochastic vector) of length n_clusters.
    - counts_vector: raw counts for each cluster, of length n_clusters.
    """
    sample_size = len(cluster_ids)  # Dynamic sample size
    counts_vector = np.zeros(n_clusters)  # Initialize raw counts with zeros for all clusters
    unique, counts = np.unique(cluster_ids, return_counts=True)  # Get unique cluster IDs and counts (Count occurrences per cluster)
    counts_vector[unique] = counts  # Assign counts to corresponding clusters (indices)
    probability_vector = counts_vector / sample_size  # Normalize by the sample size
    return probability_vector, counts_vector

# Define number of clusters for an 11x11 SOM
n_clusters = 11 * 11  # 121 clusters

# Dictionaries to store "probability vectors" and "raw counts" for each individual (excluding the first 3)
probability_vectors = {}
raw_counts_vectors = {}

# Loop over individuals from index 4 onward (excluding first 3)
for person_id in range(3, len(data['X1'])):
    person_images = data['X1'][person_id]  # Get individual's images
    person_clusters = assign_clusters(person_images, som_universal)  # Get cluster assignments for that individual
    
    # Compute probability and raw counts vectors
    prob_vec, counts_vec = compute_probability_vector(person_clusters, n_clusters)
    
    # Store in separate dictionaries
    probability_vectors[person_id] = {'label': data['y'][person_id], 'prob_vector': prob_vec}
    raw_counts_vectors[person_id] = counts_vec



# ===============    save the trained SOM & vectors    ===============
import pickle

# Save the trained SOM to a file
with open('trained_som.pkl', 'wb') as f:
    pickle.dump(som_universal, f)

print("Trained SOM has been saved to 'trained_som.pkl'.")


# Save the probability_vectors dictionary to a file
with open('probability_vectors.pkl', 'wb') as f:
    pickle.dump(probability_vectors, f)

print("Probability vectors have been saved to 'probability_vectors.pkl'.")


# opening
with open('trained_som.pkl', 'rb') as f:
    som_universal = pickle.load(f)

with open('probability_vectors.pkl', 'rb') as f:
    probability_vectors = pickle.load(f)



#@title ===============    Load the SOM model and probability vector    ===============
import pickle

# Load the trained SOM model and probability vectors
with open('trained_som.pkl', 'rb') as f:
    som_universal = pickle.load(f)

with open('probability_vectors.pkl', 'rb') as f:
    probability_vectors = pickle.load(f)


# Preview loaded SOM model (Checking dimensions and attributes)
print(f"SOM Model Loaded: {som_universal}")
print(f"SOM Grid Size: {som_universal._weights.shape[:2]}")  # Assuming _weights stores the SOM structure


# Preview probability vectors
print("\n=== Preview of Probability Vectors ===")
for person_id, data in list(probability_vectors.items()):
    print(f"Person {person_id}: Label = {data['label']}, Prob Vector (first 10) = {data['prob_vector'][:10]}")
    print(f"Min value: {data['prob_vector'].min()}, Max value: {data['prob_vector'].max()}")
    print(f"sum: {data['prob_vector'].sum()}")
    


# adjusted like below:
"""  
                    Now here comes the fun part. Remember the structure of som_train_img? 
                        We can now drop the images and store only the cluster_ids, 
                        since we have the real image IDs.

                    now we'll do this for both som_train_img and som_prj_img parts

                    then we calculate the probability vectors (total 49+52, save the different 2d array)

                    save the som too.


                    So the new data format to save in file:
                    som_trained_data = {
                        'train_clustr': array of (name, real_500_idx, cluster_ids_500),
                        'prj_clustr': array of (name, real_500_idx, cluster_ids_500),
                        'pob_vec_train': probability_vectors,
                        'pob_vec_prj': probability_vectors,
                        'som_model': som_universal
                    }

"""


# Updated Code (post-SOM training, keep simple):

import pickle
import numpy as np

# -------- Convert som_train_img and som_prj_img to only cluster info --------

def convert_img_list_to_cluster_data(img_list, som_model):
    result = []
    for name, real_idx, images in img_list:
        cluster_ids = assign_clusters(images, som_model)
        result.append((name, real_idx, cluster_ids))
    return result

# Apply to both training and projection sets
train_cluster_data = convert_img_list_to_cluster_data(som_train_img, som_universal)
prj_cluster_data = convert_img_list_to_cluster_data(som_prj_img, som_universal)


# -------- Compute probability vectors for each individual --------
# use a list of tuples:     prob_vectors = [(name, prob_vec)]

def compute_all_probability_vectors(cluster_data, total_clusters=121):
    prob_vectors = []  # List of (name, prob_vector) tuples
    for name, real_indices, cluster_ids in cluster_data:
        prob_vec, _ = compute_probability_vector(cluster_ids, total_clusters)
        prob_vectors.append((name, prob_vec))
    return prob_vectors


# Compute separately for train and projection sets
prob_vec_train = compute_all_probability_vectors(train_cluster_data, n_clusters)
prob_vec_prj = compute_all_probability_vectors(prj_cluster_data, n_clusters)


# -------- Save all in one dictionary --------

som_trained_data = {
    'train_clustr': train_cluster_data,          # list of (name, real_500_idx, cluster_ids)
    'prj_clustr': prj_cluster_data,
    'pob_vec_train': prob_vec_train,
    'pob_vec_prj': prob_vec_prj,
    'som_model': som_universal
}

# Save to file
with open('som_trained_data.pkl', 'wb') as f:
    pickle.dump(som_trained_data, f)

print("✅ SOM, clusters, and probability vectors saved to 'som_trained_data.pkl'")













# =====================================================================================================

# =====================================================================================================


# ---------------------------------------------------------------------------------

# ----------------  ENTROPY  ----------------

# how to know SOM is working or not:
# Calculate Entropy Before & After SOM
import numpy as np
from scipy.stats import entropy
from minisom import MiniSom  # Self-Organizing Map library

def calculate_entropy(data):
    """
    Computes Shannon entropy for the given dataset.
    
    Args:
        data (numpy array): Input feature matrix (samples × features).
    
    Returns:
        float: Shannon entropy value.
    """
    # Flatten data to get a distribution
    values, counts = np.unique(data, return_counts=True)
    prob_dist = counts / counts.sum()  # Convert counts to probabilities

    return entropy(prob_dist, base=2)  # Shannon entropy in bits

# Load your dataset
# Assume 'features' contains raw data: shape (9875, num_features)
features = np.array(data_P08_1_dict['feature']['contours'])  # Example with contours

# Compute entropy before applying SOM
entropy_before = calculate_entropy(features)
print(f"Entropy Before SOM: {entropy_before:.4f}")

# Apply SOM
som_size = (10, 10)  # 10x10 SOM grid
som = MiniSom(som_size[0], som_size[1], features.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(features)
som.train_random(features, 1000)  # Train SOM

# Get cluster assignments
mapped_clusters = np.array([som.winner(x) for x in features])
cluster_labels, cluster_counts = np.unique(mapped_clusters, return_counts=True)
cluster_probabilities = cluster_counts / cluster_counts.sum()

# Compute entropy after applying SOM
entropy_after = entropy(cluster_probabilities, base=2)
print(f"Entropy After SOM: {entropy_after:.4f}")

# Interpretation
if entropy_after < entropy_before:
    print("SOM reduced entropy, meaning data is more structured into clusters.")
else:
    print("Entropy remained the same or increased, possibly due to overlapping clusters.")




