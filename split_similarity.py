# split_similarity.py

import pickle
import os

# Load the large similarity matrix
with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

# Define chunk size (number of rows per chunk, adjust according to memory size)
chunk_size = 500  # Example: Split into chunks of 500 rows each
num_chunks = len(similarity) // chunk_size + 1

# Create a folder to save the chunks
if not os.path.exists('similarity_chunks'):
    os.makedirs('similarity_chunks')

# Split and save each chunk
for i in range(num_chunks):
    chunk = similarity[i * chunk_size:(i + 1) * chunk_size]
    with open(f'similarity_chunks/similarity_chunk_{i}.pkl', 'wb') as f:
        pickle.dump(chunk, f)
    
print(f"Successfully split into {num_chunks} chunks.")
