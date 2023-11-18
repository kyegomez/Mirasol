[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Mirasol {WIP}
Implementation of Mirasol with an video and audio encoder and the novel combiner mechanism.

# Video Encoder
To create a pseudocode for the described model that processes audio/video features using sparse 3D tubes and a ViT encoder, let's break down the process into its key components and steps. The model involves extracting spatial-temporal features from video snippets and encoding these features using a Transformer-based architecture.

### Pseudocode for the Audio/Video Feature Extraction and Processing Model

```plaintext
Algorithm: Audio/Video Feature Extraction and Processing

Inputs:
    videos: List of video snippets
    frame_rate: Frame rate for video processing
    tube_extractor: Function to extract sparse 3D tubes from videos
    vit_encoder: ViT (Vision Transformer) encoder for feature encoding

Output:
    video_features: Time-aligned video representations for all video snippets

Procedure:
1. Initialize an empty list video_features to store time-aligned video representations.

2. For each video snippet in videos:
    2.1 Extract frames from the video snippet at the specified frame_rate.
    
    2.2 Use the tube_extractor to extract sparse 3D tubes from the frames.
         - Sparse 3D tubes span all three dimensions (width, height, time) of the video snippet.
         - Tubes start at various locations in the video snippet to capture dynamic events.

    2.3 Alongside sparse 3D tubes, extract standard 2D patches from the video frames.
         - These patches capture spatial information in each frame.

    2.4 Combine the sparse 3D tubes and 2D patches into a unified representation.
         - This step might involve concatenation or another form of fusion.

    2.5 Pass the combined representation through the vit_encoder.
         - The ViT encoder processes the input and extracts high-level features.
         - It captures both spatial and temporal relationships in the data.

    2.6 Store the output of the ViT encoder as the time-aligned features vˆt for the video snippet.

3. Aggregate all time-aligned features vˆt from each video snippet into video_features.
   - video_features = {vˆ1, vˆ2, ..., vˆT}, where T is the total number of video snippets.

4. Return video_features as the final output.
```

### Explanation

- **Step 1**: Initializes a list to store the processed features of each video snippet.
- **Step 2**: Iterates over each video snippet to process it.
- **Step 2.1**: Extracts frames from the video at a given frame rate.
- **Step 2.2**: Applies the `tube_extractor` to get sparse 3D tubes from the video, capturing spatial-temporal information.
- **Step 2.3**: Extracts 2D patches for spatial information from each frame.
- **Step 2.4**: Combines 3D tubes and 2D patches. This step is crucial for integrating spatial and temporal information.
- **Step 2.5**: Encodes the combined representation using a ViT encoder, extracting high-level features.
- **Step 2.6**: Stores the encoded features as time-aligned features for the video snippet.
- **Step 3**: Aggregates the features from all video snippets.
- **Step 4**: Returns the aggregated features as the output.


# License
MIT



