import random
import torch
from torch import nn

class VideoTubeExtractor(nn.Module):
    """
    VideoTubeExtractor randomly extracts a tube from a video.

    Args:
        tube_size (int): The size of the tube to extract.
        num_tubes (int): The number of tubes to extract from the video.

    Returns:
        tubes (torch.Tensor): A tensor of shape (batch_size, num_tubes, num_frames, channels, tube_size, tube_size).

    Example:
        >>> video_frames = torch.randn(4, 16, 3, 224, 224)
        >>> tube_extractor = VideoTubeExtractor(tube_size=32, num_tubes=4)
        >>> tubes = tube_extractor(video_frames)
        >>> tubes.shape
        torch.Size([4, 4, 16, 3, 32, 32])
    
    """
    def __init__(
        self,
        tube_size,
        num_tubes
    ):
        super().__init__()
        self.tube_size = tube_size
        self.num_tubes = num_tubes
    
    def forward(self, video_frames):
        """
        Forward pass of the VideoTubeExtractor.

        Args:
            video_frames (torch.Tensor): A tensor of shape (batch_size, num_frames, channels, height, width).
            
        
        """
        batch_size, num_frames, channels, height, width = video_frames.shape
        tubes = torch.emoty(batch_size, self.num_tubes, num_frames, channels, self.tube_size, device=video_frames.device)

        for b in range(batch_size):
            for t in range(self.num_tubes):
                # Randomlys select the starting point for the tube
                x = random.randint(0, width - self.tube_size)
                y = random.randint(0, height - self.tube_size)

                # Extract the tube across all frames
                tube = video_frames[b, :, :, y:y + self.tube_size, x:x + self.tube_size]
                tubes[b, t] = tube
        
        return tubes
