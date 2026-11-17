"""Mars terrain segmentation on AI4Mars."""

CLASSES = ["soil", "bedrock", "sand", "big rock"]
IGNORE = 255  # unlabelled pixels, the rover itself, and terrain more than 30 m away
