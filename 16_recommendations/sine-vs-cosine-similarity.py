"""
Why Use Cosine Similarity Instead of Sine?

1. Intuitive Interpretation:
   - Cosine similarity gives a value of 1 for vectors pointing in the same direction (0° angle)
   - It gives 0 for perpendicular vectors (90° angle)
   - For text analysis/TF-IDF vectors (which are always non-negative), it ranges from 0 to 1, making it easy to interpret as a similarity percentage

2. Sine Would Give Opposite Results:
   - sin(0°) = 0 (identical vectors would show no similarity)
   - sin(90°) = 1 (perpendicular vectors would show maximum similarity)
   - This is exactly the opposite of what we want to measure for similarity!

3. Vector Magnitude Independence:
   - Cosine similarity focuses on the angle between vectors, not their magnitudes
   - This is crucial for text analysis because we care about the proportion of terms (direction) rather than the absolute frequencies (magnitude)
   - For example, two documents about "space travel" should be similar even if one is twice as long as the other
"""

import numpy as np

# Two similar vectors (small angle)
v1 = np.array([5, 4])
v2 = np.array([6, 5])
angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

print(f"For similar vectors:")
print(f"Cosine similarity: {np.cos(angle):.2f}")  # ≈ 1.00 (very similar)
print(f"Sine similarity: {np.sin(angle):.2f}")    # ≈ 0.00 (would indicate not similar)
