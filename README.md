# K-Means Image Segmentation with RGB, Position, and Texture Features in MATLAB

This project applies K-means clustering to perform image segmentation using RGB values, spatial position, and texture-based features. The impact of weighting and feature combinations is explored and visualized.

[View PDF Result](./hw11.pdf)

---

## A1. Implementing K-Means Clustering

- Segmentation was performed using K = 5 and K = 10 clusters.
- The stopping criterion was based on convergence of assignments or a fallback of 200 iterations (never reached).
- Images segmented:
  - `sunset.tiff`
  - `tiger-1.tiff`
  - `tiger-2.tiff`
- Segmentations are visually distinct, with higher K values capturing finer details.

## A2. 5-Dimensional Feature Vector (R, G, B, λ*X, λ*Y)

- Segmentation was repeated using a 5D feature vector combining color and pixel location.
- Weighting parameter λ was tested with values 1 and 10 for K = 10.
- Higher λ increases the influence of pixel position:
  - λ = 1: color dominates
  - λ = 10: spatial proximity dominates
- Observed that increasing λ causes more spatially localized segmentations.

## A3. Using Texture Features

- Texture features computed using root mean squared values over a 10×10 window.
- Feature combinations tested:
  1. Texture only
  2. Texture + RGB
  3. Texture + RGB + position (λ = 1)
- All feature vectors scaled to [0, 255] for consistency.

### Observations:

- **Texture only:** Captures some spatial structure but lacks sharp color segmentation.
- **Texture + RGB:** Adds subtle grouping but can degrade clean RGB-based segmentation.
- **Texture + RGB + position:** Adds spatial smoothing, but in these images, it reduces quality.

### Additional Notes:

- Visual artifacts appeared near borders, possibly due to edge effects from convolution used to compute local texture.
- Artifacts changed with window size, suggesting that convolution padding or boundary handling in MATLAB might be responsible.

---

## Conclusion

This assignment demonstrates that while K-means clustering is sensitive to feature representation, RGB values alone often produce the cleanest segmentations in color-dominated images. The inclusion of position and texture features introduces spatial and pattern sensitivity, which may be more useful in scenarios with low color contrast.
