# Python RAW Image Signal Processing (ISP) Pipeline

A complete, custom-built Python Image Signal Processor (ISP) pipeline that converts RAW sensor data into fully processed, visually pleasing RGB images. 

This project goes beyond standard basic ISP steps by implementing an advanced **N-White Balancing** algorithm for handling multiple illuminants and non-uniform lighting, alongside precise **Color Correction Matrix (CCM)** optimization based on CIELab color space targets.

## ✨ Key Features

* **Raw Parsing & Auto Black-Level:** Reads 16-bit `.raw` files (default resolution: 6264 x 4180), automatically estimates the black level, and subtracts it.
* **Bayer Demosaicing:** Converts BG Bayer patterns to BGR color space.
* **Global White Balance:** Analyzes a standard Gray Card reference to calculate and apply baseline RGB gains.
* **Multi-Illuminant N-White Balancing:** Implements spatial-weighted block-wise Grey-World estimation to dynamically adjust white balance across images with complex, non-uniform lighting.
* **CCM Optimization:** Extracts 24 patches from a standard ColorChecker and uses the SLSQP optimization method to compute a 3x3 Color Correction Matrix, minimizing the color difference ($\Delta E$) in the CIELab color space with a row-sum constraint of 1.
* **Tone Reproduction:** Adaptive luminance mapping using log-average luminance to compress dynamic range while preserving details.
* **Color Enhancement:** CIELab-based saturation stretching for more vibrant colors.
* **Gamma Correction:** Standard 1/2.2 gamma encoding for proper display output.

## 🛠 Dependencies

The script relies on a few standard Python libraries for computer vision and numerical optimization:

```bash
pip install numpy opencv-python scipy
```

## 📄 References
T. Akazawa, Y. Kinoshita, S. Shiota and H. Kiya, "N-White Balancing: White Balancing for Multiple Illuminants Including Non-Uniform Illumination," in IEEE Access, vol. 10, pp. 89051-89062, 2022, doi: 10.1109/ACCESS.2022.3200391.
