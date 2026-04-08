# 2025-HL-Ionosphere-dataset

An ML-ready dataset for ionospheric forecasting, integrating multiple space weather data sources at 15-minute cadence. Developed as part of NASA's Heliolab Frontier Development Lab (FDL) 2025.

## Quick start

Open the example notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FrontierDevelopmentLab/2025-HL-Ionosphere-dataset/blob/main/dataset_example_colab.ipynb)

The notebook walks through downloading sample data, loading and visualizing TEC maps, combining multiple data sources into temporal sequences, and setting up train/validation splits around geomagnetic storm events.

## Data sources

| Dataset | Description |
|---------|-------------|
| **JPLD** | JPL Global Ionospheric Maps (GIMs) — global TEC at 1°x1° resolution |
| **OMNIWeb** | Solar wind parameters and geomagnetic indices (AE, SYM-H, IMF Bz, etc.) |
| **CelesTrak** | Kp and Ap geomagnetic indices |
| **SunMoonGeometry** | Solar/lunar zenith angles and positions |
| **SET** | Solar Energetic Particle data |

Data is hosted publicly on AWS S3 (`s3://nasa-radiant-data/helioai-datasets/ionosphere-data-public/`) — no credentials required.

## References

- [Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models](https://doi.org/10.48550/arXiv.2511.15743)
- [Forecasting the Ionosphere from Sparse GNSS Data with Temporal-Fusion Transformers](https://doi.org/10.48550/arXiv.2509.00631)

## License

Apache 2.0
