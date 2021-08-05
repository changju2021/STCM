# STCM: a Spatio-Temporal Calibration Model for Low-cost Air Monitoring Sensors

This is a PyTorch implementation of STCM.

## Requirements

Our code is based on Python3 (>=3.7.7). The major libraries used are list as follows:

+ scikit-learn (>=0.24.2)
+ xgboost (>=1.4.2)
+ torch (>=1.5.1)
+ numpy (>=1.18.5)
+ pandas (>=1.0.5)

## Model

In the directory of "Models", STCM and its variants used in ablation study are included.

In the directory of "Baselines", nine baselines are included. Note that all the point-to-point baselines are implemented in the file "pointToPoint.py".