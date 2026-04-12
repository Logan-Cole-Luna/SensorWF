# Comparative Detector Failure Report

Auto-generated summary of weak anomaly tags and recommended threshold/feature changes.

## Overall Detector Averages

- RobustRollingZScore: recall=0.037, fpr=0.037, auc=0.504
- ZScore: recall=0.109, fpr=0.057, auc=0.585
- IsolationForest: recall=0.225, fpr=0.125, auc=0.571
- Autoencoder: recall=0.485, fpr=0.183, auc=0.683

## Per-Tag Weak Spots

- A5_sun_sensor_blinding: best=ZScore (recall 0.038), worst=IsolationForest (recall 0.014)
  recommendation: use shorter AE sequence (5-7) and reduce threshold percentile by ~0.5
- A1_accel_packet_dropout: best=ZScore (recall 0.072), worst=IsolationForest (recall 0.014)
  recommendation: if recall is low and fpr is low, reduce threshold percentile by ~0.5
- W2_wheel_stiction_stop: best=ZScore (recall 0.163), worst=RobustRollingZScore (recall 0.017)
  recommendation: if recall is low and fpr is low, reduce threshold percentile by ~0.5
- C3_frame_error_avalanche: best=Autoencoder (recall 0.265), worst=ZScore (recall 0.035)
  recommendation: lower IF threshold percentile by 0.5-1.0 and keep contamination >= 0.03
  recommendation: retain comms/timing derivative features; avoid aggressive PCA compression
- C2_rssi_fade: best=Autoencoder (recall 0.283), worst=ZScore (recall 0.038)
  recommendation: lower IF threshold percentile by 0.5-1.0 and keep contamination >= 0.03
  recommendation: retain comms/timing derivative features; avoid aggressive PCA compression
- A2_gyro_clipping: best=ZScore (recall 0.286), worst=RobustRollingZScore (recall 0.019)
  recommendation: if recall is low and fpr is low, reduce threshold percentile by ~0.5
- C1_packet_gap_jitter: best=Autoencoder (recall 0.292), worst=RobustRollingZScore (recall 0.032)
  recommendation: lower IF threshold percentile by 0.5-1.0 and keep contamination >= 0.03
  recommendation: retain comms/timing derivative features; avoid aggressive PCA compression
- P3_rail_latchup: best=Autoencoder (recall 0.321), worst=ZScore (recall 0.036)
  recommendation: if recall is low and fpr is low, reduce threshold percentile by ~0.5
