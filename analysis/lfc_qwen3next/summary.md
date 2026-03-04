# Mamba vs Attention Prefix Hit Rate Summary

| Scenario | Request Rate | Runs | Attn Hit Rate (95% CI) | Mamba Hit Rate (95% CI) | Mean TTFT ms (95% CI) |
|----------|-------------|------|------------------------|-------------------------|-----------------------|
| generated-shared-prefix | 1.0 | 2 | 0.4439 [0.0000, 0.8878] | 0.4439 [0.0000, 0.8877] | 2001.8 [441.4, 3562.2] |
| generated-shared-prefix | 4.0 | 2 | 0.4439 [0.0000, 0.8878] | 0.4439 [0.0000, 0.8878] | 3292.9 [458.1, 6127.6] |
