# Benchmark results

```
{
 "machine": "macOS-27.0.1-arm64-arm-64bit",
 "processor": "arm",
 "torch": "2.11.0",
 "threads": 10,
 "device": "cpu",
 "sigmaRKHS": [
  0.2,
  0.1,
  0.05
 ],
 "sigmaVar": [
  0.2,
  0.1,
  0.05,
  0.02
 ],
 "coefficients": {
  "cA": 1.0,
  "cT": 1.0,
  "cScale": 1.0,
  "cNR": 10.0,
  "cL": 1.0
 },
 "gamma": 0.1,
 "registration_points": 2000,
 "registration_steps": 10
}
```

## Part A: deformation only (median seconds)

| N | mode | flow | H | forward | fwd+bwd | rel. diff. to closed form |
|---|---|---|---|---|---|---|
| 1000 | all-mixed (xmodmap) | Ralston nt=10 | 0.0010 | 0.1576 | 0.9990 |  |
| 1000 | translation | closed form | 0.0000 | 0.0001 | 0.0002 |  |
| 1000 | translation | Ralston nt=10 | 0.0000 | 0.0013 | 0.0018 | 7.2e-07 |
| 1000 | rotation | closed form | 0.0000 | 0.0002 | 0.0004 |  |
| 1000 | rotation | Ralston nt=10 | 0.0000 | 0.0047 | 0.0069 | 3.1e-05 |
| 1000 | scaling | closed form | 0.0000 | 0.0002 | 0.0005 |  |
| 1000 | scaling | Ralston nt=10 | 0.0000 | 0.0037 | 0.0073 | 2.1e-05 |
| 1000 | rigid | closed form | 0.0000 | 0.0002 | 0.0004 |  |
| 1000 | rigid | Ralston nt=10 | 0.0000 | 0.0057 | 0.0081 | 4.3e-07 |
| 1000 | similarity | closed form | 0.0001 | 0.0002 | 0.0006 |  |
| 1000 | similarity | Ralston nt=10 | 0.0001 | 0.0065 | 0.0148 | 5.6e-07 |
| 1000 | affine | closed form | 0.0000 | 0.0003 | 0.0007 |  |
| 1000 | affine | Ralston nt=10 | 0.0000 | 0.0047 | 0.0075 | 1.2e-05 |
| 1000 | non-rigid | Ralston nt=10 | 0.0010 | 0.1574 | 0.9010 |  |
| 1000 | non-rigid+rigid | Ralston nt=10 | 0.0010 | 0.1559 | 0.9191 |  |
| 1000 | all-mixed | Ralston nt=10 | 0.0011 | 0.1621 | 0.9386 |  |
| 10000 | all-mixed (xmodmap) | Ralston nt=10 | 0.0929 | 15.0984 | 89.4361 |  |
| 10000 | translation | closed form | 0.0000 | 0.0004 | 0.0006 |  |
| 10000 | translation | Ralston nt=10 | 0.0000 | 0.0024 | 0.0032 | 2.0e-07 |
| 10000 | rotation | closed form | 0.0001 | 0.0005 | 0.0009 |  |
| 10000 | rotation | Ralston nt=10 | 0.0001 | 0.0094 | 0.0153 | 2.4e-05 |
| 10000 | scaling | closed form | 0.0001 | 0.0006 | 0.0010 |  |
| 10000 | scaling | Ralston nt=10 | 0.0001 | 0.0088 | 0.0179 | 1.3e-04 |
| 10000 | rigid | closed form | 0.0001 | 0.0005 | 0.0009 |  |
| 10000 | rigid | Ralston nt=10 | 0.0001 | 0.0101 | 0.0158 | 1.2e-06 |
| 10000 | similarity | closed form | 0.0002 | 0.0005 | 0.0011 |  |
| 10000 | similarity | Ralston nt=10 | 0.0002 | 0.0129 | 0.0288 | 1.2e-06 |
| 10000 | affine | closed form | 0.0001 | 0.0005 | 0.0011 |  |
| 10000 | affine | Ralston nt=10 | 0.0001 | 0.0073 | 0.0162 | 6.4e-06 |
| 10000 | non-rigid | Ralston nt=10 | 0.0905 | 15.3157 | 92.8902 |  |
| 10000 | non-rigid+rigid | Ralston nt=10 | 0.1078 | 14.9405 | 96.0936 |  |
| 10000 | all-mixed | Ralston nt=10 | 0.0957 | 14.7738 | 94.5165 |  |

## Part B: registration

| mode | closed form | N | s / loss eval | optim. wall (s) | loss evals | H loss | data loss | angle | angle err. | transl. err. | RMS to target |
|---|---|---|---|---|---|---|---|---|---|---|---|
| translation | True | 2000 | 0.043 | 0.4 | 11 | 3.992e-03 | 1.651e+01 | 0.0000 | 6.0e-01 | 3.8e-01 | 1.8e-01 |
| rotation | True | 2000 | 0.035 | 0.5 | 14 | 1.986e-02 | 4.655e+01 | -0.4456 | 1.0e+00 | 4.0e-01 | 4.3e-01 |
| scaling | True | 2000 | 0.036 | 0.3 | 11 | 4.162e-01 | 3.734e+01 | 0.0000 | 6.0e-01 | 6.3e-01 | 4.1e-01 |
| rigid | True | 2000 | 0.043 | 1.8 | 36 | 4.101e-02 | 2.289e-05 | 0.5998 | 1.8e-04 | 1.1e-04 | 5.3e-05 |
| similarity | True | 2000 | 0.048 | 1.7 | 40 | 4.101e-02 | 7.629e-06 | 0.5998 | 1.8e-04 | 1.1e-04 | 5.2e-05 |
| affine | True | 2000 | 0.044 | 2.8 | 57 | 4.101e-02 | 1.526e-05 | 0.5999 | 1.4e-04 | 9.5e-05 | 5.5e-05 |
| non-rigid | False | 2000 | 3.694 | 401.2 | 114 | 1.601e+00 | 1.282e+00 | -0.1308 | 7.3e-01 | 4.2e-01 | 2.7e-01 |
| non-rigid+rigid | False | 2000 | 3.464 | 403.7 | 116 | 3.903e-02 | 5.707e-03 | 0.5997 | 2.9e-04 | 1.1e-04 | 1.4e-03 |
| all-mixed | False | 2000 | 3.597 | 406.8 | 116 | 3.906e-02 | 4.089e-03 | 0.5998 | 1.8e-04 | 1.6e-04 | 1.2e-03 |
