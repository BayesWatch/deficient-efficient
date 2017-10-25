All these have been taught by THE SAME 2.2million param WRN_40_2 with 3x3 convs.

A * indicates the experiment is being run.

| Stu Arc | Convs   | params   | Acc   | KD    | AT    |
|---------|---------|----------|-------|-------|-------|
|WRN_40_1 | 3x3     | 566650   | 93.52 | 93.61 | 94.50 |
|WRN_16_2 | 3x3     | 693498   | 93.47 | 93.97 | 94.34 |
|WRN_16_1 | 3x3     | 175994   | 91.19 | 91.25 | 92.28 |
|WRN_40_2 | sep(d1) | 304074   | 91.49 | 91.99 | 93.43 |
|WRN_40_2 | sep(d2) | 327258   | 92.88 | 93.17 | 94.02 |
|WRN_40_2 | sep(d4) | 373626   | 93.07 | 93.55 | 94.69 |
|WRN_40_2 | sep(d8) | 466362   | 93.93 | 94.39 | 94.94 | 
|WRN_40_2 | sep(d16)| 651834   | 94.28 | 94.28 | 94.88 |
|WRN_40_2 | bottle2 | 437242   | 93.64 | 93.72 | 94.63 |
|WRN_40_2 | bottle4 | 155002   | 92.06 | 92.17 | 93.07 |
|WRN_40_2 | bottle8 | 68314    | 90.12 | 89.94 | 90.69 |
|WRN_40_2 | bottle16| 38578    | 87.71 | 87.78 | 87.40 |
|WRN_40_2 | DConv   | 304074   | 91.54 | 91.91 | 93.26 |
|WRN_40_2 | DConvB2 | 152986   | 92.05 | 92.01 | 93.33 |
|WRN_40_2 | DConvB4 | 85450    | 90.96 | 91.39 | 92.13 |
|WRN_40_2 | DConvB8 | 51682    | 89.20 | 89.61 | 89.96 |
|WRN_40_2 | DConvB16| 34798    | 86.11 | 86.62 | 87.34 |
|WRN_40_2 | G2B2    | 159034   | 92.55 | 92.53 | 93.83 |
|WRN_40_2 | G4B2    | 171130   | 92.94 | 92.85 | 93.97 |
|WRN_40_2 | G8B2    | 195322   | 93.25 | 93.51 | 94.06 |
|WRN_40_2 | G16B2   | 234704   | 93.74 | 93.50 | 93.98 |
|WRN_40_2 | A2      | 1369530  | 94.70 | 94.63 | 95.13 |
|WRN_40_2 | A4      | 825210   | 94.50 | 94.19 | 95.00 |
|WRN_40_2 | A8      | 553050   | 94.08 | 94.28 | 94.95 |
|WRN_40_2 | A16     | 416970   | 93.35 | 93.62 | 94.87 |
|WRN_40_2 | A2B2    | 292090   | 93.88 | 93.75 | 94.43 |
|WRN_40_2 | A4B2    | 219514   | 93.25 | 93.25 | 93.95 |
|WRN_40_2 | A8B2    | 183226   | 93.06 | 93.02 | 93.91 |
|WRN_40_2 | A16B2   | 165082   | 93.23 | 93.03 | 93.81 |

Key:
sep(dX) means separable convolutions with no groups = no_channel/X
bottleX means the bottleneck residual structure where the first conv reduces the dimension by a factor of X
DConv is just sep(d1). Looks like i've done it twice and the results are quite similar.
DConvBX is the bottleneck structure with the convolution having groups = no_channels
GXB2 is a factor 2 bottleneck followed by a conv with groups = no_channels/X
AX is the seperable conv with groups = X
AXB2 is the above with a bottleneck structure


