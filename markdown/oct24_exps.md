Seeing if linear maps for attention transfer help.

Teacher: WRN-40-2
Student: WRN-16-2

Comparing (i) no map, (ii) linear map of 3x3 convs, (iii) linear map of 1x1 convs (in both cases retaining channel size)

| Map   | Exp1  | Exp2  | Exp3  | Exp4  | Exp5  |
|-------|-------|-------|-------|-------|-------|
| none  | 94.43 | 94.53 | 94.42 | 94.47 |       |
| 3X3   | 94.36 | 94.58 | 94.44 | 94.47 | 94.48 |
| 1X1   | 94.16 | 94.12 | 93.93 | 94.22 | 94.26 |

So makes no difference.
