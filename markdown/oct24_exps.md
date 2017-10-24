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

--------------------------------------------

We had noticed in the past that doing attention transfer with the "squeeze excite" vectors didn't work as well as with the attention maps. So now seeing if it's to do with the learning rate:

Teacher: WRN-40-2
Student: WRN-16-2

| Beta | Acc   |
|------|-------|
| 1e0  | 93.66 |  
| 1e1  | 93.6  |
| 1e2  | 94.18 |
| 5e2  | 94.18 |
| 1e3  | 93.72 |
| 2.5e3| 93.15 |
| 5e3  | 92.11 |
| 1e4  | 90.37 |
| 1e5  | 83.76 |

Nope. It doesn't work.

--------------------------------------------
