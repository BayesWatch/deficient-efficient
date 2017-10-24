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

Now to see (over multiple experiments) whether multiple AT points help for stufdent networks with separable convolutions (in assorted groups).

Note the original model scored 95.21%

Run 1

| Student | type    | params     | Acc   | KD    | AT1   | AT2    | AT3   |
|---------|---------|------------|-------|-------|-------|--------|-------|
|WRN_40_2 | sep(d1) |   304074   |       | 91.99 | 93.43 | 93.46  | 93.56 |
|WRN_40_2 | sep(d2) |   327258   |       | 93.17 | 94.02 |        |       |
|WRN_40_2 | sep(d4) |   373626   | 93.07 | 93.55 | 94.69 | 94.55  | 94.59 |
|WRN_40_2 | sep(d8) |   466362   | 93.93 | 94.39 | 94.94 | 94.93  | 94.64 |
|WRN_40_2 | sep(d16)|   651834   | 94.28 | 94.28 | 94.88 | 94.76  | 94.74 |

Run 2

| Student | type    | params     | Acc   | KD    | AT1   | AT2    | AT3   |
|---------|---------|------------|-------|-------|-------|--------|-------|
|WRN_40_2 | sep(d1) |   304074   |       | 92.10 | 93.24 | 93.79  | 93.46 |
|WRN_40_2 | sep(d2) |   327258   |       | 93.20 | 94.29 |        |       |
|WRN_40_2 | sep(d4) |   373626   | 93.07 | 93.46 | 94.60 | 94.11  | 94.44 |
|WRN_40_2 | sep(d8) |   466362   | 93.93 | 93.99 | 94.46 | 94.74  | 94.80 |
|WRN_40_2 | sep(d16)|   651834   | 94.28 | 93.94 | 94.89 | 95.11  | 94.83 |


Conclusions:
- Attention transfer is fantastic for students with grouped convolutions
- Additional points don't really help :'(
