from typing import Tuple
from io import StringIO

import numpy as np

__vertex_data = """
0.000000 0.000000 -1.000000 0.102381 -0.315090 -0.943523
0.425323 -0.309011 -0.850654 0.102381 -0.315090 -0.943523
-0.162456 -0.499995 -0.850654 0.102381 -0.315090 -0.943523
0.723607 -0.525725 -0.447220 0.700224 -0.268032 -0.661699
0.425323 -0.309011 -0.850654 0.700224 -0.268032 -0.661699
0.850648 0.000000 -0.525736 0.700224 -0.268032 -0.661699
0.000000 0.000000 -1.000000 -0.268034 -0.194736 -0.943523
-0.162456 -0.499995 -0.850654 -0.268034 -0.194736 -0.943523
-0.525730 0.000000 -0.850652 -0.268034 -0.194736 -0.943523
0.000000 0.000000 -1.000000 -0.268034 0.194737 -0.943523
-0.525730 0.000000 -0.850652 -0.268034 0.194737 -0.943523
-0.162456 0.499995 -0.850654 -0.268034 0.194737 -0.943523
0.000000 0.000000 -1.000000 0.102381 0.315090 -0.943523
-0.162456 0.499995 -0.850654 0.102381 0.315090 -0.943523
0.425323 0.309011 -0.850654 0.102381 0.315090 -0.943523
0.723607 -0.525725 -0.447220 0.904989 -0.268032 -0.330385
0.850648 0.000000 -0.525736 0.904989 -0.268032 -0.330385
0.951058 -0.309013 0.000000 0.904989 -0.268032 -0.330385
-0.276388 -0.850649 -0.447220 0.024747 -0.943521 -0.330386
0.262869 -0.809012 -0.525738 0.024747 -0.943521 -0.330386
0.000000 -1.000000 0.000000 0.024747 -0.943521 -0.330386
-0.894426 0.000000 -0.447216 -0.889697 -0.315095 -0.330385
-0.688189 -0.499997 -0.525736 -0.889697 -0.315095 -0.330385
-0.951058 -0.309013 0.000000 -0.889697 -0.315095 -0.330385
-0.276388 0.850649 -0.447220 -0.574602 0.748784 -0.330388
-0.688189 0.499997 -0.525736 -0.574602 0.748784 -0.330388
-0.587786 0.809017 0.000000 -0.574602 0.748784 -0.330388
0.723607 0.525725 -0.447220 0.534576 0.777865 -0.330387
0.262869 0.809012 -0.525738 0.534576 0.777865 -0.330387
0.587786 0.809017 0.000000 0.534576 0.777865 -0.330387
0.723607 -0.525725 -0.447220 0.802609 -0.583126 -0.125627
0.951058 -0.309013 0.000000 0.802609 -0.583126 -0.125627
0.587786 -0.809017 0.000000 0.802609 -0.583126 -0.125627
-0.276388 -0.850649 -0.447220 -0.306569 -0.943522 -0.125629
0.000000 -1.000000 0.000000 -0.306569 -0.943522 -0.125629
-0.587786 -0.809017 0.000000 -0.306569 -0.943522 -0.125629
-0.894426 0.000000 -0.447216 -0.992077 -0.000000 -0.125628
-0.951058 -0.309013 0.000000 -0.992077 -0.000000 -0.125628
-0.951058 0.309013 0.000000 -0.992077 -0.000000 -0.125628
-0.276388 0.850649 -0.447220 -0.306569 0.943522 -0.125629
-0.587786 0.809017 0.000000 -0.306569 0.943522 -0.125629
0.000000 1.000000 0.000000 -0.306569 0.943522 -0.125629
0.723607 0.525725 -0.447220 0.802609 0.583126 -0.125627
0.587786 0.809017 0.000000 0.802609 0.583126 -0.125627
0.951058 0.309013 0.000000 0.802609 0.583126 -0.125627
0.276388 -0.850649 0.447220 0.408946 -0.628425 0.661698
0.688189 -0.499997 0.525736 0.408946 -0.628425 0.661698
0.162456 -0.499995 0.850654 0.408946 -0.628425 0.661698
-0.723607 -0.525725 0.447220 -0.471300 -0.583122 0.661699
-0.262869 -0.809012 0.525738 -0.471300 -0.583122 0.661699
-0.425323 -0.309011 0.850654 -0.471300 -0.583122 0.661699
-0.723607 0.525725 0.447220 -0.700224 0.268032 0.661699
-0.850648 0.000000 0.525736 -0.700224 0.268032 0.661699
-0.425323 0.309011 0.850654 -0.700224 0.268032 0.661699
0.276388 0.850649 0.447220 0.038530 0.748779 0.661699
-0.262869 0.809012 0.525738 0.038530 0.748779 0.661699
0.162456 0.499995 0.850654 0.038530 0.748779 0.661699
0.894426 0.000000 0.447216 0.724042 0.194736 0.661695
0.688189 0.499997 0.525736 0.724042 0.194736 0.661695
0.525730 0.000000 0.850652 0.724042 0.194736 0.661695
0.525730 0.000000 0.850652 0.268034 0.194737 0.943523
0.162456 0.499995 0.850654 0.268034 0.194737 0.943523
0.000000 0.000000 1.000000 0.268034 0.194737 0.943523
0.525730 0.000000 0.850652 0.491119 0.356821 0.794657
0.688189 0.499997 0.525736 0.491119 0.356821 0.794657
0.162456 0.499995 0.850654 0.491119 0.356821 0.794657
0.688189 0.499997 0.525736 0.408946 0.628425 0.661699
0.276388 0.850649 0.447220 0.408946 0.628425 0.661699
0.162456 0.499995 0.850654 0.408946 0.628425 0.661699
0.162456 0.499995 0.850654 -0.102381 0.315090 0.943523
-0.425323 0.309011 0.850654 -0.102381 0.315090 0.943523
0.000000 0.000000 1.000000 -0.102381 0.315090 0.943523
0.162456 0.499995 0.850654 -0.187594 0.577345 0.794658
-0.262869 0.809012 0.525738 -0.187594 0.577345 0.794658
-0.425323 0.309011 0.850654 -0.187594 0.577345 0.794658
-0.262869 0.809012 0.525738 -0.471300 0.583122 0.661699
-0.723607 0.525725 0.447220 -0.471300 0.583122 0.661699
-0.425323 0.309011 0.850654 -0.471300 0.583122 0.661699
-0.425323 0.309011 0.850654 -0.331305 0.000000 0.943524
-0.425323 -0.309011 0.850654 -0.331305 0.000000 0.943524
0.000000 0.000000 1.000000 -0.331305 0.000000 0.943524
-0.425323 0.309011 0.850654 -0.607060 0.000000 0.794656
-0.850648 0.000000 0.525736 -0.607060 0.000000 0.794656
-0.425323 -0.309011 0.850654 -0.607060 0.000000 0.794656
-0.850648 0.000000 0.525736 -0.700224 -0.268032 0.661699
-0.723607 -0.525725 0.447220 -0.700224 -0.268032 0.661699
-0.425323 -0.309011 0.850654 -0.700224 -0.268032 0.661699
-0.425323 -0.309011 0.850654 -0.102381 -0.315090 0.943523
0.162456 -0.499995 0.850654 -0.102381 -0.315090 0.943523
0.000000 0.000000 1.000000 -0.102381 -0.315090 0.943523
-0.425323 -0.309011 0.850654 -0.187594 -0.577345 0.794658
-0.262869 -0.809012 0.525738 -0.187594 -0.577345 0.794658
0.162456 -0.499995 0.850654 -0.187594 -0.577345 0.794658
-0.262869 -0.809012 0.525738 0.038530 -0.748779 0.661699
0.276388 -0.850649 0.447220 0.038530 -0.748779 0.661699
0.162456 -0.499995 0.850654 0.038530 -0.748779 0.661699
0.162456 -0.499995 0.850654 0.268034 -0.194737 0.943523
0.525730 0.000000 0.850652 0.268034 -0.194737 0.943523
0.000000 0.000000 1.000000 0.268034 -0.194737 0.943523
0.162456 -0.499995 0.850654 0.491119 -0.356821 0.794657
0.688189 -0.499997 0.525736 0.491119 -0.356821 0.794657
0.525730 0.000000 0.850652 0.491119 -0.356821 0.794657
0.688189 -0.499997 0.525736 0.724042 -0.194736 0.661695
0.894426 0.000000 0.447216 0.724042 -0.194736 0.661695
0.525730 0.000000 0.850652 0.724042 -0.194736 0.661695
0.951058 0.309013 0.000000 0.889697 0.315095 0.330385
0.688189 0.499997 0.525736 0.889697 0.315095 0.330385
0.894426 0.000000 0.447216 0.889697 0.315095 0.330385
0.951058 0.309013 0.000000 0.794656 0.577348 0.187595
0.587786 0.809017 0.000000 0.794656 0.577348 0.187595
0.688189 0.499997 0.525736 0.794656 0.577348 0.187595
0.587786 0.809017 0.000000 0.574602 0.748784 0.330388
0.276388 0.850649 0.447220 0.574602 0.748784 0.330388
0.688189 0.499997 0.525736 0.574602 0.748784 0.330388
0.000000 1.000000 0.000000 -0.024747 0.943521 0.330386
-0.262869 0.809012 0.525738 -0.024747 0.943521 0.330386
0.276388 0.850649 0.447220 -0.024747 0.943521 0.330386
0.000000 1.000000 0.000000 -0.303531 0.934171 0.187597
-0.587786 0.809017 0.000000 -0.303531 0.934171 0.187597
-0.262869 0.809012 0.525738 -0.303531 0.934171 0.187597
-0.587786 0.809017 0.000000 -0.534576 0.777865 0.330387
-0.723607 0.525725 0.447220 -0.534576 0.777865 0.330387
-0.262869 0.809012 0.525738 -0.534576 0.777865 0.330387
-0.951058 0.309013 0.000000 -0.904989 0.268032 0.330385
-0.850648 0.000000 0.525736 -0.904989 0.268032 0.330385
-0.723607 0.525725 0.447220 -0.904989 0.268032 0.330385
-0.951058 0.309013 0.000000 -0.982246 0.000000 0.187599
-0.951058 -0.309013 0.000000 -0.982246 0.000000 0.187599
-0.850648 0.000000 0.525736 -0.982246 0.000000 0.187599
-0.951058 -0.309013 0.000000 -0.904989 -0.268031 0.330385
-0.723607 -0.525725 0.447220 -0.904989 -0.268031 0.330385
-0.850648 0.000000 0.525736 -0.904989 -0.268031 0.330385
-0.587786 -0.809017 0.000000 -0.534576 -0.777865 0.330387
-0.262869 -0.809012 0.525738 -0.534576 -0.777865 0.330387
-0.723607 -0.525725 0.447220 -0.534576 -0.777865 0.330387
-0.587786 -0.809017 0.000000 -0.303531 -0.934171 0.187597
0.000000 -1.000000 0.000000 -0.303531 -0.934171 0.187597
-0.262869 -0.809012 0.525738 -0.303531 -0.934171 0.187597
0.000000 -1.000000 0.000000 -0.024747 -0.943521 0.330386
0.276388 -0.850649 0.447220 -0.024747 -0.943521 0.330386
-0.262869 -0.809012 0.525738 -0.024747 -0.943521 0.330386
0.587786 -0.809017 0.000000 0.574602 -0.748784 0.330388
0.688189 -0.499997 0.525736 0.574602 -0.748784 0.330388
0.276388 -0.850649 0.447220 0.574602 -0.748784 0.330388
0.587786 -0.809017 0.000000 0.794656 -0.577348 0.187595
0.951058 -0.309013 0.000000 0.794656 -0.577348 0.187595
0.688189 -0.499997 0.525736 0.794656 -0.577348 0.187595
0.951058 -0.309013 0.000000 0.889697 -0.315095 0.330385
0.894426 0.000000 0.447216 0.889697 -0.315095 0.330385
0.688189 -0.499997 0.525736 0.889697 -0.315095 0.330385
0.587786 0.809017 0.000000 0.306569 0.943522 0.125629
0.000000 1.000000 0.000000 0.306569 0.943522 0.125629
0.276388 0.850649 0.447220 0.306569 0.943522 0.125629
0.587786 0.809017 0.000000 0.303531 0.934171 -0.187597
0.262869 0.809012 -0.525738 0.303531 0.934171 -0.187597
0.000000 1.000000 0.000000 0.303531 0.934171 -0.187597
0.262869 0.809012 -0.525738 0.024747 0.943521 -0.330386
-0.276388 0.850649 -0.447220 0.024747 0.943521 -0.330386
0.000000 1.000000 0.000000 0.024747 0.943521 -0.330386
-0.587786 0.809017 0.000000 -0.802609 0.583126 0.125627
-0.951058 0.309013 0.000000 -0.802609 0.583126 0.125627
-0.723607 0.525725 0.447220 -0.802609 0.583126 0.125627
-0.587786 0.809017 0.000000 -0.794656 0.577348 -0.187595
-0.688189 0.499997 -0.525736 -0.794656 0.577348 -0.187595
-0.951058 0.309013 0.000000 -0.794656 0.577348 -0.187595
-0.688189 0.499997 -0.525736 -0.889697 0.315095 -0.330385
-0.894426 0.000000 -0.447216 -0.889697 0.315095 -0.330385
-0.951058 0.309013 0.000000 -0.889697 0.315095 -0.330385
-0.951058 -0.309013 0.000000 -0.802609 -0.583126 0.125627
-0.587786 -0.809017 0.000000 -0.802609 -0.583126 0.125627
-0.723607 -0.525725 0.447220 -0.802609 -0.583126 0.125627
-0.951058 -0.309013 0.000000 -0.794656 -0.577348 -0.187595
-0.688189 -0.499997 -0.525736 -0.794656 -0.577348 -0.187595
-0.587786 -0.809017 0.000000 -0.794656 -0.577348 -0.187595
-0.688189 -0.499997 -0.525736 -0.574602 -0.748784 -0.330388
-0.276388 -0.850649 -0.447220 -0.574602 -0.748784 -0.330388
-0.587786 -0.809017 0.000000 -0.574602 -0.748784 -0.330388
0.000000 -1.000000 0.000000 0.306569 -0.943522 0.125629
0.587786 -0.809017 0.000000 0.306569 -0.943522 0.125629
0.276388 -0.850649 0.447220 0.306569 -0.943522 0.125629
0.000000 -1.000000 0.000000 0.303531 -0.934171 -0.187597
0.262869 -0.809012 -0.525738 0.303531 -0.934171 -0.187597
0.587786 -0.809017 0.000000 0.303531 -0.934171 -0.187597
0.262869 -0.809012 -0.525738 0.534576 -0.777865 -0.330387
0.723607 -0.525725 -0.447220 0.534576 -0.777865 -0.330387
0.587786 -0.809017 0.000000 0.534576 -0.777865 -0.330387
0.951058 -0.309013 0.000000 0.992077 0.000000 0.125628
0.951058 0.309013 0.000000 0.992077 0.000000 0.125628
0.894426 0.000000 0.447216 0.992077 0.000000 0.125628
0.951058 -0.309013 0.000000 0.982246 0.000000 -0.187599
0.850648 0.000000 -0.525736 0.982246 0.000000 -0.187599
0.951058 0.309013 0.000000 0.982246 0.000000 -0.187599
0.850648 0.000000 -0.525736 0.904989 0.268031 -0.330385
0.723607 0.525725 -0.447220 0.904989 0.268031 -0.330385
0.951058 0.309013 0.000000 0.904989 0.268031 -0.330385
0.425323 0.309011 -0.850654 0.471300 0.583122 -0.661699
0.262869 0.809012 -0.525738 0.471300 0.583122 -0.661699
0.723607 0.525725 -0.447220 0.471300 0.583122 -0.661699
0.425323 0.309011 -0.850654 0.187594 0.577345 -0.794658
-0.162456 0.499995 -0.850654 0.187594 0.577345 -0.794658
0.262869 0.809012 -0.525738 0.187594 0.577345 -0.794658
-0.162456 0.499995 -0.850654 -0.038530 0.748779 -0.661699
-0.276388 0.850649 -0.447220 -0.038530 0.748779 -0.661699
0.262869 0.809012 -0.525738 -0.038530 0.748779 -0.661699
-0.162456 0.499995 -0.850654 -0.408946 0.628425 -0.661698
-0.688189 0.499997 -0.525736 -0.408946 0.628425 -0.661698
-0.276388 0.850649 -0.447220 -0.408946 0.628425 -0.661698
-0.162456 0.499995 -0.850654 -0.491119 0.356821 -0.794657
-0.525730 0.000000 -0.850652 -0.491119 0.356821 -0.794657
-0.688189 0.499997 -0.525736 -0.491119 0.356821 -0.794657
-0.525730 0.000000 -0.850652 -0.724042 0.194736 -0.661695
-0.894426 0.000000 -0.447216 -0.724042 0.194736 -0.661695
-0.688189 0.499997 -0.525736 -0.724042 0.194736 -0.661695
-0.525730 0.000000 -0.850652 -0.724042 -0.194736 -0.661695
-0.688189 -0.499997 -0.525736 -0.724042 -0.194736 -0.661695
-0.894426 0.000000 -0.447216 -0.724042 -0.194736 -0.661695
-0.525730 0.000000 -0.850652 -0.491119 -0.356821 -0.794657
-0.162456 -0.499995 -0.850654 -0.491119 -0.356821 -0.794657
-0.688189 -0.499997 -0.525736 -0.491119 -0.356821 -0.794657
-0.162456 -0.499995 -0.850654 -0.408946 -0.628425 -0.661698
-0.276388 -0.850649 -0.447220 -0.408946 -0.628425 -0.661698
-0.688189 -0.499997 -0.525736 -0.408946 -0.628425 -0.661698
0.850648 0.000000 -0.525736 0.700224 0.268032 -0.661699
0.425323 0.309011 -0.850654 0.700224 0.268032 -0.661699
0.723607 0.525725 -0.447220 0.700224 0.268032 -0.661699
0.850648 0.000000 -0.525736 0.607060 0.000000 -0.794656
0.425323 -0.309011 -0.850654 0.607060 0.000000 -0.794656
0.425323 0.309011 -0.850654 0.607060 0.000000 -0.794656
0.425323 -0.309011 -0.850654 0.331305 0.000000 -0.943524
0.000000 0.000000 -1.000000 0.331305 0.000000 -0.943524
0.425323 0.309011 -0.850654 0.331305 0.000000 -0.943524
-0.162456 -0.499995 -0.850654 -0.038530 -0.748779 -0.661699
0.262869 -0.809012 -0.525738 -0.038530 -0.748779 -0.661699
-0.276388 -0.850649 -0.447220 -0.038530 -0.748779 -0.661699
-0.162456 -0.499995 -0.850654 0.187594 -0.577345 -0.794658
0.425323 -0.309011 -0.850654 0.187594 -0.577345 -0.794658
0.262869 -0.809012 -0.525738 0.187594 -0.577345 -0.794658
0.425323 -0.309011 -0.850654 0.471300 -0.583122 -0.661699
0.723607 -0.525725 -0.447220 0.471300 -0.583122 -0.661699
0.262869 -0.809012 -0.525738 0.471300 -0.583122 -0.661699
"""

__element_indices = """
0 1 2
3 4 5
6 7 8
9 10 11
12 13 14
15 16 17
18 19 20
21 22 23
24 25 26
27 28 29
30 31 32
33 34 35
36 37 38
39 40 41
42 43 44
45 46 47
48 49 50
51 52 53
54 55 56
57 58 59
60 61 62
63 64 65
66 67 68
69 70 71
72 73 74
75 76 77
78 79 80
81 82 83
84 85 86
87 88 89
90 91 92
93 94 95
96 97 98
99 100 101
102 103 104
105 106 107
108 109 110
111 112 113
114 115 116
117 118 119
120 121 122
123 124 125
126 127 128
129 130 131
132 133 134
135 136 137
138 139 140
141 142 143
144 145 146
147 148 149
150 151 152
153 154 155
156 157 158
159 160 161
162 163 164
165 166 167
168 169 170
171 172 173
174 175 176
177 178 179
180 181 182
183 184 185
186 187 188
189 190 191
192 193 194
195 196 197
198 199 200
201 202 203
204 205 206
207 208 209
210 211 212
213 214 215
216 217 218
219 220 221
222 223 224
225 226 227
228 229 230
231 232 233
234 235 236
237 238 239
"""


def sphere_model_data() -> Tuple[np.ndarray, np.ndarray]:
    """Returns the vertex data, and element array of a low poly sphere (80 triangles)"""
    vertex_and_normals = np.loadtxt(StringIO(__vertex_data), delimiter=" ", dtype=np.float32)
    element_indices = np.loadtxt(StringIO(__element_indices), delimiter=' ', dtype=np.int32)

    return vertex_and_normals, element_indices