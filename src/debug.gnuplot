plot ",,f0.data" title "o_d", \
 ",,f1.data" title "o_od", \
 ",,f2.data" title "o_d+o_od", \
 ",,f3.data" using 1:($2*1e-4) title "grad", \
 ",,f4.data" using 1:($2*1e-2) title "grad_od", \
 ",,f5.data" using 1:($2*1e-4) title "grad_d"
