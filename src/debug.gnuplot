!python guess_derivatives.py ,,f0.data
!python guess_derivatives.py ,,f1.data
plot ",,f0.data" title "o_d", \
 ",,f1.data" title "o_od", \
 ",,f2.data" title "o_d+o_od", \
 ",,f3.data" using 1:($2*1e-3) title "grad", \
 ",,f4.data" using 1:($2*1e-1-4) title "grad_od" with lines, \
 ",,f5.data" using 1:($2*1e-3) title "grad_d", \
 ",,f0.data.deriv" using 1:($2*1e-3) title "grad_d real" with lines, \
 ",,f1.data.deriv" using 1:($2*1e-1) title "grad_od real" with lines
