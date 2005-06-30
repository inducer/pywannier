!python guess_derivatives.py ,,f0.data
!python guess_derivatives.py ,,f1.data
set grid
plot \
 ",,f0.data" title "o_d", \
 ",,f2.data" using 1:($2*1e-2) title "0.1 grad_d New" with lines, \
 ",,f4.data" using 1:(-$2*1e-2) title "0.1 grad_d Marzari", \
 ",,f0.data.deriv" using 1:($2*1e-2) title "0.1 grad_d FinDiff" with lines, \
 ",,f1.data" title "o_od", \
 ",,f3.data" using 1:($2*1e-1) title "0.1 grad_od New" with lines, \
 ",,f5.data" using 1:(-$2*1e-1) title "0.1 grad_od Marzari", \
 ",,f1.data.deriv" using 1:($2*1e-1) title "0.1 grad_od FinDiff" with lines
