!python guess_derivatives.py ,,f0.data
!python guess_derivatives.py ,,f1.data
set grid
plot \
 ",,f0.data" title "o_d", \
 ",,f5.data" using 1:($2*1e-2) title "grad_d", \
 ",,f0.data.deriv" using 1:($2*1e-2) title "grad_d real"
#",,f1.data" title "o_od", \
#",,f4.data" using 1:($2*1e-1) title "grad_od", \
# ",,f1.data.deriv" using 1:($2*1e-1) title "grad_od real"
# ",,f3.data" using 1:($2*1e-2) title "grad", \
#",,f2.data" title "o_d+o_od", \
