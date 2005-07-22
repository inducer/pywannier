!python guess_derivatives.py ,,f0.data
!python guess_derivatives.py ,,f1.data
set grid
plot \
 ",,f0.data" title "Omega_D", \
 ",,f2.data" using 1:($2*1e-2) title "0.1 grad Omega_D New" with lines, \
 ",,f0.data.deriv" using 1:($2*1e-2) title "0.1 grad Omega_D FinDiff" with lines, \
 ",,f1.data" title "Omega_OD", \
 ",,f3.data" using 1:($2*1e-1) title "0.1 grad Omega_OD New" with lines, \
 ",,f1.data.deriv" using 1:($2*1e-1) title "0.1 grad Omega_OD FinDiff" with lines
# ",,f5.data" using 1:(-$2*1e-1) title "-0.1 grad Omega_OD Marzari", \
# ",,f4.data" using 1:(-$2*1e-2) title "-0.1 grad Omega_D Marzari", \
