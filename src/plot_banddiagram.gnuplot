set terminal postscript landscape  
set xlabel "Punktindex im k-Raum"
set ylabel "Frequenz (omega*a/2*pi*c)"
set yrange [0:4.5]
set xrange[0:123]
set title "Baenderdiagramm eines photonischen Kristalls"
set output "~/band_diagram.ps"
plot ",,band_diagram.data" notitle with linespoints pointsize 0.3
set output "~/band_diagram_points_only.ps"
plot ",,band_diagram.data" notitle with points pointsize 0.5
