import sys

fn = sys.argv[1]
lines = file(fn, "r").readlines()
x = []
y = []
for line in lines:
    values = line.split("\t")
    x.append(float(values[0]))
    y.append(float(values[1]))

df = file(fn + ".deriv", "w")
for n in range(len(x)-1):
    deriv = (y[n+1]-y[n])/(x[n+1]-x[n])
    df.write("%f\t%f\n" % (x[n], deriv))
    

