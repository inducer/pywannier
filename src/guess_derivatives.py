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
for n in range(len(x)):
    if n == 0:
        deriv = (y[n+1]-y[n])/(x[n+1]-x[n])
    elif n == len(x)-1:
        deriv = (y[n]-y[n-1])/(x[n]-x[n-1])
    else:
        deriv = (y[n+1]-y[n-1])/(x[n+1]-x[n-1])

    df.write("%f\t%f\n" % (x[n], deriv))
    

