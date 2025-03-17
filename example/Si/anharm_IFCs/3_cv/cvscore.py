#!/home/app/coreutils/8.23/bin/env python
import numpy as np
import sys

files = sys.argv[1:]
nfold = len(files)

if nfold == 0:
    print("Please specify the files")
    exit(1)

alpha = np.loadtxt(
    files[0],
    usecols=[
        0,
    ],
)
nalpha = len(alpha)

data1 = np.zeros((nalpha, nfold))
data2 = np.zeros((nalpha, nfold))

icount = 0
for file in files:
    data_tmp = np.loadtxt(file, usecols=[1, 2])
    if len(data_tmp) != nalpha:
        print("Inconsistent number of entries: %s" % file)
        exit(1)

    data1[:, icount] = data_tmp[:, 0]
    data2[:, icount] = data_tmp[:, 1]
    icount += 1

mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)

std1 = np.std(data1, axis=1)
std2 = np.std(data2, axis=1)

print("#alpha, fitting error (mean, dev.), cvscore (mean, dev.)")
for i in range(nalpha):
    print(
        "%12.5e %12.5e %12.5e %12.5e %12.5e"
        % (alpha[i], mean1[i], std1[i], mean2[i], std2[i])
    )

minloc = np.argmin(mean2)

print("#Minimum cvscore at ", alpha[minloc])
