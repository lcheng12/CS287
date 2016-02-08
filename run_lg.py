import os

iters = 200
samplesize = 1024
l = 2

for i in range(1, 10):
  eta = float(i)/10
  print("th new.lua -datafile SST1.hdf5 -classifier lg -eta %f -lambda %f -samplesize %d -iters %d" % (eta, l, samplesize, iters))
  os.system("th new.lua -datafile SST1.hdf5 -classifier lg -eta %f -lambda %f -samplesize %d -iters %d" % (eta, l, samplesize, iters))
  print i
