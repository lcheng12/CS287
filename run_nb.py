import os

for i in range(0, 1):
  alpha = float(i)/10
  #print("th new.lua -datafile SST1.hdf5 -classifier nb -alpha %f" % (alpha))
  os.system("th new.lua -datafile SST1.hdf5 -classifier nb -alpha %f" % (alpha))
