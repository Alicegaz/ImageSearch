import h5py
from main import *
k_best = [32, 64, 128, 256]
with open('validation_queries.dat') as f:
    lines = f.readlines()
queries = [line.rstrip('\n') for line in open('validation_queries.dat')]
for k in k_best:
    name = "vocab"+str(k)+'.h5'
    h5 = h5py.File(name, 'r')
    vocab = h5['vocab'][:]
    res = train(queries[:2], vocab[:10])
    name = "results"+str(k)+".txt"
    save_results(res, name)
    print("results"+str(k)+"is written!!!!!!")
