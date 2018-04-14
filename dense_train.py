from feature_extract import normalize, get_vocabb
from search import train, save_results
import h5py
from sklearn.cluster import MiniBatchKMeans

h5 = h5py.File('ddescriptors_res.h5', 'r')
descriptors = h5['descriptors'][:]
lengthe = h5['lenghtes'][:]
h5.close()
descript_stack = descriptors

k_best = [32, 64, 128, 256]
for k_best1 in k_best:
    print('Kmeans '+k_best1)
    kmeans_ret1 = MiniBatchKMeans(n_clusters=k_best1, random_state=0).fit(descriptors)
    h5 = h5py.File('dense_vocab'+str(k_best1)+'.h5', 'w')
    print('clusters finished')
    vocab1 = normalize(get_vocabb(len(lengthe), descriptors, k_best1, kmeans_ret1, lengthe))
    h5.create_dataset('vocab', data=vocab1)
    h5.close()
    print('dense_vocab'+str(k_best1)+"finished"+"!!!!!!!")

with open('validation_queries.dat') as f:
    lines = f.readlines()
queries = [line.rstrip('\n') for line in open('validation_queries.dat')]
for k in k_best:
    name = "dense_vocab"+str(k)+'.h5'
    h5 = h5py.File(name, 'r')
    vocab = h5['vocab'][:]
    res = train(queries, vocab)
    name = "dense_results"+str(k)+".txt"
    save_results(res, name)
    print("dense_results"+str(k)+"is written!!!!!!")
