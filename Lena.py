import numpy as np
import matplotlib.pylab as plt

Lena = plt.imread('Lena.tif')

plt.title('Part A: Lena.tif was opened.')
plt.imshow(Lena)
plt.show()


print("\nPart B: Total number entries that we need: " + str(Lena.shape[0]*Lena.shape[1]))


U, S, V = np.linalg.svd(Lena,full_matrices = False)
S = np.diag(S)
print('\nPart C:')
print('The shape of U: ' + str(U.shape))
print('The shape of S: ' + str(S.shape))
print('The shape of V: ' + str(V.shape))


print('\nPart D and E:')
LenaReproduce = np.zeros((450,512))
ranks = [3, 10, 20, 30, 50, 60, 70, 80, 100]
MSE = []

for r in ranks:
    LenaReproduce = np.matmul(S[0:r,0:r], V[0:r, 0:512])
    LenaReproduce = np.matmul(U[0:450, 0:r], LenaReproduce)
    
    MSE.append(np.sum(np.sqrt((Lena - LenaReproduce)**2)))
    NumberofEntries = LenaReproduce.shape[0]*r + \
                        r*r + r*LenaReproduce.shape[1]
    plt.title('Rank is: ' + str(r)+', Entries: ' + str(NumberofEntries))
    plt.imshow(LenaReproduce)
    plt.show()

plt.title('The Erros Values For Each Rank')
plt.xlabel('Number of Ranks')
plt.ylabel('The Errors')
plt.plot(ranks, MSE)
plt.show()