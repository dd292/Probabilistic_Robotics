import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np



#
# plot for part b
# r= np.array([1/64, 1/16, 1/4, 4, 16, 64])
# data = np.load('mean_Anees_custom_PF.npy')
# colors=cm.rainbow(np.linspace(0, 1, 10))
# for j in range(6):
#     for i in range(10):
#         plt.scatter(np.log(r[j]),np.log(data[j][i]),color=colors[i])
#
# plt.xlabel('log_e(r)')
# plt.ylabel('log_e(Mean position error)')
# plt.title('log-log plot of mean position error vs r value')
# plt.legend(('trail 1','trail 2','trail 3','trail 4','trail 5', 'trail 6','trail 7','trail 8','trial 9','trail 10'))
#
# plt.figure()
# for j in range(6):
#         plt.scatter(r[j],data[j].mean(),color= 'b')
# plt.xlabel('r')
# plt.ylabel('Mean position error')
# plt.title('Plot of mean position error vs r value')
# plt.legend(("Mean of 10 trial runs",))
#
# plt.show()

## plot for part C
# r= np.array([1/64, 1/16, 1/4, 4, 16, 64])
# data = np.load('mean_Anees_default.npy')
# print(data.shape)
# for j in range(6):
#     plt.scatter(np.log(r[j]),np.log(data[0][j]),color='b')
#     plt.scatter(np.log(r[j]),np.log(data[1][j]),color='r')
#
# plt.xlabel('log_e(r)')
# plt.ylabel('log_e(error)')
# plt.title('log-log plot of mean position and ANEES error vs r value')
# plt.legend(("Means Position error",'ANEES error'))
#
# plt.figure()
# for j in range(6):
#         plt.scatter(r[j],data[0][j].mean(),color= 'b')
#         plt.scatter(r[j],data[1][j].mean(),color= 'r')
# plt.xlabel('r')
# plt.ylabel('error')
# plt.title('Plot of mean position adn ANEES error vs r value')
# plt.legend(("Means Position error",'ANEES error'))
#
# plt.show()

#plot part d
r= np.array([1/64, 1/16, 1/4, 4, 16, 64])
par=np.array([20,50,500])
data = np.load('mean_Anees_particles_default.npy')
colors= np.array([['r','g'],['c','m'],['y','k']])
print(colors.shape)


print(data.shape)
for i in range(3):
    plt.plot(np.log(r),np.log(data[:,i,0]),color=colors[i,0])
    plt.plot(np.log(r),np.log(data[:,i,1]),color=colors[i,1])


plt.xlabel('log_e(r)')
plt.ylabel('log_e(Error)')
plt.title('log-log plot of mean position adn ANEES error vs r value')
plt.legend(('Mean Position error N=20','ANEES error N=20','Mean Position error N=50','ANEES error N=50','Mean Position error N=100','ANEES error N=100'))

plt.figure()
for i in range(3):
    plt.plot((r),(data[:,i,0]),color=colors[i,0])
    plt.plot((r),(data[:,i,1]),color=colors[i,1])

plt.xlabel('r')
plt.ylabel('Error')
plt.legend(('Mean Position error N=20','ANEES error N=20','Mean Position error N=50','ANEES error N=50','Mean Position error N=100','ANEES error N=100'))


plt.show()

