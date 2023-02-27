from LinearRegression import *
import numpy as np
import matplotlib.pyplot as plt

x = 2*np.random.rand(100,1)
y = 4+ 3*x + np.random.randn(100,1)

# x1,y1 = 1,1
# x2, y2 = 2,2

m,n = LinearRegression(x,y) #[x1,x2], [y1,y2])
print(m)
print(n)
plt.plot(x,y, 'o')

x1 = max(x)
x2 = min(x)

y1 = m*x1+n
y2 = m*x2+n

plt.plot([x1,x2], [y1,y2])
plt.legend(["data points", "regression line"])
plt.show()