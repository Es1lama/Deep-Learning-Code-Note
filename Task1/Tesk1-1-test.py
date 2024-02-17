# 2024/2/14
'''
1.⽤体测50⽶跑成绩和肺活量为⾃变量，1000⽶⽤时为因变量，以线性回归⽅
式，画出线性回归后的平⾯（z=ax+by+c)（⽤⾃⼰班的体测成绩或者随便找点）
示例数据(xls)：
性别	肺活量测试成绩	换算分数	50测试成绩	换算分数	长跑测试成绩	换算分数
男	1530	0	6.11	100	3'40"	80
男	1532	0	8.498	66	4'10"	68
男	1556	0	6.674	95	4'07"	68

'''

'''
1、加载数据
2、归一化
3、cost函数(正向传播)
4、梯度下降(反向传播)
5、画出平面
*6、如果需要进行模型预测，则需要把归一化中的
mu_y=y_train.sum()/m
dif=y_train.max()-y_train.min()
y_train=(y_train-mu_y)/dif
逆运算
'''
import xlrd
import numpy as np
import math
import matplotlib.pyplot as plt
import copy


#1-1加载数据，肺活量在B列，50米在D列，长跑在F列
col_x =[ord('B')-ord('A'),ord('D')-ord('A')]
col_y=ord('F')-ord('A')
DataPath=r'./DataFix.xls'#获取训练数据路径
x_train=[]
y_train=[]
worksheet=xlrd.open_workbook(DataPath).sheet_by_index(0)

for row_num in range(27,297):
    x_train.append([int(worksheet.cell(row_num,col_x[0]).value),float(worksheet.cell(row_num,col_x[1]).value)])#一个int一个float

    temp_y=worksheet.cell(row_num,col_y).value
    tmie_parts=list(filter(str.isdigit,str(temp_y)))#提取所有数字部分
    minutes=int(tmie_parts[0])*60
    seconds=int(''.join(tmie_parts[1:]))#合并剩余数字为秒数
    total_seconds=minutes+seconds
    y_train.append(total_seconds)

#1-2转化为np数组，并且查看是否正确加载数据

x_train=np.array(x_train)
y_train=np.array(y_train)



for i in range(len(x_train)):
    xi = x_train[i]
    yi = y_train[i]
    print("{:<{width}} | {:<{width}}".format(str(xi), str(yi), width=25))

#2-1归一化mean normalization
m,n=x_train.shape


x_train_s=copy.deepcopy(x_train)
y_train_s=copy.deepcopy(y_train)

mu_x0=x_train[:,0].sum()/m
dif=x_train[:,0].max()-x_train[:,0].min()
x_train_s[:,0]=(x_train[:,0]-mu_x0)/dif

mu_x1=x_train[:,1].sum()/m
dif=x_train[:,1].max()-x_train[:,1].min()
x_train_s[:,1]=(x_train[:,1]-mu_x1)/dif

# mu_y=y_train.sum()/m
# dif=y_train.max()-y_train.min()
# y_train=(y_train-mu_y)/dif
for i in range(len(x_train_s)):
    xi = x_train_s[i]
    yi = y_train[i]
    print("{:<{width}} | {:<{width}}".format(str(xi), str(yi), width=25))



'''
测试数据
x_train_s=np.array([[1,1],[2,2],[3,3]])
y_train=np.array([1,2,3])

出来的结果是
Iteration    0: Cost 2.101620370370370   
Iteration 50000: Cost 0.000000000000000   
……
b,w found by gradient descent: 0.0000000000000160982057444,[[0.5]
 [0.5]] 
prediction: 1.0000000000, target value: 1
prediction: 2.0000000000, target value: 2
prediction: 3.0000000000, target value: 3
证明接下来的模型是对的
'''







#cost函数
def cost(x,y,w,b):
    m=x.shape[0]
    cost=0.0
    for i in range(m):# 0,1,2,~m-1
        f_wb_i = np.dot(x[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost = cost +(f_wb_i - y[i])**2 # scalar
    cost=cost/m/2
    return cost


def compute_gradient(x, y, w, b):
    m, n = x.shape
    # Predictions (shape: (m,))

    predictions = np.dot(x, w) + b
    # Errors (Residuals) (shape: (m,))差
    y=y.reshape(y.shape[0],1)
    err = predictions - y
    # Gradient w.r.t. weights (dw) (shape: (n,))一维
    dj_dw = 1 / m * np.dot(x.T, err)
    # Gradient w.r.t. bias (db) (shape: scalar)标量
    dj_db = 1 / m * np.sum(err)
    return dj_db, dj_dw

#梯度下降
def gradient_descent(X, y, w_in, b_in,  alpha, num_iters):
    # An array to store cost J and w's at each iteration primarily for graphing later这是一个存储J的历史记录的数组
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function防止全局变量被改变
    b = b_in

    for i in range(num_iters):

        dj_db, dj_dw = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:  # 防止疲劳
            J_history.append(cost(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1][0]:8.15f}   ")

    return w, b, J_history  # return final w,b and J history for graphing

w_init = np.zeros((x_train_s.shape[1],1))
b_init=0
iterations=1000############3加个0精确一点
alpha = 5.0e-2
# run gradient descent
w_final, b_final, J_hist = gradient_descent(x_train_s, y_train, w_init,b_init,  alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.25f},{w_final} ")
m,_ = x_train_s.shape
for i in range(m):
    x_val=x_train_s[i].reshape(1,2)
    print(f"prediction: {np.dot(x_val, w_final)[0][0] + b_final:0.10f}, target value: {y_train[i]}")






from mpl_toolkits.mplot3d import Axes3D


# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点
ax.scatter(x_train[:, 0], x_train[:, 1], y_train, c='b', marker='o', label='Data Points')

#绘制平面
x_grid, y_grid = np.meshgrid(np.linspace(x_train_s[:, 0].min(), x_train_s[:, 0].max(), 100),
                             np.linspace(x_train_s[:, 1].min(), x_train_s[:, 1].max(), 100))
z_plane = w_final[0][0] * x_grid + w_final[1][0] * y_grid + b_final
ax.plot_surface(x_grid, y_grid, z_plane, alpha=0.5, cmap='viridis', label='Plane')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图例
ax.legend()

# 显示图表
plt.show()
"""
梯度下降注释：
Performs batch gradient descent to learn theta. Updates theta by taking
num_iters gradient steps with learning rate alpha

Args:
  X (ndarray (m,n))   : Data, m examples with n features
  y (ndarray (m,))    : target values
  w_in (ndarray (n,)) : initial model parameters
  b_in (scalar)       : initial model parameter
  cost_function       : function to compute cost
  gradient_function   : function to compute the gradient
  alpha (float)       : Learning rate
  num_iters (int)     : number of iterations to run gradient descent

Returns:
  w (ndarray (n,)) : Updated values of parameters
  b (scalar)       : Updated value of parameter
  """