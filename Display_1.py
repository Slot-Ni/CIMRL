import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

# data = "Algebra"
data = "Bars-Rev"
# data = "Restaurants-Rev"
# data = "Geometry"
# data = "Music-Rev"

SNG_mode = 'H_Degree'


#传播参数
beta = 0.05
T = 30

#greedy算法参数
times = 1000
#RL所需参数
epsilon = 0.1
alpha = 0.1
gamma = 0.9
#指标初筛比例
CandidateNodes_Ratio = 0.05
#社区划分方法
method = "louvain_method"


# Alpha = 0.1
# Alpha = 0.2
# Alpha = 0.3
# Alpha = 0.4
# Alpha = 0.5
# Alpha = 0.6
# Alpha = 0.7
# Alpha = 0.8
# Alpha = 0.9
# Alpha = 1

# Order = 1
# Order = 2
# Order = 3
# Order = 4
# Order = 5
# Order = 6
# Order = 7
# Order = 8
# Order = 9
Order = 10



#相同Alpha
# Indicator_selection_1 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(1) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Indicator_selection_2 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(2) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Indicator_selection_3 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(3) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Indicator_selection_4 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(4) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Indicator_selection_5 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(5) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Indicator_selection_6 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(6) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Indicator_selection_7 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(7) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Indicator_selection_8 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(8) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Indicator_selection_9 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(9) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Indicator_selection_10 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(10) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
#
# Result_Indicator_selection_1 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(1) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Result_Indicator_selection_2 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(2) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Result_Indicator_selection_3 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(3) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Result_Indicator_selection_4 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(4) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Result_Indicator_selection_5 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(5) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Result_Indicator_selection_6 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(6) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Result_Indicator_selection_7 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(7) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Result_Indicator_selection_8 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(8) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Result_Indicator_selection_9 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(9) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()
# Result_Indicator_selection_10 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(10) + '_alpha' + str(Alpha) + ').npy', allow_pickle=True).tolist()


#相同Order
Indicator_selection_1 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.1) + ').npy', allow_pickle=True).tolist()
Indicator_selection_2 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.2) + ').npy', allow_pickle=True).tolist()
Indicator_selection_3 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.3) + ').npy', allow_pickle=True).tolist()
Indicator_selection_4 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.4) + ').npy', allow_pickle=True).tolist()
Indicator_selection_5 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.5) + ').npy', allow_pickle=True).tolist()
Indicator_selection_6 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.6) + ').npy', allow_pickle=True).tolist()
Indicator_selection_7 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.7) + ').npy', allow_pickle=True).tolist()
Indicator_selection_8 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.8) + ').npy', allow_pickle=True).tolist()
Indicator_selection_9 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.9) + ').npy', allow_pickle=True).tolist()
Indicator_selection_10 = np.load('./Result_List/New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(1) + ').npy', allow_pickle=True).tolist()

Result_Indicator_selection_1 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.1) + ').npy', allow_pickle=True).tolist()
Result_Indicator_selection_2 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.2) + ').npy', allow_pickle=True).tolist()
Result_Indicator_selection_3 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.3) + ').npy', allow_pickle=True).tolist()
Result_Indicator_selection_4 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.4) + ').npy', allow_pickle=True).tolist()
Result_Indicator_selection_5 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.5) + ').npy', allow_pickle=True).tolist()
Result_Indicator_selection_6 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.6) + ').npy', allow_pickle=True).tolist()
Result_Indicator_selection_7 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.7) + ').npy', allow_pickle=True).tolist()
Result_Indicator_selection_8 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.8) + ').npy', allow_pickle=True).tolist()
Result_Indicator_selection_9 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(0.9) + ').npy', allow_pickle=True).tolist()
Result_Indicator_selection_10 = np.load('./Result_Data/Result_New_Indicator_selection_(' + data + ')_(' + SNG_mode + ')_order' + str(Order) + '_alpha' + str(1) + ').npy', allow_pickle=True).tolist()


print('Indicator_selection_1:' + str(Indicator_selection_1))
print('Indicator_selection_2:' + str(Indicator_selection_2))
print('Indicator_selection_3:' + str(Indicator_selection_3))
print('Indicator_selection_4:' + str(Indicator_selection_4))
print('Indicator_selection_5:' + str(Indicator_selection_5))
print('Indicator_selection_6:' + str(Indicator_selection_6))
print('Indicator_selection_7:' + str(Indicator_selection_7))
print('Indicator_selection_8:' + str(Indicator_selection_8))
print('Indicator_selection_9:' + str(Indicator_selection_9))
print('Indicator_selection_10:' + str(Indicator_selection_10))


print('Result_Indicator_selection_1' + str(Result_Indicator_selection_1))
print('Result_Indicator_selection_2' + str(Result_Indicator_selection_2))
print('Result_Indicator_selection_3' + str(Result_Indicator_selection_3))
print('Result_Indicator_selection_4' + str(Result_Indicator_selection_4))
print('Result_Indicator_selection_5' + str(Result_Indicator_selection_5))
print('Result_Indicator_selection_6' + str(Result_Indicator_selection_6))
print('Result_Indicator_selection_7' + str(Result_Indicator_selection_7))
print('Result_Indicator_selection_8' + str(Result_Indicator_selection_8))
print('Result_Indicator_selection_9' + str(Result_Indicator_selection_9))
print('Result_Indicator_selection_10' + str(Result_Indicator_selection_10))


size = len(Result_Indicator_selection_1)

# 绘制主图
fig, pic = plt.subplots(1, 1, figsize=(10, 5))
#相同Alpha
# pic.plot(range(1, size + 1), Result_Indicator_selection_1, markersize = 6, markerfacecolor='None', label = 'Order_1', marker = '8', color = '#F7AC53')
# pic.plot(range(1, size + 1), Result_Indicator_selection_2, markersize = 6, markerfacecolor='None', label = 'Order_2', marker = 's', color = '#FFD869')
# pic.plot(range(1, size + 1), Result_Indicator_selection_3, markersize = 6, markerfacecolor='None', label = 'Order_3', marker = '>', color = '#BD7795')
# pic.plot(range(1, size + 1), Result_Indicator_selection_4, markersize = 6, markerfacecolor='None', label = 'Order_4', marker = '^', color = '#7FABD1')
# pic.plot(range(1, size + 1), Result_Indicator_selection_5, markersize = 6, markerfacecolor='None', label = 'Order_5', marker = '<', color = '#91CCC0')
# pic.plot(range(1, size + 1), Result_Indicator_selection_6, markersize = 6, markerfacecolor='None', label = 'Order_6', marker = '*', color = '#B5CE4E')
# pic.plot(range(1, size + 1), Result_Indicator_selection_7, markersize = 6, markerfacecolor='None', label = 'Order_7', marker = 'x', color = '#EC6E66')
# pic.plot(range(1, size + 1), Result_Indicator_selection_8, markersize = 6, markerfacecolor='None', label = 'Order_8', marker = 'h', color = '#B4B4D5')
# pic.plot(range(1, size + 1), Result_Indicator_selection_9, markersize = 6, markerfacecolor='None', label = 'Order_9', marker = 'd', color = '#808080')
# pic.plot(range(1, size + 1), Result_Indicator_selection_10, markersize = 6, markerfacecolor='None', label = 'Order_10', marker = 'p', color = '#FAC7B3')

#相同Order
pic.plot(range(1, size + 1), Result_Indicator_selection_1, markersize = 6, markerfacecolor='None', label = 'Alpha_0.1', marker = '8', color = '#F7AC53')
pic.plot(range(1, size + 1), Result_Indicator_selection_2, markersize = 6, markerfacecolor='None', label = 'Alpha_0.2', marker = 's', color = '#FFD869')
pic.plot(range(1, size + 1), Result_Indicator_selection_3, markersize = 6, markerfacecolor='None', label = 'Alpha_0.3', marker = '>', color = '#BD7795')
pic.plot(range(1, size + 1), Result_Indicator_selection_4, markersize = 6, markerfacecolor='None', label = 'Alpha_0.4', marker = '^', color = '#7FABD1')
pic.plot(range(1, size + 1), Result_Indicator_selection_5, markersize = 6, markerfacecolor='None', label = 'Alpha_0.5', marker = '<', color = '#91CCC0')
pic.plot(range(1, size + 1), Result_Indicator_selection_6, markersize = 6, markerfacecolor='None', label = 'Alpha_0.6', marker = '*', color = '#B5CE4E')
pic.plot(range(1, size + 1), Result_Indicator_selection_7, markersize = 6, markerfacecolor='None', label = 'Alpha_0.7', marker = 'x', color = '#EC6E66')
pic.plot(range(1, size + 1), Result_Indicator_selection_8, markersize = 6, markerfacecolor='None', label = 'Alpha_0.8', marker = 'h', color = '#B4B4D5')
pic.plot(range(1, size + 1), Result_Indicator_selection_9, markersize = 6, markerfacecolor='None', label = 'Alpha_0.9', marker = 'd', color = '#808080')
pic.plot(range(1, size + 1), Result_Indicator_selection_10, markersize = 6, markerfacecolor='None', label = 'Alpha_1', marker = 'p', color = '#FAC7B3')


pic.legend(loc="lower right")
pic.set_xlabel('Compelete Nodess')
pic.set_ylabel('Influence Number')


#展示
#相同Alpha
# plt.title(data + '_Alpha_' + str(Alpha))
# plt.savefig('C://Users//dell//Desktop//CompeteResult_' + data + '_' + SNG_mode + '_alpha_' + str(Alpha) + ').png')

#相同Order
plt.title(data + '_Order_' + str(Order))
plt.savefig('C://Users//dell//Desktop//CompeteResult_' + data + '_' + SNG_mode + '_order_' + str(Order) + ').png')

plt.show()

