import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def to_percent(temp, position):
  return '%1.1f'%(100*temp) + '%'



data_1 = 'hept'
data_2= 'phy'
data_3 = 'soc-anybeat'
data_4= 'lastfm_asia'
data_5= 'Email-Enron'
data_6= 'musae-github'




SNG_mode = 'Degree'
# SNG_mode = 'AvgDegree'


x_axis = [0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02]
font = {'size':15}
fig = plt.figure(figsize=(24, 12))


def formatnum_0(x, pos):
  return '%.2f×10$^{2}$' % (x / 1e2)
def formatnum_1(x, pos):
  return '%.1f×10$^{2}$' % (x / 1e2)

def formatnum_2(x, pos):
  return '%.1f×10$^{3}$' % (x / 1e3)

def formatnum_3(x, pos):
  return '%.1f×10$^{4}$' % (x / 1e4)

def formatnum_4(x, pos):
  return '%.2f×10$^{4}$' % (x / 1e4)



AblationResult_1 = np.load('AblationResult_' + data_1 + '_' + SNG_mode + '.npy', allow_pickle=True)
AblationResult_1 = AblationResult_1.tolist()
first = fig.add_subplot(2,3,1)
plt.plot(x_axis,AblationResult_1[0],markersize = 12,markerfacecolor='None',label = 'LI',marker = '8',color = '#B8474D')
plt.plot(x_axis,AblationResult_1[1],markersize = 12,markerfacecolor='None',label = 'LI-1',marker = 'd',color = '#4E6691')
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_2))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_1)) # AvgDegree
plt.tick_params(labelsize=20)
# plt.title("(a)",y = -0.35,fontsize = 16)
plt.text(0.002, min(AblationResult_1[0]+AblationResult_1[1]), "(a)", fontsize=20,
         fontweight=600)
plt.xlabel('对抗节点比例',fontname='FangSong',fontsize=20)
plt.ylabel('传播范围',fontname='FangSong',fontsize=20)






AblationResult_2 = np.load('AblationResult_' + data_2 + '_' + SNG_mode + '.npy', allow_pickle=True)
AblationResult_2 = AblationResult_2.tolist()
second = fig.add_subplot(2,3,2)
plt.plot(x_axis,AblationResult_2[0],markersize = 12,markerfacecolor='None',label = 'LI',marker = '8',color = '#B8474D')
plt.plot(x_axis,AblationResult_2[1],markersize = 12,markerfacecolor='None',label = 'LI-1',marker = 'd',color = '#4E6691')
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_2))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_2)) # AvgDegree
plt.tick_params(labelsize=20)
# plt.title("(b)",y = -0.35,fontsize = 16)
plt.text(0.002, min(AblationResult_2[0]+AblationResult_2[1]), "(b)", fontsize=20,
         fontweight=600)
plt.xlabel('对抗节点比例',fontname='FangSong',fontsize=20)
plt.ylabel('传播范围',fontname='FangSong',fontsize=20)






AblationResult_3 = np.load('AblationResult_' + data_3 + '_' + SNG_mode + '.npy', allow_pickle=True)
AblationResult_3 = AblationResult_3.tolist()
thrird = fig.add_subplot(2,3,3)
plt.plot(x_axis,AblationResult_3[0],markersize = 12,markerfacecolor='None',label = 'LI',marker = '8',color = '#B8474D')
plt.plot(x_axis,AblationResult_3[1],markersize = 12,markerfacecolor='None',label = 'LI-1',marker = 'd',color = '#4E6691')
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_4))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_0)) # AvgDegree
plt.tick_params(labelsize=20)
# plt.title("(c)",y = -0.35,fontsize = 16)
plt.text(0.002, min(AblationResult_3[0]+AblationResult_3[1]), "(c)", fontsize=20,
         fontweight=600)
plt.xlabel('对抗节点比例',fontname='FangSong',fontsize=20)
plt.ylabel('传播范围',fontname='FangSong',fontsize=20)






AblationResult_4 = np.load('AblationResult_' + data_4+ '_' + SNG_mode + '.npy', allow_pickle=True)
AblationResult_4 = AblationResult_4.tolist()
fourth = fig.add_subplot(2,3,4)
plt.plot(x_axis,AblationResult_4[0],markersize = 12,markerfacecolor='None',label = 'LI',marker = '8',color = '#B8474D')
plt.plot(x_axis,AblationResult_4[1],markersize = 12,markerfacecolor='None',label = 'LI-1',marker = 'd',color = '#4E6691')
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_2))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_1)) # AvgDegree
plt.tick_params(labelsize=20)
# plt.title("(d)",y = -0.35,fontsize = 16)
plt.text(0.002, min(AblationResult_4[0]+AblationResult_4[1]), "(d)", fontsize=20,
         fontweight=600)
plt.xlabel('对抗节点比例',fontname='FangSong',fontsize=20)
plt.ylabel('传播范围',fontname='FangSong',fontsize=20)





AblationResult_5 = np.load('AblationResult_' + data_5 + '_' + SNG_mode + '.npy', allow_pickle=True)
AblationResult_5 = AblationResult_5.tolist()
fifth = fig.add_subplot(2,3,5)
plt.plot(x_axis,AblationResult_5[0],markersize = 12,markerfacecolor='None',label = 'LI',marker = '8',color = '#B8474D')
plt.plot(x_axis,AblationResult_5[1],markersize = 12,markerfacecolor='None',label = 'LI-1',marker = 'd',color = '#4E6691')
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_3))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_1)) # AvgDegree
plt.tick_params(labelsize=20)
# plt.title("(e)",y = -0.35,fontsize = 16)
plt.text(0.002, min(AblationResult_5[0]+AblationResult_5[1]), "(e)", fontsize=20,
         fontweight=600)
plt.xlabel('对抗节点比例',fontname='FangSong',fontsize=20)
plt.ylabel('传播范围',fontname='FangSong',fontsize=20)





AblationResult_6 = np.load('AblationResult_' + data_6 + '_' + SNG_mode + '.npy', allow_pickle=True)
AblationResult_6 = AblationResult_6.tolist()
sixth = fig.add_subplot(2,3,6)
plt.plot(x_axis,AblationResult_6[0],markersize = 12,markerfacecolor='None',label = 'LI',marker = '8',color = '#B8474D')
plt.plot(x_axis,AblationResult_6[1],markersize = 12,markerfacecolor='None',label = 'LI-1',marker = 'd',color = '#4E6691')
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_3))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(formatnum_0)) # AvgDegree
plt.tick_params(labelsize=20)
# plt.title("(f)",y = -0.35,fontsize = 16)
plt.text(0.002, min(AblationResult_6[0]+AblationResult_6[1]), "(f)", fontsize=20,
         fontweight=600)
plt.xlabel('对抗节点比例',fontname='FangSong',fontsize=20)
plt.ylabel('传播范围',fontname='FangSong',fontsize=20)



plot, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(plot, labels, loc = 'upper center', ncol=8, framealpha=False, bbox_to_anchor=(0.5, 0.95), prop = {'size':23}) # 图例的位置
plt.subplots_adjust(wspace=0.4,hspace=0.4)
plt.savefig('C://Users//dell//Desktop//AblationResult_' + SNG_mode +'.png')
plt.show()