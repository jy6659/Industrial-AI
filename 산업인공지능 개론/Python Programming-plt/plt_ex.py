a = [242, 256, 237, 223, 263, 81, 46]

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_name = fort_manager.FontProperties(fname="c:/Window/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

x_data = ['MON', 'TUE', 'WED', 'THR', 'FRI', 'SAT', 'SUN']

plt.title("일주일간 유동 인구수 데이터", fontsize = 16)
plt.xlabel("요일", fontsize=12)
plt.ylabel("유동 인구수", fontsize=12)

plt.scatter(x_data, a)
plt.plot(x_data, a)
plt.show()
