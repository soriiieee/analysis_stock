# init
# 2021.03.01 sorimachi - yuichi
# 2021.04.01
# 2021.06.16 ax.set_aspect('equal','datalim')の書き込み

* python -v
Python 3.7.9 (default, Aug 31 2020, 12:42:55) 
[GCC 7.3.0] :: Anaconda, Inc. on linux

# tips
* lsなど重い処理を実行して止めたいときのid検索(2021.03.04)
`ps aux | awk '$1 ~ /ysori/{print $0}'`

* 容量の確認
`du -sh ./work/* `
`du -sh /home/ysorimachi/work/*`


# -------------------------------------------------------------------
# matplotlib関連のtips
# --------------------------------------------------------
* plt.rcParams['figure.figsize'] = [6.4,4.0]  # 図の縦横のサイズ([横(inch),縦(inch)])
* lauout -> https://qiita.com/aurueps/items/d04a3bb127e2d6e6c21b
* ref -> https://matplotlib.org/stable/api/matplotlib_configuration_api.html?highlight=rcparams#matplotlib.RcParams
fontsize=14
plt.rcParams['xtick.labelsize'] = fontsize        # 目盛りのフォントサイズ
plt.rcParams['ytick.labelsize'] = fontsize        # 目盛りのフォントサイズ
plt.rcParams['figure.subplot.wspace'] = 0.20 # 図が複数枚ある時の左右との余白
plt.rcParams['figure.subplot.hspace'] = 0.20 # 図が複数枚ある時の上下との余白
plt.rcParams['font.size'] = fontsize
plt.rcParams['lines.linewidth'] = 5
plt.rcParams['text.usetex'] = True
r'$\displaystyle'

# matplot で日本語
 <!-- rm ~/.cache/matplotlib/fontlist-v330.json -->
 <!-- rm ~/.cache/matplotlib/fontlist-v310.json -->
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


# matplot でlatex
~/.cache/matplotlib/ にある*jsonを削除
plt.rcParams['text.usetex'] = True
r"地点平均日射量[KW/$m^2$]"

# ax の縦横比を調整するやり方
<!-- https://cattech-lab.com/science-tools/simulation-lecture-2-5/ -->
<!-- ax.set_aspect('equal', 'datalim') -->

# 軸を消す
ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
               
# 凡例(https://qiita.com/matsui-k20xx/items/291400ed56a39ed63462)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.legend(ncol=3)で3行設定

# グラフとの間隔をあける場合
plt.subplots_adjust(wspace=0.4, hspace=0.5)

# x軸ラベルの間隔調整
    ax[i].set_xlim(0,len(df)) #x軸の限界値を設定
    st, ed = ax[i].get_xlim()
    ax[i].xaxis.set_ticks(np.arange(int(st), int(ed),step))
    ax[i].set_xticklabels(_t[::step], fontsize=12)

# axの使わないグラフを消去する
for i in range(last_day,5*7):
    ax[i].set_visible(False)

# 自作のカラーマップ()
<!-- cmap = plt.get_cmap("tab10") # ココがポイント -->
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]

# -------------------------------------------------
# pandas 関連
# -------------------------------------------------
# apply(lambda x: )でgoupbyで日別のエラーを計算するやり方
#local functin
def err_func(x):
    _err = []
    if err_name=="rmse":
        _err.append(rmse(x["ame_obs"], x["synfos"]))
        _err.append(rmse(x["ame_obs"], x["ecmwf"]))
    if err_name=="me":
        _err.append(me(x["ame_obs"], x["synfos"]))
        _err.append(me(x["ame_obs"], x["ecmwf"]))
    if err_name=="%rmse":
        _err.append(rmse(x["synfos"]/x["ame_obs"], x["1"]))
        _err.append(rmse(x["ecmwf"]/x["ame_obs"], x["1"]))
    return pd.Series(_err, index=[f"{err_name}_s",f"{err_name}_e"])

_df = []
for month in _month:
tmp = df[df["month"]==month]
tmp["1"] = 1.
if not tmp.empty:
    tmp2 = tmp.groupby("dd").apply(err_func)


# bar に表示する
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
