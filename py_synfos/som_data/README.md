# info ----
* 


# データセット ---
* 再解析データセット　ERA5について
/home/ysorimachi/work/ecmwf/py/utils_ERA5.py にて取得処理
* synfosのデータセット(mnt情報から取得)
* synfosのメッシュセット(mnt情報から取得)

# 解析src(pythonベース) ---
/home/ysorimachi/work/synfos/py/som_data

* SOM クラスタリング
[util_SOM2.py] : class(som)　->　ssim/ed
[util_Data.py] : era5　の再解析データセットのnetCDFかpytorch用の読み込みobjectの実装
[c01_som_cluster.py] : 解析program 

* fitting
[fit_cluster_rad1.py]: 統合予測+　残差補正
[fit_cluster_rad2.py]: 直接予測
