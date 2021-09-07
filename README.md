# details 
* author :sorimachi yuichi
* date : 2021.09.03 (init)
* date : 2021.09.08 (git連携の更新作業&日本株データリスト取得の作業)
* Git(remote) : git@github.com:soriiieee/analysis_stock.git
# init 
`cd /Users/soriiieee/work2/stock`
`source stockenv/bin/activate`

<!-- 一時的に -->
<!-- python -m venv stock_env --> 2021.09.03  作成済
<!-- pip install -r requirements.txt --> 021.09.08  更新済

# git 関連
* ssh接続関連　-> https://qiita.com/shunsa10/items/e43564cf48f84b95455b
* ssh接続の際の鍵についての情報設定(pathphrazeの有無に関し) ->https://qiita.com/hnishi/items/5dec4c7fca9b5121430f
2) sshしたいサーバー側に公開鍵（public key）を覚えさせる。
$ ssh-copy-id -i ~/.ssh/id_rsa.pub git@github.com:soriiieee/analysis_stock.git:
ユーザーのパスワードが要求されるので入力すると、
* 公開鍵のgit設定　-> https://qiita.com/shizuma/items/2b2f873a0034839e47ce
* 接続確認　　`ssh -T git@github.com` -> OK 2021.09.08

