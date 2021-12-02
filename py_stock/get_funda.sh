
LOCAL=$1
YY=$2

DATADIR=$LOCAL/$YY
[ ! -e $DATADIR ] && mkdir -p $DATADIR

#------
# date : 2021.09.07
## referene --> https://irbank.net/download
#------
[ $YY == "2021" ] && {
  URL_HOME="https://f.irbank.net/files/0000"
  _FILE=("fy-balance-sheet.csv" "fy-cash-flow-statement.csv" "fy-profit-and-loss.csv" "fy-stock-dividend.csv" "qq-yoy-net-sales.csv" "qq-yoy-operating-income.csv" "qq-yoy-ordinary-income.csv" "qq-yoy-profit-loss.csv")
} || {
  YY2=${YY:2:2}
  URL_HOME="https://f.irbank.net/files/00${YY2}"
  # href="https://f.irbank.net/files/0018/fy-balance-sheet.csv"
  _FILE=("fy-balance-sheet.csv" "fy-cash-flow-statement.csv" "fy-profit-and-loss.csv" "fy-stock-dividend.csv")
}

# echo ${_FILE[@]}
# echo $URL_HOME
# exit
# href="https://f.irbank.net/files/0020/fy-balance-sheet.csv"
cd $DATADIR
rm ./*.csv #init
#年間データ
for FILE in ${_FILE[@]};do
URL=${URL_HOME}/$FILE
wget $URL
nkf -w --overwrite $FILE
sleep 2 #アクセス過多を避けるため
done
echo "[END] Fundamental data ..."

