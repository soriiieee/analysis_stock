
LOCAL=$1
rm ./*.csv #init



#------
# date : 2021.09.07
## referene --> https://irbank.net/download
#------

URL_HOME="https://f.irbank.net/files/0000"
cd $LOCAL
#年間データ
for FILE in "fy-balance-sheet.csv" "fy-cash-flow-statement.csv" "fy-profit-and-loss.csv" "fy-stock-dividend.csv";do
URL=${URL_HOME}/$FILE
wget $URL
nkf -w --overwrite $FILE
sleep 2 #アクセス過多を避けるため
done
echo "[END-001] YEAR Fundamental data ..."

#四半期データ
for FILE in "qq-yoy-net-sales.csv" "qq-yoy-operating-income.csv" "qq-yoy-ordinary-income.csv" "qq-yoy-profit-loss.csv";do
URL=${URL_HOME}/$FILE
wget $URL
# nkf -w --overwrite $FILE
done
echo "[END-002] 4 TERM Fundamental data ..."
