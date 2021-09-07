
LOCAL=$1

#------
# date : 2021.09.07
## referene --> https://irbank.net/download
#------

# URL_HOME="https://f.irbank.net/files/0000"
cd $LOCAL
rm ./*.csv ./*.xls #init
#年間データ



URL="https://stockdatacenter.com/stockdata/companylist.csv"
wget $URL
FILE=`basename $URL`
nkf -w --overwrite $FILE
sleep 2 #アクセス過多を避けるため

URL_HOME=https://www.jpx.co.jp
URL=${URL_HOME}/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls
wget $URL
FILE=`basename $URL`
nkf -w --overwrite $FILE
sleep 2 #アクセス過多を避けるため
