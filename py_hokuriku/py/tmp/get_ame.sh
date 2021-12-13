


LOCAL=$1
URL_HOME=https://www.jma.go.jp/jma/kishou/know/amedas
UPDATE_DD=20210902

cd $LOCAL
#url - snow 
URL=${URL_HOME}/snow_master_${UPDATE_DD}.zip
wget $URL
FILE=`basename $URL`
unzip $FILE
rm -rf $FILE

#url - snow 
URL=${URL_HOME}/ame_master_${UPDATE_DD}.zip
wget $URL
FILE=`basename $URL`
unzip $FILE
rm -rf $FILE

nkf -w --overwrite *.csv