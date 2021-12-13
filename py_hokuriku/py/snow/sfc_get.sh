#!/bin/bash
# 2021/05/24) 55号機から富山のtsを取得してくる

#=======================================================================
# Set Environment
#=======================================================================
DIR=`dirname $0`
# . ${DIR}/com.conf
# . ${DIR}/prod.conf

# SETMIC
# SETCHOKI

COM=`basename $0`

FTP_MGET(){
 ## 指定ファイルをFTP GETする  FTP_MGET address user_name passwd $DIR
cat << EOF | ftp -nv
  open $1
  user $2 $3
  cd $4
  prompt off
  bin
  mget *.dat
  bye
EOF
}

FTP_GET(){
 ## 指定ファイルをFTP GETする  FTP_MGET address user_name passwd $DIR $FILE
cat << EOF | ftp -nv
  open $1
  user $2 $3
  cd $4
  prompt off
  bin
  get $5
  bye
EOF
}

HOME=/home/ysorimachi/work/hokuriku
TOOL=/home/ysorimachi/tool
PY=$HOME/py
TBL=$HOME/tbl
# alias python='/opt/pyenv/versions/miniconda3-latest/envs/anaconda201910/bin/python'
################################################################################################
# -----------------------------------------
#とにかくデータをftpして保存するような
# 
_MM20=(202004 202005 202006 202007 202008 202009 202010 202011 202012 202101 202102 202103)
for MONTH in ${_MM20[@]};do
#--------
# REMOTE=/media/satdb-1/work/ysorimachi/solasat8now/UNYO_8now_snow/ts/${YY}
LOCAL=/work/ysorimachi/hokuriku/snow_hosei/rad210524/sfc
cd $LOCAL

if [ $MONTH -le 202101 ];then
    #ftp get
    REMOTE=/home/ysorimachi/data/SOKUHOU_210129/${MONTH}/sfc
    for CODE in 47607 47616 ;do
    FILE=sfc_10minh_${MONTH}_${CODE}.csv
    FTP_GET 133.105.83.58 ysorimachi ysorimachi123 $REMOTE $FILE
    nkf -w --overwrite $FILE
    done #code
    
else
    # echo "cp $MONTH"
    TMP=/home/ysorimachi/work/make_SOKUHOU3/ftp/out/${MONTH}/sfc2
    for CODE in 47607 47616 ;do
    cp ${TMP}/sfc_10minh_${MONTH}_${CODE}.csv ./
    nkf -w --overwrite sfc_10minh_${MONTH}_${CODE}.csv
    done #CODE
fi
done
