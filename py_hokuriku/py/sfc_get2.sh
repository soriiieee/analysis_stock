#!/bin/bash
# 2021/05/24) 55号機から富山のtsを取得してくる
# 2021/07/26) 55号機から富山のtsを取得してくる

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



MONTH=$1
CODE=$2
LOCAL=$3

[ $MONTH -lt 202004 ] && {
  IP=133.105.83.57
  USER=share
  PASSWD=share
  REMOTE=/home/share/usbdisk/DATA/OBS/01-JMA/SOKUHOU/${MONTH:0:4}_${MONTH:4:2}/surface/10min_h/${MONTH}
} || {
  IP=133.105.83.58
  USER=ysorimachi
  PASSWD=ysorimachi123
  REMOTE=/home/ysorimachi/data/SOKUHOU_210129/${MONTH}/sfc
}

#common

cd $LOCAL
FILE=sfc_10minh_${MONTH}_${CODE}.csv
FTP_GET $IP $USER $PASSWD $REMOTE $FILE
nkf -w --overwrite $FILE