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
MONTH=$1
CODE=$2
LOCAL=$3


REMOTE=/home/ysorimachi/data/SOKUHOU_210129/${MONTH}/amd

cd $LOCAL
FILE=amd_10minh_${MONTH}_${CODE}.csv
FTP_GET 133.105.83.58 ysorimachi ysorimachi123 $REMOTE $FILE
nkf -w --overwrite $FILE
