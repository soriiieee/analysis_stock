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
REMOTE=/media/satdb-1/work/ysorimachi/solasat8now/UNYO_8now_snow/case_hokuriku_2021/ts/$MONTH
LOCAL=/work/ysorimachi/hokuriku/snow_hosei/rad210524/8now0

#--------
# REMOTE=/media/satdb-1/work/ichizawa/data/obs/ghi_01min/obs_01min_${YY}
# LOCAL=/work/ysorimachi/hokuriku/snow_hosei/ame1min/${YY}
[ ! -e $LOCAL ] && mkdir -p $LOCAL
cd $LOCAL
# rm ./*.dat
for PP in 15 18;do
FILE=ofile_${MONTH}_cpnt${PP}.dat
FTP_GET 133.105.83.55 ysorimachi ysorimachi123 $REMOTE $FILE
# exit
done #PP
done
