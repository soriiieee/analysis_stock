#!/bin/bash
# 2021/05/24) 55号機から富山のtsを取得してくる
# 2021/09/01) 55号機から富山のtsを取得してくる

#=======================================================================
# Set Environment
#=======================================================================
DIR=`dirname $0`
. /home/ysorimachi/work/hokuriku/bin/com.conf
. /home/ysorimachi/work/hokuriku/bin/prod.conf

SETMIC
SETCHOKI

COM=`basename $0`

INI_U=$1
LOCAL=$2


cd $LOCAL
FILE=111095-000000-0000-${INI_U}00.bin
GET_PRODUCT "RETRY" $FILE

# 参考にしたサイト
# https://qiita.com/m-taque/items/31e9268129ab64198916
wgrib2 $FILE -netcdf ${INI_U}.nc

