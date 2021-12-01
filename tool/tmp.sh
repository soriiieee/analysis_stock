#!/bin/bash
# LFM

cat /home/ysorimachi/work/synfos/tbl/list_amedas.csv | awk -F[,] 'NR>1{printf"%5s%30s%30s%30s%10.4f%10.4f\n" ,$1,$2,$3,$4,$5,$6}' > /home/ysorimachi/work/synfos/tbl/list_amedas.tbl