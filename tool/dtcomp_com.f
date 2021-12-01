      integer idt(5),jdt(5)
      character cdate*12,cc*1,dd*5
      call getarg(1,cdate)
      read(cdate,'(i4,4i2)')(idt(jj),jj=1,5)
      call getarg(2,cdate)
      read(cdate,'(i4,4i2)')(jdt(jj),jj=1,5)
      call getarg(3,cc)
      read(cc,'(i1)')ii

c      print *,'idt=',idt
c      print *,'ii =',ii
c      print *,'jj =',jj
      
      call dtcomp(idt,jdt,ii,jj)
      print *,jj
      end
      subroutine dtcomp(nd,md,jc,ia)
      dimension nd(*),md(*),mmd(13,2)
      data mmd/0,31,59,90,120,151,181,212,243,273,304,334,365
     +        ,0,31,60,91,121,152,182,213,244,274,305,335,366/
c nd(5):年,月,日,時,分
c md(5):年,月,日,時,分
c nd(5):年,月,日,時,分
c jc   :比較する時間単位(1.年,2.月,3.日,4.時,5.分)
c ia   :時間差
c 1994/02/25 3年以上にまたがる処理も可能とした
c ただし、iaを

      if(jc.lt.1.or.jc.gt.5) stop'date error(sub.dtcomp:jc)'

      ia=0
      do 10 i=1,jc
      if(nd(i).ne.md(i)) goto 20
   10 continue
      return

   20 continue
      j1=nd(1)
      j2=md(1)
      i1=nd(2)
      i2=md(2)
      if(i1.lt.1.or.i1.gt.12) stop'date error(sub.dtcomp:nd(2))'
      if(i2.lt.1.or.i2.gt.12) stop'date error(sub.dtcomp:md(2))'
      l1=1
      l2=1
      if(mod(nd(1),4).eq.0) l1=2
      if(mod(md(1),4).eq.0) l2=2


c 複数年にまたがる場合の処理
      if(jc.le.2) then
         ia=j1-j2
      else
      if(j1.gt.j2) then
        do 30 j=j2,j1-1
        if(mod(j,4).eq.0) then
          ia=ia+mmd(13,2)
        else
          ia=ia+mmd(13,1)
        endif
c       write(6,*)' j1>j2  j:',j,' ia:',ia
   30   continue
      else if(j1.lt.j2) then
        do 40 j=j1,j2-1
        if(mod(j,4).eq.0) then
          ia=ia-mmd(13,2)
        else
          ia=ia-mmd(13,1)
        endif
c       write(6,*)' j1<j2  j:',j,' ia:',ia
   40   continue
      endif
      endif
c     write(6,*)' ia:',ia


C 月単位の計算
      if(jc.eq.2) then
         ia=ia*12+nd(2) - md(2)
      ENDIF

c 日単位の計算
      ia=ia+mmd(i1,l1)+nd(3) - mmd(i2,l2)-md(3)

c 時間単位の計算
      if(jc.ge.4) then
        ia=ia*24+nd(4)-md(4)
c 分単位の計算
        if(jc.eq.5) then
          ia=ia*60+(nd(5)-md(5))
        endif
      endif

      return
      end
