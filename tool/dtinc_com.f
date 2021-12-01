      integer idt(5)
      character cdate*12,cc*1,dd*5
      call getarg(1,cdate)
      read(cdate,'(i4,4i2)')(idt(jj),jj=1,5)
      call getarg(2,cc)
      read(cc,'(i1)')ii
      call getarg(3,dd)
      read(dd,'(i5)')jj

c      print *,'idt=',idt
c      print *,'ii =',ii
c      print *,'jj =',jj
      
      call dtinc(idt,idt,ii,jj)
      print '(i4,4i2.2)',(idt(jj),jj=1,5)
      end

      subroutine dtinc(ic,jc,ldm,ipls)
      dimension ic(5),jc(5),mday(12),ip(5)
      data mday/31,28,31,30,31,30,31,31,30,31,30,31/
      do 10 i=1,5
         ip(i)=0
   10 continue
      ip(ldm)=ipls

      do 20 i=1,5
   20 jc(i)=ic(i)

      if(mod(ic(1),4).eq.0) then
         mday(2)=29
        else
         mday(2)=28
      end if

c EN11?E?・?I?i1c
      if(ipls.eq.0) return

c ldm ?I1O?-Ae
      go to (11,12,13,14,15),ldm
      stop '￡I￡A￡I\¨\e!?'

c E￢?IAy，o?I?i1c
   15 continue
      jc(5)=ic(5)+ip(5)
   25 if(jc(5).ge.0 .and. jc(5).lt.60) go to 14
      if(jc(5).ge.60) then
         ip(4)=ip(4)+1
         jc(5)=jc(5)-60
       else
         ip(4)=ip(4)-1
         jc(5)=jc(5)+60
      end if
      go to 25

c ≫t1i?IAy，o?I?i1c
   14 continue
      jc(4)=ic(4)+ip(4)
   30 if(jc(4).ge.0 .and. jc(4).lt.24) go to 13
      if(jc(4).ge.24) then
         ip(3)=ip(3)+1
         jc(4)=jc(4)-24
       else
         ip(3)=ip(3)-1
         jc(4)=jc(4)+24
      end if
      go to 30

c AuEO?IAy，o?I?i1c
   13 continue
      jc(3)=ic(3)+ip(3)
   40 if(jc(3).ge.1 .and. jc(3).le.mday(jc(2))) return
      if(jc(3).gt.mday(jc(2))) then
         jc(3)=jc(3)-mday(jc(2))
         jc(2)=jc(2)+1
         if(jc(2).gt.12) then
            jc(2)=jc(2)-12
            jc(1)=jc(1)+1
            if(mod(jc(1),4).eq.0) then
               mday(2)=29
             else
               mday(2)=28
            end if
         end if
       else
         jc(2)=jc(2)-1
         if(jc(2).lt.1) then
            jc(2)=jc(2)+12
            jc(1)=jc(1)-1
            if(mod(jc(1),4).eq.0) then
               mday(2)=29
             else
               mday(2)=28
            end if
         end if
         jc(3)=jc(3)+mday(jc(2))
      end if
      go to 40

c ・i?IAy，o?I?i1c
   12 continue
      jc(2)=ic(2)+ip(2)
   50 if(jc(2).ge.1 .and. jc(2).le.12) return
      if(jc(2).gt.12) then
         jc(2)=jc(2)-12
         jc(1)=jc(1)+1
       else
         jc(2)=jc(2)+12
         jc(1)=jc(1)-1
      end if
      go to 50

c C￣?IAy，o?I?i1c
   11 continue
      jc(1)=ic(1)+ip(1)
      return
      end
