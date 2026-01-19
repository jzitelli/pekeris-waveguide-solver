C     Simple XERMSG stub for SLATEC routines
C     Just prints warning messages without stopping
      SUBROUTINE XERMSG(LIBRAR, SUBROU, MESSG, NERR, LEVEL)
      CHARACTER*(*) LIBRAR, SUBROU, MESSG
      INTEGER NERR, LEVEL
      IF (LEVEL .GE. 2) THEN
         WRITE(*,*) 'ERROR in ', SUBROU, ': ', MESSG
      ELSE IF (LEVEL .GE. 1) THEN
         WRITE(*,*) 'WARNING in ', SUBROU, ': ', MESSG
      ENDIF
      RETURN
      END
