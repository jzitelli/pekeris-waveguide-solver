MODULE pekeris_lookup
  IMPLICIT NONE
  TYPE :: table_entry
     DOUBLE PRECISION :: x(2)
     DOUBLE COMPLEX :: H, gradH(2)
  END TYPE table_entry

  TYPE(table_entry), ALLOCATABLE :: table(:)
  INTEGER, PARAMETER :: MAX_ENTRIES = 50000*81
  INTEGER :: NUM_ENTRIES

  CONTAINS

    SUBROUTINE init_lookup
      IMPLICIT NONE
      INTEGER :: i

      ALLOCATE(table(MAX_ENTRIES))
      DO i = 1, MAX_ENTRIES
         table(i)%x(1) = -100.d0
      ENDDO
      NUM_ENTRIES = 0
    END SUBROUTINE init_lookup

    SUBROUTINE get_index(x, index)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN) :: x(2)
      INTEGER :: index, i

      DO i = 1, NUM_ENTRIES
         IF (ABS(table(i)%x(1) - x(1)) < 1.d-9 .AND. &
             ABS(table(i)%x(2) - x(2)) < 1.d-9) THEN
            index = i
            RETURN
         ENDIF
      ENDDO
      index = 0
    END SUBROUTINE get_index

    SUBROUTINE insert_entry(x, H, gradH, full_flag)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN) :: x(2)
      DOUBLE COMPLEX, INTENT(IN) :: H, gradH(2)
      LOGICAL, INTENT(OUT) :: full_flag

      full_flag = .FALSE.
      IF (NUM_ENTRIES < MAX_ENTRIES) THEN
         NUM_ENTRIES = NUM_ENTRIES + 1
         table(NUM_ENTRIES)%x = x
         table(NUM_ENTRIES)%H = H
         table(NUM_ENTRIES)%gradH = gradH

         IF (NUM_ENTRIES .EQ. MAX_ENTRIES) THEN
            WRITE(*,*) 'pekeris_lookup: lookup table full!'
         ENDIF
      ELSE
         full_flag = .TRUE.
      ENDIF
    END SUBROUTINE insert_entry

END MODULE pekeris_lookup
