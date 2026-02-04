MODULE pml
  IMPLICIT NONE

  DOUBLE PRECISION :: PML_left, PML_right, PML_top, PML_bottom, PML_omega, PML_width
  DOUBLE PRECISION, PARAMETER :: PML_const = 1500.d0 * 500.d0 !150.d0

CONTAINS

SUBROUTINE PML_coordinates(x, z, dz_dx)
  IMPLICIT NONE
  DOUBLE PRECISION, INTENT(IN) :: x(2)
  DOUBLE COMPLEX, INTENT(OUT) :: z(2), dz_dx(2)
  
  DOUBLE PRECISION :: xi_1, xi_2, temp
  
  IF (x(1) > PML_right) THEN
     xi_1 = (x(1) - PML_right) / PML_width
     temp = PML_const / PML_omega * xi_1**3 / PML_width
     dz_dx(1) = 1.0 + temp - (0,1)*temp
     temp = temp * PML_width / 4.0 * xi_1
     z(1) = x(1) + temp - (0,1)*temp
  ELSEIF (x(1) < PML_left) THEN
     xi_1 = (PML_left - x(1)) / PML_width
     temp = PML_const / PML_omega * xi_1**3 / PML_width
     dz_dx(1) = 1.0 + temp - (0,1)*temp
     temp = temp * PML_width / 4.0 * xi_1
     z(1) = x(1) - temp + (0,1)*temp
  ELSE
     z(1) = x(1)
     dz_dx(1) = 1.0
  ENDIF
  IF (x(2) > PML_top) THEN
     xi_2 = (x(2) - PML_top) / PML_width
     temp = PML_const / PML_omega * xi_2**3 / PML_width
     dz_dx(2) = 1.0 + temp - (0,1)*temp
     temp = temp * PML_width / 4.0 * xi_2
     z(2) = x(2) + temp - (0,1)*temp
  ELSEIF (x(2) < PML_bottom) THEN
     xi_2 = (PML_bottom - x(2)) / PML_width
     temp = PML_const / PML_omega * xi_2**3 / PML_width
     dz_dx(2) = 1.0 + temp - (0,1)*temp
     temp = temp * PML_width / 4.0 * xi_2
     z(2) = x(2) - temp + (0,1)*temp
  ELSE
     z(2) = x(2)
     dz_dx(2) = 1.0
  ENDIF
END SUBROUTINE PML_coordinates

  ! SUBROUTINE PML_coordinates(x, z, dz_dx)
  !   IMPLICIT NONE
  !   DOUBLE PRECISION, INTENT(IN) :: x(2)
  !   DOUBLE COMPLEX, INTENT(OUT) :: z(2), dz_dx(2)

  !   DOUBLE PRECISION :: temp

  !   IF (x(1) > PML_right) THEN
  !      !temp = 40.d0 / PML_omega * (x(1) - PML_right)**3
  !      temp = PML_const / PML_omega * (x(1) - PML_right)**3
  !      dz_dx(1) = 1.0 + temp - (0,1)*temp
  !      temp = temp / 4.0 * (x(1) - PML_right)
  !      !temp = temp / 3.0 * (x(1) - PML_right)
  !      z(1) = x(1) + temp - (0,1)*temp
  !   ELSEIF (x(1) < PML_left) THEN
  !      !temp = 40.d0 / PML_omega * (PML_left - x(1))**3
  !      temp = PML_const / PML_omega * (PML_left - x(1))**3
  !      dz_dx(1) = 1.0 + temp - (0,1)*temp
  !      temp = temp / 4.0 * (PML_left - x(1))
  !      !temp = temp / 3.0 * (PML_left - x(1))
  !      z(1) = x(1) - temp + (0,1)*temp
  !   ELSE
  !      z(1) = x(1)
  !      dz_dx(1) = 1.0
  !   ENDIF

  !   IF (x(2) > PML_top) THEN
  !      !temp = 40.d0 / PML_omega * (x(2) - PML_top)**3
  !      temp = PML_const / PML_omega * (x(2) - PML_top)**3
  !      dz_dx(2) = 1.0 + temp - (0,1)*temp
  !      temp = temp / 4.0 * (x(2) - PML_top)
  !      !temp = temp / 3.0 * (x(2) - PML_top)
  !      z(2) = x(2) + temp - (0,1)*temp
  !   ELSEIF (x(2) < PML_bottom) THEN
  !      !temp = 40.d0 / PML_omega * (PML_bottom - x(2))**3
  !      temp = PML_const / PML_omega * (PML_bottom - x(2))**3
  !      dz_dx(2) = 1.0 + temp - (0,1)*temp
  !      temp = temp / 4.0 * (PML_bottom - x(2))
  !      !temp = temp / 3.0 * (PML_bottom - x(2))
  !      z(2) = x(2) - temp + (0,1)*temp
  !   ELSE
  !      z(2) = x(2)
  !      dz_dx(2) = 1.0
  !   ENDIF
  ! END SUBROUTINE PML_coordinates

  SUBROUTINE print_PML_data
    WRITE(*,*) 'PML_left:', PML_left
    WRITE(*,*) 'PML_right:', PML_right
    WRITE(*,*) 'PML_top:', PML_top
    WRITE(*,*) 'PML_bottom:', PML_bottom
    WRITE(*,*) 'PML_width:', PML_width
    WRITE(*,*) 'PML_omega:', PML_omega
  END SUBROUTINE print_PML_data

END MODULE pml
