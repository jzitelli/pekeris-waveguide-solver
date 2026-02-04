MODULE underwater
  ! this module contains data common to both 2d and axisymmetric underwater problems
  USE underwater_layers
  USE user_routines
  USE pml
  IMPLICIT NONE

  DOUBLE PRECISION :: OMEGA, SOURCE_RAD, SOURCE_DEPTH
  DOUBLE COMPLEX :: U_N ! normal velocity on the source boundary
  CHARACTER(LEN=256) :: PARAMS_FILE
  INTEGER, PARAMETER :: PARAMS_FILE_UNIT = 45

CONTAINS

  SUBROUTINE print_underwater_data()
    IMPLICIT NONE

    WRITE(*,*) 'Omega:', OMEGA
    WRITE(*,*) 'Source depth:', SOURCE_DEPTH
    WRITE(*,*) 'Source radius:', SOURCE_RAD
    WRITE(*,*) 'Num layers:', N_LAYERS
    WRITE(*,*) 'Source velocity:', U_N
    CALL print_PML_data
  END SUBROUTINE print_underwater_data

  SUBROUTINE set_omega(omega_new)
    ! set or change omega
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: omega_new

    OMEGA = omega_new
    PML_omega = omega_new
  END SUBROUTINE set_omega

  FUNCTION g_func(x, normal)
    ! compute the neumann data
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: x(2), normal(2)
    DOUBLE COMPLEX :: g_func

    g_func = (0, 1.d0) * v_func(x, normal) * OMEGA
  END FUNCTION g_func

  FUNCTION v_func(x, normal)
    ! compute the velocity (neumann) BC at physical point x with the specified
    ! normal
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: x(2), normal(2)
    DOUBLE COMPLEX :: v_func

    v_func = U_N
  END FUNCTION v_func

  FUNCTION k_func(x, mdle)
    ! compute the wavenumber at physical point x
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: x(2)
    INTEGER, INTENT(IN), OPTIONAL :: mdle
    DOUBLE COMPLEX :: k_func
    
    k_func = CMPLX(OMEGA / c_func(x, mdle), -alpha_func(x), KIND=8)
  END FUNCTION k_func

END MODULE underwater
