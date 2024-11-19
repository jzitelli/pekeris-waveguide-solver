MODULE pekeris
  ! pekeris exact solution module
  IMPLICIT NONE

  DOUBLE PRECISION, ALLOCATABLE :: eigenvals(:), k1_z(:), k2_z(:), k_r(:), A_sqrd(:)
  DOUBLE PRECISION :: k1, k2, k1_sqrd, k2_sqrd, branch_start, rho1, rho2, c1, c2, rho2_rho1, &
       height, z_s
  INTEGER :: n_modes
  LOGICAL :: pekeris_initialized = .FALSE.
  DOUBLE PRECISION :: r, z
  ! quadpack:
  INTEGER, PARAMETER :: IWORK_LENGTH = 10000, WORK_LENGTH = 80000
  INTEGER :: IWORK(IWORK_LENGTH)
  DOUBLE PRECISION :: WORK(WORK_LENGTH)
  DOUBLE PRECISION, PARAMETER :: quadrature_error = 1.d-7
  LOGICAL :: USE_DISCRETE_MODES_ONLY = .FALSE.

  CONTAINS

  SUBROUTINE init_pekeris
    USE underwater
    IMPLICIT NONE

    IF (pekeris_initialized .EQV. .TRUE.) RETURN
    IF (N_LAYERS < 2) THEN
       WRITE(*,*) 'init_pekeris: 2 layers needed!'
       RETURN
    ENDIF

    k1 = OMEGA / LAYER_C(1)
    k2 = OMEGA / LAYER_C(2)
    k1_sqrd = k1**2
    k2_sqrd = k2**2
    c1 = LAYER_C(1)
    c2 = LAYER_C(2)
    rho1 = LAYER_RHO(1)
    rho2 = LAYER_RHO(2)
    
    branch_start = SQRT(k1_sqrd - k2_sqrd)
    rho2_rho1 = rho2 / rho1
    height = -LAYER_BOTTOM(1)
    z_s = SOURCE_DEPTH

    CALL setPekerisParams(OMEGA, rho1, rho2, z_s, height, c1, c2, n_modes)

    ALLOCATE(eigenvals(n_modes))
    CALL getEigenvalues(eigenvals)
    ALLOCATE(k1_z(n_modes))
    ALLOCATE(k2_z(n_modes))
    ALLOCATE(k_r(n_modes))
    ALLOCATE(A_sqrd(n_modes))

    k1_z(1:n_modes) = SQRT(k1_sqrd - eigenvals(1:n_modes))
    k2_z(1:n_modes) = SQRT(eigenvals(1:n_modes) - k2_sqrd)
    k_r(1:n_modes) = SQRT(eigenvals(1:n_modes))
    A_sqrd(1:n_modes) = 2.d0 / &
         (1/rho1 * (height - SIN(2*k1_z(1:n_modes)*height)/(2*k1_z(1:n_modes))) + &
          1/rho2 * SIN(k1_z(1:n_modes)*height)**2 / k2_z(1:n_modes) )

    pekeris_initialized = .TRUE.
    CALL print_pekeris
  END SUBROUTINE init_pekeris

  SUBROUTINE print_pekeris
    IMPLICIT NONE
    WRITE(*,*) 'pekeris: n_modes =', n_modes
  END SUBROUTINE print_pekeris

  ! quadrature integrand functions:
  FUNCTION real_integrand(x)
    IMPLICIT NONE
    DOUBLE PRECISION :: real_integrand
    DOUBLE PRECISION, INTENT(IN) :: x
    DOUBLE PRECISION, EXTERNAL :: DBESJ0
    DOUBLE PRECISION :: temp, temp2, temp3, fk_r, fk2_z

    temp = k1_sqrd - x**2
    fk_r = SQRT(temp)
    fk2_z = SQRT(k2_sqrd - temp)
    temp = SIN(x*height)**2
    temp2 = x/fk2_z
    temp3 = (rho2_rho1*temp2)**2

    real_integrand = DBESJ0(fk_r*r) * temp2 * SIN(x*z_s) * &
         SIN(x*z)/(temp3 + (1-temp3)*temp)
  END FUNCTION real_integrand

  FUNCTION imag_integrand(x)
    IMPLICIT NONE
    DOUBLE PRECISION :: imag_integrand
    DOUBLE PRECISION, INTENT(IN) :: x
    DOUBLE PRECISION, EXTERNAL :: DBESY0
    DOUBLE PRECISION :: temp, temp2, temp3, fk_r, fk2_z

    temp = k1_sqrd - x**2
    fk_r = SQRT(temp)
    fk2_z = SQRT(k2_sqrd - temp)
    temp = SIN(x*height)**2
    temp2 = x/fk2_z
    temp3 = (rho2_rho1*temp2)**2

    imag_integrand = DBESY0(fk_r*r) * temp2 * SIN(x*z_s) * &
         SIN(x*z)/(temp3 + (1-temp3)*temp)
  END FUNCTION imag_integrand

  FUNCTION real_integrand_dr(x)
    IMPLICIT NONE
    DOUBLE PRECISION :: real_integrand_dr
    DOUBLE PRECISION, INTENT(IN) :: x
    DOUBLE PRECISION, EXTERNAL :: DBESJ1
    DOUBLE PRECISION :: temp, temp2, temp3, fk_r, fk2_z

    temp = k1_sqrd - x**2
    fk_r = SQRT(temp)
    fk2_z = SQRT(k2_sqrd - temp)
    temp = SIN(x*height)**2
    temp2 = x/fk2_z
    temp3 = (rho2_rho1*temp2)**2

    real_integrand_dr = -fk_r * DBESJ1(fk_r*r) * temp2 * SIN(x*z_s) * &
         SIN(x*z)/(temp3 + (1-temp3)*temp)
  END FUNCTION real_integrand_dr

  FUNCTION imag_integrand_dr(x)
    IMPLICIT NONE
    DOUBLE PRECISION :: imag_integrand_dr
    DOUBLE PRECISION, INTENT(IN) :: x
    DOUBLE PRECISION, EXTERNAL :: DBESY1
    DOUBLE PRECISION :: temp, temp2, temp3, fk_r, fk2_z

    temp = k1_sqrd - x**2
    fk_r = SQRT(temp)
    fk2_z = SQRT(k2_sqrd - temp)
    temp = SIN(x*height)**2
    temp2 = x/fk2_z
    temp3 = (rho2_rho1*temp2)**2

    imag_integrand_dr = -fk_r * DBESY1(fk_r*r) * temp2 * SIN(x*z_s) * &
         SIN(x*z)/(temp3 + (1-temp3)*temp)
  END FUNCTION imag_integrand_dr

  FUNCTION real_integrand_dz(x)
    IMPLICIT NONE
    DOUBLE PRECISION :: real_integrand_dz
    DOUBLE PRECISION, INTENT(IN) :: x
    DOUBLE PRECISION, EXTERNAL :: DBESJ0
    DOUBLE PRECISION :: temp, temp2, temp3, fk_r, fk2_z

    temp = k1_sqrd - x**2
    fk_r = SQRT(temp)
    fk2_z = SQRT(k2_sqrd - temp)
    temp = SIN(x*height)**2
    temp2 = x/fk2_z
    temp3 = (rho2_rho1*temp2)**2

    real_integrand_dz = x * DBESJ0(fk_r*r) * temp2 * SIN(x*z_s) * &
         COS(x*z)/(temp3 + (1-temp3)*temp)
  END FUNCTION real_integrand_dz

  FUNCTION imag_integrand_dz(x)
    IMPLICIT NONE
    DOUBLE PRECISION :: imag_integrand_dz
    DOUBLE PRECISION, INTENT(IN) :: x
    DOUBLE PRECISION, EXTERNAL :: DBESY0
    DOUBLE PRECISION :: temp, temp2, temp3, fk_r, fk2_z

    temp = k1_sqrd - x**2
    fk_r = SQRT(temp)
    fk2_z = SQRT(k2_sqrd - temp)
    temp = SIN(x*height)**2
    temp2 = x/fk2_z
    temp3 = (rho2_rho1*temp2)**2

    imag_integrand_dz = x * DBESY0(fk_r*r) * temp2 * SIN(x*z_s) * &
         COS(x*z)/(temp3 + (1-temp3)*temp)
  END FUNCTION imag_integrand_dz

  FUNCTION imag_integrand_indef(x)
    IMPLICIT NONE
    DOUBLE PRECISION :: imag_integrand_indef
    DOUBLE PRECISION, INTENT(IN) :: x
    DOUBLE PRECISION :: temp, temp2, fk_r, fk2_z
    DOUBLE PRECISION, EXTERNAL :: DBESK0
    DOUBLE PRECISION, PARAMETER :: HALFPI = 2/ACOS(-1.d0)

    temp = k1_sqrd - x**2
    fk_r = SQRT(-temp)
!    IF (fk_r*r > 100) THEN
!       imag_integrand_indef = 0
!       RETURN
!    ENDIF
    fk2_z = SQRT(k2_sqrd - temp)
    temp = SIN(x*height)**2
    temp2 = (rho2_rho1*x/fk2_z)**2

    imag_integrand_indef = -HALFPI * DBESK0(fk_r*r) * x / fk2_z * SIN(x*z_s) * &
         SIN(x*z)/(temp2 + (1-temp2)*temp)
  END FUNCTION imag_integrand_indef

  FUNCTION imag_integrand_dr_indef(x)
    IMPLICIT NONE
    DOUBLE PRECISION :: imag_integrand_dr_indef
    DOUBLE PRECISION, INTENT(IN) :: x
    DOUBLE PRECISION :: temp, temp2, fk_r, fk2_z
    DOUBLE PRECISION, EXTERNAL :: DBESK1
    DOUBLE PRECISION, PARAMETER :: HALFPI = 2/ACOS(-1.d0)

    temp = k1_sqrd - x**2
    fk_r = SQRT(-temp)
!    IF (fk_r*r > 100) THEN
!       imag_integrand_dr_indef = 0
!       RETURN
!    ENDIF
    fk2_z = SQRT(k2_sqrd - temp)
    temp = SIN(x*height)**2
    temp2 = (rho2_rho1*x/fk2_z)**2

    imag_integrand_dr_indef = HALFPI * fk_r * DBESK1(fk_r*r) * x / fk2_z * SIN(x*z_s) * &
         SIN(x*z)/(temp2 + (1-temp2)*temp)
  END FUNCTION imag_integrand_dr_indef

  FUNCTION imag_integrand_dz_indef(x)
    IMPLICIT NONE
    DOUBLE PRECISION :: imag_integrand_dz_indef
    DOUBLE PRECISION, INTENT(IN) :: x
    DOUBLE PRECISION :: temp, temp2, fk_r, fk2_z
    DOUBLE PRECISION, EXTERNAL :: DBESK0
    DOUBLE PRECISION, PARAMETER :: HALFPI = 2/ACOS(-1.d0)

    temp = k1_sqrd - x**2
    fk_r = SQRT(-temp)
!    IF (fk_r*r > 100) THEN
!       imag_integrand_dz_indef = 0
!       RETURN
!    ENDIF
    fk2_z = SQRT(k2_sqrd - temp)
    temp = SIN(x*height)**2
    temp2 = (rho2_rho1*x/fk2_z)**2

    imag_integrand_dz_indef = -HALFPI * DBESK0(fk_r*r) * x / fk2_z * SIN(x*z_s) * &
         x * COS(x*z)/(temp2 + (1-temp2)*temp)
  END FUNCTION imag_integrand_dz_indef
END MODULE pekeris


SUBROUTINE exact_pekeris(x, H, gradH)
  USE pekeris
  USE underwater
  USE pekeris_lookup
  IMPLICIT NONE
  DOUBLE PRECISION, INTENT(IN) :: x(2)
  DOUBLE COMPLEX, INTENT(OUT) :: H, gradH(2)
  DOUBLE PRECISION, EXTERNAL :: DBESJ0, DBESJ1, DBESY0, DBESY1
  DOUBLE PRECISION, PARAMETER :: PI = ACOS(-1.d0)
  INTEGER :: n, neval, ierr, ilast
  DOUBLE PRECISION :: abs_error, quadrature_result
  DOUBLE COMPLEX :: integral
  DOUBLE PRECISION :: cutoff
  INTEGER :: index
  LOGICAL :: full_flag

  IF (pekeris_initialized .EQV. .FALSE.) THEN
     CALL init_pekeris
     CALL init_lookup
  ENDIF

  H = 0
  gradH = 0

  IF (x(1) > pml_right) RETURN
  IF (x(2) < pml_bottom) RETURN

  index = 0
  CALL get_index(x, index)
  IF (index /= 0) THEN
     H = table(index)%H
     gradH = table(index)%gradH
     RETURN
  ENDIF

  r = x(1)
  z = -x(2)

  IF (r < 1.d0) THEN
     CALL pekeris_moi(x, 512, H, gradH)
     RETURN
  ENDIF

  ! trapped modes:
  IF (z <= height) THEN
     DO n = 1, n_modes
        H = H + A_sqrd(n) * SIN(k1_z(n)*SOURCE_DEPTH) * SIN(k1_z(n)*z) * &
             CMPLX(DBESJ0(k_r(n)*r), DBESY0(k_r(n)*r), KIND=8)

        gradH(1) = gradH(1) - k_r(n) * A_sqrd(n) * SIN(k1_z(n)*SOURCE_DEPTH) * SIN(k1_z(n)*z) * &
             CMPLX(DBESJ1(k_r(n)*r), DBESY1(k_r(n)*r), KIND=8)

        gradH(2) = gradH(2) + k1_z(n) * A_sqrd(n) * SIN(k1_z(n)*SOURCE_DEPTH) * COS(k1_z(n)*z) * &
             CMPLX(DBESJ0(k_r(n)*r), DBESY0(k_r(n)*r), KIND=8)
     ENDDO
  ELSE
     DO n = 1, n_modes
        H = H + A_sqrd(n) * SIN(k1_z(n)*SOURCE_DEPTH) * SIN(k1_z(n)*height) * &
             EXP(k2_z(n)*(height-z)) * &
             CMPLX(DBESJ0(k_r(n)*r), DBESY0(k_r(n)*r), KIND=8)

        gradH(1) = gradH(1) - &
             k_r(n) * A_sqrd(n) * SIN(k1_z(n)*SOURCE_DEPTH) * SIN(k1_z(n)*height) * &
             EXP(k2_z(n)*(height-z)) * &
             CMPLX(DBESJ1(k_r(n)*r), DBESY1(k_r(n)*r), KIND=8)

        gradH(2) = gradH(2) - &
             k2_z(n) * A_sqrd(n) * SIN(k1_z(n)*SOURCE_DEPTH) * SIN(k1_z(n)*height) * &
             EXP(k2_z(n)*(height-z)) * &
             CMPLX(DBESJ0(k_r(n)*r), DBESY0(k_r(n)*r), KIND=8)
     ENDDO
  ENDIF

  H = CMPLX(-AIMAG(H), DBLE(H), KIND=8) * PI / rho1
  gradH = gradH * PI * (0,1.d0)/rho1

  IF (USE_DISCRETE_MODES_ONLY) THEN
     ! correct for z = -x(2):
     gradH(2) = -gradH(2)
     ! adjust for exp(+iwt) time dependence:
     H = CONJG(H)
     gradH = CONJG(gradH)
     RETURN
  ENDIF

  ! integrals:
  integral = 0
  CALL dqagse(real_integrand, branch_start, k1, 0.d0, quadrature_error, &
       IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
       WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
       IWORK, ilast)
  integral = quadrature_result
  
  CALL dqagse(imag_integrand, branch_start, k1, 0.d0, quadrature_error, &
       IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
       WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
       IWORK, ilast)
  integral = CMPLX(DBLE(integral), quadrature_result, KIND=8)
  
  cutoff = SQRT((100.d0 / r)**2 + k1_sqrd)
  CALL dqagse(imag_integrand_indef, k1, cutoff, 0.d0, quadrature_error, &
       IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
       WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
       IWORK, ilast)
  !   CALL dqagie(imag_integrand_indef, k1, 1, 0.d0, quadrature_error, &
  !        IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
  !        WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
  !        IWORK, ilast)
  integral = integral + CMPLX(0.d0, quadrature_result, KIND=8)
  
  H = H + 2*rho2_rho1 * CMPLX(-AIMAG(integral), DBLE(integral), KIND=8)

  ! d/dr
  CALL dqagse(real_integrand_dr, branch_start, k1, 0.d0, quadrature_error, &
       IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
       WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
       IWORK, ilast)
  integral = quadrature_result
  
  CALL dqagse(imag_integrand_dr, branch_start, k1, 0.d0, quadrature_error, &
       IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
       WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
       IWORK, ilast)
  integral = CMPLX(DBLE(integral), quadrature_result, KIND=8)
  
  CALL dqagse(imag_integrand_dr_indef, k1, cutoff, 0.d0, quadrature_error, &
       IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
       WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
       IWORK, ilast)
  !   CALL dqagie(imag_integrand_dr_indef, k1, 1, 0.d0, quadrature_error, &
  !        IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
  !        WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
  !        IWORK, ilast)
  integral = integral + CMPLX(0.d0, quadrature_result, KIND=8)
  
  gradH(1) = gradH(1) + 2*rho2_rho1 * CMPLX(-AIMAG(integral), DBLE(integral), KIND=8)
  
  
  ! d/dz
  CALL dqagse(real_integrand_dz, branch_start, k1, 0.d0, quadrature_error, &
       IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
       WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
       IWORK, ilast)
  integral = quadrature_result
  
  CALL dqagse(imag_integrand_dz, branch_start, k1, 0.d0, quadrature_error, &
       IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
       WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
       IWORK, ilast)
  integral = CMPLX(DBLE(integral), quadrature_result, KIND=8)
  
  CALL dqagse(imag_integrand_dz_indef, k1, cutoff, 0.d0, quadrature_error, &
       IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
       WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
       IWORK, ilast)
  !   CALL dqagie(imag_integrand_dz_indef, k1, 1, 0.d0, quadrature_error, &
  !        IWORK_LENGTH, quadrature_result, abs_error, neval, ierr, &
  !        WORK(1), WORK(1 + IWORK_LENGTH), WORK(1 + 2*IWORK_LENGTH), WORK(1 + 3*IWORK_LENGTH), &
  !        IWORK, ilast)
  integral = integral + CMPLX(0.d0, quadrature_result, KIND=8)

  gradH(2) = gradH(2) + 2*rho2_rho1 * CMPLX(-AIMAG(integral), DBLE(integral), KIND=8)

  ! correct for z = -x(2):
  gradH(2) = -gradH(2)
  ! adjust for exp(+iwt) time dependence:
  H = CONJG(H)
  gradH = CONJG(gradH)
  
  IF (index .EQ. 0) THEN
     CALL insert_entry(x, H, gradH, full_flag)
  ENDIF
END SUBROUTINE exact_pekeris


SUBROUTINE pekeris_moi(x, n_rays, p, grad_p)
  USE pekeris
  IMPLICIT NONE
  DOUBLE PRECISION, INTENT(IN) :: x(2)
  INTEGER, INTENT(IN) :: n_rays
  DOUBLE COMPLEX, INTENT(OUT) :: p, grad_p(2)

  INTEGER :: i, j
  DOUBLE PRECISION :: zs(4), n_s(4), n_b(4), d, z1, z2, fac, h, cos_inc
  DOUBLE COMPLEX :: temp, temp2, k2z, ref, cos_tra, kr

  IF (pekeris_initialized .EQV. .FALSE.) THEN
     CALL init_pekeris
  ENDIF

  r = x(1)
  z = -x(2)
  !z_s = SOURCE_DEPTH
  h = HEIGHT
  z1 = rho1 * c1
  z2 = rho2 * c2

  p = 0
  grad_p = 0

  DO i = 1, n_rays
     zs(1) = 2*(i-1)*h + z_s - z
     zs(2) = 2*(i-1)*h + z_s + z
     zs(3) = 2*i*h - z_s - z
     zs(4) = 2*i*h - z_s + z

     n_s(1) = i-1
     n_s(2) = i
     n_s(3) = i-1
     n_s(4) = i

     n_b(1) = i-1
     n_b(2) = i-1
     n_b(3) = i
     n_b(4) = i

     DO j = 1, 4
        d = SQRT(zs(j)**2 + r**2)
        cos_inc = ABS(zs(j))/d
        kr = x(1)/d * k1
        k2z = SQRT(k2**2 - kr**2)
        cos_tra = k2z/k2
        Ref = (z2*cos_inc - z1*cos_tra) / (z1*cos_tra + z2*cos_inc)
        temp = EXP((0.d0, 1.d0)*k1*d)/d * (-1)**n_s(j) * Ref**n_b(j)
        p = p + temp
        temp2 = temp/d * ((0.d0, 1.d0) * k1 - 1/d)
        grad_p(1) = grad_p(1) + r*temp2
        grad_p(2) = grad_p(2) + (-1)**j * zs(j)*temp2
     ENDDO
  ENDDO
  p = CONJG(p)
  grad_p(2) = -grad_p(2)
  grad_p = CONJG(grad_p)

END SUBROUTINE pekeris_moi


SUBROUTINE pekeris_test
  USE pekeris
  IMPLICIT NONE

  CALL init_pekeris
END SUBROUTINE pekeris_test
