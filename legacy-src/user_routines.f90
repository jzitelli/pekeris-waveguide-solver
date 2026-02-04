MODULE user_routines

CONTAINS

  FUNCTION rho_func(x, mdle)
    ! compute the density at physical point x
    USE pml
    USE underwater_layers
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: x(2)
    INTEGER, INTENT(IN), OPTIONAL :: mdle
    DOUBLE PRECISION :: rho_func

    INTEGER :: figure
    DOUBLE PRECISION :: xf(2,4)

    ! IF (PRESENT(mdle)) THEN
    !    CALL refel(mdle, figure, xf)
    !    figure = figure / 10
    !    IF (figure .EQ. 27) THEN
    !       rho_func = LAYER_RHO(1)
    !       RETURN
    !    ELSE IF (figure .EQ. 33) THEN
    !       rho_func = LAYER_RHO(2)
    !       RETURN
    !    ENDIF
    ! ENDIF

    IF (N_LAYERS >= 1) THEN
       rho_func = rho_func_layers(x)
       RETURN
    ENDIF

  END FUNCTION rho_func


  FUNCTION c_func(x, mdle)
    ! compute the speed of sound at physical point x
    USE pml
    USE underwater_layers
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: x(2)
    INTEGER, INTENT(IN), OPTIONAL :: mdle
    DOUBLE PRECISION :: c_func

    INTEGER :: figure
    DOUBLE PRECISION :: xf(2,4)

    ! IF (PRESENT(mdle)) THEN
    !    CALL refel(mdle, figure, xf)
    !    figure = figure / 10
    !    IF (figure .EQ. 27) THEN
    !       c_func = LAYER_C(1)
    !       RETURN
    !    ELSE IF (figure .EQ. 33) THEN
    !       c_func = LAYER_C(2)
    !       RETURN
    !    ENDIF
    ! ENDIF

    IF (N_LAYERS >= 1) THEN
       c_func = c_func_layers(x)
       RETURN
    ENDIF

  END FUNCTION c_func


  FUNCTION alpha_func(x)
    ! calculate the absorption coefficient at the physical point x.  The returned
    ! value has units [nepers/distance].
    USE underwater_layers
    USE pml
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: x(2)
    DOUBLE PRECISION :: alpha_func
    DOUBLE PRECISION, PARAMETER :: PI = ACOS(-1.d0)
    ! conversion from dB/km to nepers/m
    DOUBLE PRECISION, PARAMETER :: conv_factor = 1 / (20 * LOG10(EXP(1.d0))) / 1000.0
    DOUBLE PRECISION, PARAMETER :: boric_acid_coef = &
         8.86 * 10.d0**(8.0*0.78 - 5.0)
    DOUBLE PRECISION, PARAMETER :: boric_acid_freq = 28000.d0
    DOUBLE PRECISION :: omega
    DOUBLE PRECISION :: f_sqrd

    omega = PML_omega

    ! no absorption
    alpha_func = 0
    RETURN

    ! COA, p. 35
    f_sqrd = omega**2 / (2*PI*1000.d0)**2 ! frequency in kilohertz
    alpha_func = 3.3d-3 + 0.11*f_sqrd/(1 + f_sqrd)
    alpha_func = alpha_func * conv_factor
    RETURN

    ! Blackstock, p. 517
    alpha_func = boric_acid_coef / c_func(x) * boric_acid_freq * OMEGA**2 &
         / (boric_acid_freq**2 + OMEGA**2)
    alpha_func = alpha_func * conv_factor
    RETURN
  END FUNCTION alpha_func

END MODULE user_routines
