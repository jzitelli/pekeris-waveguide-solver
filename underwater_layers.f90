MODULE underwater_layers
  IMPLICIT NONE

  DOUBLE PRECISION, ALLOCATABLE :: LAYER_RHO(:), LAYER_C(:), LAYER_BOTTOM(:)
  INTEGER :: N_LAYERS

  CONTAINS

  FUNCTION rho_func_layers(x)
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: x(2)
    DOUBLE PRECISION :: rho_func_layers
    INTEGER :: i_layer

    DO i_layer = 1, N_LAYERS-1
       IF (x(2) >= LAYER_BOTTOM(i_layer)) THEN
          rho_func_layers = LAYER_RHO(i_layer)
          RETURN
       ENDIF
    ENDDO

    rho_func_layers = LAYER_RHO(N_LAYERS)
  END FUNCTION rho_func_layers

  FUNCTION c_func_layers(x)
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: x(2)
    DOUBLE PRECISION :: c_func_layers
    INTEGER :: i_layer

    DO i_layer = 1, N_LAYERS-1
       IF (x(2) >= LAYER_BOTTOM(i_layer)) THEN
          c_func_layers = LAYER_C(i_layer)
          RETURN
       ENDIF
    ENDDO

    c_func_layers = LAYER_C(i_layer)
  END FUNCTION c_func_layers

END MODULE underwater_layers
