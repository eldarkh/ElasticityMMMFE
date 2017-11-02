# ElasticityMMMFE
Multiscale Mortar Mixed Finite Element method for elasticity

  This program implements multiscale mortar mixed finite element
  method for linear elasticity model. The elasticity system is
  written in a three-field form, with stress, displacement and
  rotation as variables. The domain decomposition procedure
  is then obtained by matching the normal components of stresses
  across the interface.
 
  This implementation allows for non-matching grids by utilizing
  the mortar finite element space on the interface. To speed things
  up a little, the multiscale stress basis is also available for
  the cases when the mortar grid is much coarser than the subdomain
  ones.
