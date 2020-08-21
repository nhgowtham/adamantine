/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef TYPES_HH
#define TYPES_HH

namespace dealii
{
namespace LinearAlgebra
{
}

/**
 * Shorten dealii::LinearAlgebra to dealii::LA.
 */
namespace LA = LinearAlgebra;
} // namespace dealii

namespace adamantine
{
/**
 * Enum on the possible materials.
 */
enum class MaterialState
{
  powder,
  solid,
  liquid,
  SIZE
};

/**
 * Enum on the possible material properties that depend on the state of the
 * material.
 */
// TODO add AnisotropicStateProperty
enum class StateProperty
{
  density,
  specific_heat,
  thermal_conductivity,
  emissivity,
  radiation_heat_transfer_coef,
  SIZE
};

/**
 * Enum on the possible material properties that do not depend on the state of
 * the material.
 */
enum class Property
{
  liquidus,
  solidus,
  latent_heat,
  radiation_temperature_infty,
  SIZE
};

/**
 * Enum on the possible timers.
 */
enum Timing
{
  main,
  refine,
  evol_time,
  evol_time_eval_th_ph,
  evol_time_J_inv,
  evol_time_eval_mat_prop
};

/**
 * TODO
 */
enum BoundaryFace
{
  bottom = 2,
  top = 3
};

/**
 * TODO
 */
struct Constant
{
  /**
   * Stefan-Boltzmann constant. Value form NIST [w/(m^2 K^4)].
   */
  static double constexpr stefan_boltzmann = 5.670374419e-8;
};
} // namespace adamantine

#endif
