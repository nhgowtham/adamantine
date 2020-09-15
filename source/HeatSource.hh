/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef HEAT_SOURCE_HH
#define HEAT_SOURCE_HH

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace adamantine
{
template <int dim>
class HeatSource : public dealii::Function<dim>
{
public:
  /**
   * Compute the heat source at a given point at the current time.
   */
  virtual double value(dealii::Point<dim> const &point,
                       unsigned int const component = 0) const = 0;

  /**
   * Reset the current time and the position to the last saved state.
   */
  virtual void rewind_time() = 0;

  /**
   * Save the current time and the position in the list of successive positions
   * of the beam.
   */
  virtual void save_time() = 0;
};
} // namespace adamantine

#endif
