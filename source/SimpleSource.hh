/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef SIMPLE_SOURCE_HH
#define SIMPLE_SOURCE_HH

#include <HeatSource.hh>

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
template <int dim>
class SimpleSource : public HeatSource<dim>
{
public:
  SimpleSource(boost::property_tree::ptree const &database);

  /**
   * Compute the heat source at a given point at the current time.
   */
  double value(dealii::Point<dim> const &point,
               unsigned int const component = 0) const override;

  /**
   * Reset the current time and the position to the last saved state.
   */
  void rewind_time() override;

  /**
   * Save the current time and the position in the list of successive positions
   * of the beam.
   */
  void save_time() override;

private:
  mutable double _current_time;
  double _saved_time;
  double _end_time;
  double _value;
  dealii::Point<dim> _min_point;
  dealii::Point<dim> _max_point;
};
} // namespace adamantine

#endif
