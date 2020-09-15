/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <SimpleSource.hh>
#include <instantiation.hh>

namespace adamantine
{
template <int dim>
SimpleSource<dim>::SimpleSource(boost::property_tree::ptree const &database)
{
  _end_time = database.get<double>("end_time");
  _value = database.get<double>("value");
  _min_point[0] = database.get<double>("min_x");
  _max_point[0] = database.get<double>("max_x");
  _min_point[1] = database.get<double>("min_y");
  _max_point[1] = database.get<double>("max_y");
  if (dim == 3)
  {
    _min_point[2] = database.get<double>("min_z");
    _max_point[2] = database.get<double>("max_z");
  }
}

template <int dim>
double SimpleSource<dim>::value(dealii::Point<dim> const &point,
                                unsigned int const /*component*/) const
{
  _current_time = this->get_time();
  if (_current_time < _end_time)
  {
    bool in_source = true;
    for (int i = 0; i < dim; ++i)
    {
      if ((point[i] < _min_point[i]) || (point[i] > _max_point[i]))
      {
        in_source = false;
        break;
      }
    }

    if (in_source)
      return _value;
  }

  return 0.;
}

template <int dim>
void SimpleSource<dim>::rewind_time()
{
  _current_time = _saved_time;
}

template <int dim>
void SimpleSource<dim>::save_time()
{
  _saved_time = _current_time;
}
} // namespace adamantine

INSTANTIATE_DIM(SimpleSource)
