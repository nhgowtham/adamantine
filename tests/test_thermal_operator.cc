/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ThermalOperator

#include "main.cc"

#include "Geometry.hh"
#include "ThermalOperator.hh"
#include <boost/mpi.hpp>
#include <boost/property_tree/ptree.hpp>

BOOST_AUTO_TEST_CASE(thermal_operator)
{
  boost::mpi::communicator communicator;
  std::shared_ptr<adamantine::MaterialProperty> mat_properties;
  adamantine::ThermalOperator<2, 2, double> thermal_operator(communicator,
                                                             mat_properties);

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  // Create the DoFHandler
  dealii::DoFHandler<2> dof_handler(geometry.get_triangulation());
  dealii::FE_Q<2> fe(2);
  dof_handler.distribute_dofs(fe);
  dealii::ConstraintMatrix constraint_matrix;
  constraint_matrix.close();
  dealii::QGauss<1> quad(3);

  // Initialize the ThermalOperator
  thermal_operator.reinit(dof_handler, constraint_matrix, quad);
  BOOST_CHECK(thermal_operator.m() == 289);
  BOOST_CHECK(thermal_operator.m() == thermal_operator.n());

  // Check matrix-vector multiplications
  double const tolerance = 1e-15;
  dealii::LA::distributed::Vector<double> src;
  dealii::LA::distributed::Vector<double> dst_1;
  dealii::LA::distributed::Vector<double> dst_2;
  dealii::LA::distributed::Vector<double> dst_3;

  src = 1.;
  thermal_operator.vmult(dst_1, src);
  BOOST_CHECK_CLOSE(dst_1.l1_norm(), dst_1.l1_norm(), tolerance);

  thermal_operator.Tvmult(dst_2, src);
  BOOST_CHECK_CLOSE(dst_2.l1_norm(), dst_1.l1_norm(), tolerance * dst_1.size());

  thermal_operator.vmult_add(dst_2, src);
  dst_3 = dst_1;
  dst_3 += src;
  BOOST_CHECK_CLOSE(dst_2.l1_norm(), dst_3.l1_norm(), tolerance * dst_1.size());

  thermal_operator.Tvmult_add(dst_1, src);
  BOOST_CHECK_CLOSE(dst_1.l1_norm(), dst_2.l1_norm(), tolerance * dst_1.size());
}