/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ThermalPhysics

// clang-format off
#include "main.cc"

#include "test_thermal_physics.hh"
// clang-format on

// For now the radiative boundary conditions are not implemented on the device

BOOST_AUTO_TEST_CASE(thermal_2d_radiative_loss_host)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 10);
  geometry_database.put("length_divisions", 10);
  geometry_database.put("height", 10);
  geometry_database.put("height_divisions", 10);
  // Build Geometry
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  boost::property_tree::ptree database;
  // Material property
  database.put("materials.property_format", "polynomial");
  database.put("materials.n_materials", 1);
  database.put("materials.material_0.solid.density", 0.5);
  database.put("materials.material_0.powder.density", 0.5);
  database.put("materials.material_0.liquid.density", 0.5);
  database.put("materials.material_0.solid.specific_heat", 4.);
  database.put("materials.material_0.powder.specific_heat", 4.);
  database.put("materials.material_0.liquid.specific_heat", 4.);
  database.put("materials.material_0.solid.thermal_conductivity", 2.);
  database.put("materials.material_0.powder.thermal_conductivity", 2.);
  database.put("materials.material_0.liquid.thermal_conductivity", 2.);
  // Source database
  database.put("sources.n_sources", 1);
  database.put("sources.n_beams", 0);
  database.put("sources.hs_0.end_time", 5);
  database.put("sources.hs_0.value", 5);
  database.put("sources.hs_0.min_x", 4);
  database.put("sources.hs_0.min_y", 4);
  database.put("sources.hs_0.max_x", 6);
  database.put("sources.hs_0.max_y", 6);
  // Time-stepping database
  database.put("time_stepping.method", "forward_euler");
  // Boundary condition database
  database.put("boundary_conditions.radiative_loss", true);
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, dealii::MemorySpace::Host, dealii::QGauss<1>>
      physics(communicator, database, geometry);
  physics.setup_dofs();
  physics.compute_inverse_mass_matrix();

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solution;
  double constexpr initial_temperature = 10;
  // The final temperature should be 10.5 that requires a very small time step.
  double constexpr final_temperature = 10.505;
  physics.initialize_dof_vector(initial_temperature, solution);
  std::vector<adamantine::Timer> timers(6);
  double time = 0;
  int counter = 0;
  while (time < 100)
  {
    time = physics.evolve_one_time_step(time, 0.05, solution, timers);
    ++counter;
    if (counter % 25 == 0)
    {
      double max = -1;
      double min = 1e4;
      for (auto v : solution)
      {
        if (max < v)
          max = v;
        if (min > v)
          min = v;
      }
      std::cout << solution.mean_value() << " " << min << " " << max << " "
                << max - min << " " << time << std::endl;
    }
  }
}

BOOST_AUTO_TEST_CASE(thermal_2d_explicit_host)
{
  boost::property_tree::ptree database;
  // Time-stepping database
  database.put("time_stepping.method", "forward_euler");
  thermal_2d<dealii::MemorySpace::Host>(database, 0.05);
}

BOOST_AUTO_TEST_CASE(thermal_2d_implicit_host)
{
  boost::property_tree::ptree database;
  // Time-stepping database
  database.put("time_stepping.method", "backward_euler");
  database.put("time_stepping.max_iteration", 100);
  database.put("time_stepping.tolerance", 1e-6);
  database.put("time_stepping.n_tmp_vectors", 100);

  thermal_2d<dealii::MemorySpace::Host>(database, 0.025);
}

BOOST_AUTO_TEST_CASE(thermal_2d_manufactured_solution_host)
{
  thermal_2d_manufactured_solution<dealii::MemorySpace::Host>();
}

BOOST_AUTO_TEST_CASE(initial_temperature_host)
{
  initial_temperature<dealii::MemorySpace::Host>();
}

BOOST_AUTO_TEST_CASE(energy_conservation_host)
{
  energy_conservation<dealii::MemorySpace::Host>();
}
