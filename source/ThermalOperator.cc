/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ThermalOperator.hh>
#include <instantiation.hh>

#include <deal.II/base/index_set.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/matrix_free/fe_evaluation.h>

namespace adamantine
{

template <int dim, int fe_degree, typename MemorySpaceType>
ThermalOperator<dim, fe_degree, MemorySpaceType>::ThermalOperator(
    MPI_Comm const &communicator,
    std::shared_ptr<MaterialProperty<dim>> material_properties,
    bool radiative_bc)
    : _communicator(communicator), _radiative_bc(radiative_bc),
      _material_properties(material_properties),
      _inverse_mass_matrix(
          new dealii::LA::distributed::Vector<double, MemorySpaceType>())
{
  _matrix_free_data.tasks_parallel_scheme =
      dealii::MatrixFree<dim, double>::AdditionalData::partition_color;
  _matrix_free_data.mapping_update_flags = dealii::update_gradients |
                                           dealii::update_JxW_values |
                                           dealii::update_quadrature_points;
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints,
    dealii::QGaussLobatto<1> const &quad)
{
  _matrix_free.reinit(dof_handler, affine_constraints, quad, _matrix_free_data);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints,
    dealii::QGauss<1> const &quad)
{
  _matrix_free.reinit(dof_handler, affine_constraints, quad, _matrix_free_data);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::
    compute_inverse_mass_matrix(
        dealii::DoFHandler<dim> const &dof_handler,
        dealii::AffineConstraints<double> const &affine_constraints)
{
  // Compute the inverse of the mass matrix
  dealii::QGaussLobatto<1> mass_matrix_quad(fe_degree + 1);
  dealii::MatrixFree<dim, double> mass_matrix_free;
  typename dealii::MatrixFree<dim, double>::AdditionalData mf_data;
  mf_data.tasks_parallel_scheme =
      dealii::MatrixFree<dim, double>::AdditionalData::partition_color;
  mf_data.mapping_update_flags = dealii::update_values |
                                 dealii::update_JxW_values |
                                 dealii::update_quadrature_points;

  mass_matrix_free.reinit(dof_handler, affine_constraints, mass_matrix_quad,
                          mf_data);
  mass_matrix_free.initialize_dof_vector(*_inverse_mass_matrix);
  dealii::VectorizedArray<double> one =
      dealii::make_vectorized_array(static_cast<double>(1.));
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(
      mass_matrix_free);
  unsigned int const n_q_points = fe_eval.n_q_points;
  for (unsigned int cell = 0; cell < mass_matrix_free.n_macro_cells(); ++cell)
  {
    fe_eval.reinit(cell);
    for (unsigned int q = 0; q < n_q_points; ++q)
      fe_eval.submit_value(one, q);
    fe_eval.integrate(true, false);
    fe_eval.distribute_local_to_global(*_inverse_mass_matrix);
  }
  _inverse_mass_matrix->compress(dealii::VectorOperation::add);
  unsigned int const local_size = _inverse_mass_matrix->local_size();
  for (unsigned int k = 0; k < local_size; ++k)
  {
    if (_inverse_mass_matrix->local_element(k) > 1e-15)
      _inverse_mass_matrix->local_element(k) =
          1. / _inverse_mass_matrix->local_element(k);
    else
      _inverse_mass_matrix->local_element(k) = 0.;
  }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::clear()
{
  _matrix_free.clear();
  _inverse_mass_matrix->reinit(0);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::vmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  dst = 0.;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::Tvmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  dst = 0.;
  Tvmult_add(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::vmult_add(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  // Execute the matrix-free matrix-vector multiplication
  _matrix_free.loop(&ThermalOperator::cell_local_apply,
                    &ThermalOperator::face_local_apply,
                    &ThermalOperator::boundary_local_apply, this, dst, src);

  // Because cell_loop resolves the constraints, the constrained dofs are not
  // called they stay at zero. Thus, we need to force the value on the
  // constrained dofs by hand. The variable scaling is used so that we get the
  // right order of magnitude.
  // TODO: for now the value of scaling is set to 1
  double const scaling = 1.;
  std::vector<unsigned int> const &constrained_dofs =
      _matrix_free.get_constrained_dofs();
  for (auto &dof : constrained_dofs)
    dst.local_element(dof) += scaling * src.local_element(dof);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::Tvmult_add(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  // The system of equation is symmetric so we can use vmult_add
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::cell_local_apply(
    dealii::MatrixFree<dim, double> const &data,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src,
    std::pair<unsigned int, unsigned int> const &cell_range) const
{
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(data);
  dealii::Tensor<1, dim> unit_tensor;
  for (unsigned int i = 0; i < dim; ++i)
    unit_tensor[i] = 1.;

  // Loop over the "cells". Note that we don't really work on a cell but on a
  // set of quadrature point.
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    // Reinit fe_eval on the current cell
    fe_eval.reinit(cell);
    // Store in a local vector the local values of src
    fe_eval.read_dof_values(src);
    // Evaluate only the function gradients on the reference cell
    fe_eval.evaluate(false, true);
    // Apply the Jacobian of the transformation, multiply by the variable
    // coefficients and the quadrature points
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      fe_eval.submit_gradient(-_inv_rho_cp(cell, q) *
                                  _thermal_conductivity(cell, q) *
                                  fe_eval.get_gradient(q),
                              q);
    // Sum over the quadrature points.
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::face_local_apply(
    dealii::MatrixFree<dim, double> const &,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &,
    std::pair<unsigned int, unsigned int> const &) const
{
  // no-op
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::boundary_local_apply(
    dealii::MatrixFree<dim, double> const &data,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src,
    std::pair<unsigned int, unsigned int> const &face_range) const
{
  dealii::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
      fe_face_eval(data, false);

  for (unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    auto const boundary_id = data.get_boundary_id(face);
    if (boundary_id !=
        static_cast<dealii::types::boundary_id>(BoundaryFace::bottom))
    {
      fe_face_eval.reinit(face);
      // gather_evaluate combines read_dof_value and evaluate
      fe_face_eval.gather_evaluate(src, dealii::EvaluationFlags::values);
      for (unsigned int q = 0; q < fe_face_eval.n_q_points; ++q)
        fe_face_eval.submit_value(-_inv_rho_cp_boundary(face, q) *
                                      _rad_heat_transfer(face, q) *
                                      fe_face_eval.get_value(q),
                                  q);
      // integrate_scatter combines integrate and distribute_local_to_global
      fe_face_eval.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
  }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperator<dim, fe_degree, MemorySpaceType>::
    evaluate_material_properties(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &temperature)
{
  // Update the state of the materials
  _material_properties->update(_matrix_free.get_dof_handler(), temperature);

  // Store the volumetric material properties
  unsigned int const n_cells = _matrix_free.n_macro_cells();
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(
      _matrix_free);
  unsigned int const fe_eval_n_q_points = fe_eval.n_q_points;
  _inv_rho_cp.reinit(n_cells, fe_eval_n_q_points);
  _thermal_conductivity.reinit(n_cells, fe_eval_n_q_points);
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    for (unsigned int q = 0; q < fe_eval_n_q_points; ++q)
      for (unsigned int i = 0;
           i < _matrix_free.n_active_entries_per_cell_batch(cell); ++i)
      {
        typename dealii::DoFHandler<dim>::cell_iterator cell_it =
            _matrix_free.get_cell_iterator(cell, i);
        // Cast to Triangulation<dim>::cell_iterator to access the material_id
        typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(
            cell_it);

        _thermal_conductivity(cell, q)[i] = _material_properties->get(
            cell_tria, StateProperty::thermal_conductivity);

        double const inv_rho_cp =
            1. / (_material_properties->get(cell_tria, StateProperty::density) *
                  _material_properties->get(cell_tria,
                                            StateProperty::specific_heat));
        ASSERT(std::isfinite(inv_rho_cp), "density or specific_heat is zero");
        _inv_rho_cp(cell, q)[i] = inv_rho_cp;
        _cell_it_to_mf_cell_map[cell_it] = std::make_pair(cell, i);
      }

  // Store the material properties of the boundary faces
  // Skip if the radiative boundary is turned off
  if (this->_radiative_bc)
  {
    // First we need to skip all the inner faces
    unsigned int const inner_face_offset = _matrix_free.n_inner_face_batches();
    unsigned int const n_boundary_faces =
        _matrix_free.n_boundary_face_batches();
    dealii::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
        fe_face_eval(_matrix_free);
    unsigned int const fe_face_eval_n_q_points = fe_face_eval.n_q_points;
    _inv_rho_cp_boundary.reinit(n_boundary_faces, fe_face_eval_n_q_points);
    _rad_heat_transfer.reinit(n_boundary_faces, fe_face_eval_n_q_points);
    for (unsigned int boundary_face = inner_face_offset;
         boundary_face < inner_face_offset + n_boundary_faces; ++boundary_face)
      for (unsigned int q = 0; q < fe_face_eval_n_q_points; ++q)
        for (unsigned int i = 0;
             i < _matrix_free.n_active_entries_per_face_batch(boundary_face);
             ++i)
        {
          auto cell_it_face_id_pair =
              _matrix_free.get_face_iterator(boundary_face, i);
          // Cast to Triangulation<dim>::cell_iterator to access the material_id
          typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(
              cell_it_face_id_pair.first);

          // Skip the bottom boundary because there is no radiation loss.
          if (cell_tria->face(cell_it_face_id_pair.second)->boundary_id() !=
              static_cast<dealii::types::boundary_id>(BoundaryFace::bottom))
          {

            // Assume the material property is constant per cell. So we don't
            // need the face number to get the material property.
            _inv_rho_cp_boundary(boundary_face, q)[i] =
                1. /
                (_material_properties->get(cell_tria, StateProperty::density) *
                 _material_properties->get(cell_tria,
                                           StateProperty::specific_heat));
            _rad_heat_transfer(boundary_face, q)[i] = _material_properties->get(
                cell_tria, StateProperty::radiation_heat_transfer_coef);
          }
        }
  }
}
} // namespace adamantine

INSTANTIATE_DIM_FEDEGREE_HOST(TUPLE(ThermalOperator))
