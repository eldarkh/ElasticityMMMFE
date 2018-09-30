/* ---------------------------------------------------------------------
 * Implementation of the MixedElasticityProblemDD class
 * ---------------------------------------------------------------------
 *
 * Author: Eldar Khattatov, University of Pittsburgh, 2016 - 2017
 */

// Internals
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgp_nonparametric.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
// Extra for MPI and mortars
#include <deal.II/base/timer.h>

#include <deal.II/numerics/fe_field_function.h>
// C++
#include <cstdlib>
#include <fstream>
#include <iostream>
// Utilities, data, etc.
#include "../inc/data.h"
#include "../inc/elasticity_mfedd.h"
#include "../inc/utilities.h"

//TODO: FIX INTERFACE ERROR COMPUTATION
namespace dd_elasticity
{
  using namespace dealii;

  // MixedElasticityDD class constructor
  template <int dim>
  MixedElasticityProblemDD<dim>::MixedElasticityProblemDD(
    const unsigned int degree,
    const unsigned int mortar_flag,
    const unsigned int mortar_degree)
    : mpi_communicator(MPI_COMM_WORLD)
    , P_coarse2fine(false)
    , P_fine2coarse(false)
    , n_domains(dim, 0)
    , degree(degree)
    , mortar_degree(mortar_degree)
    , mortar_flag(mortar_flag)
    , cg_iteration(0)
    , qdegree(11)
    , fe(FE_BDM<dim>(degree),
         dim,
         FE_DGQ<dim>(degree - 1),
         dim,
         FE_Q<dim>(degree),
         0.5 * dim * (dim - 1))
    , dof_handler(triangulation)
    , fe_mortar(FE_RaviartThomas<dim>(mortar_degree),
                dim,
                FE_Nothing<dim>(),
                dim,
                FE_Nothing<dim>(),
                0.5 * dim * (dim - 1))
    , dof_handler_mortar(triangulation_mortar)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}


  // MixedElasticityProblemDD::make_grid_and_dofs
  template <int dim>
  void
  MixedElasticityProblemDD<dim>::make_grid_and_dofs()
  {
    TimerOutput::Scope t(computing_timer, "Make grid and DoFs");
    system_matrix.clear();

    double             lower_left, upper_right;
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(mpi_communicator);
    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);

    // Find neighbors
    neighbors.resize(GeometryInfo<dim>::faces_per_cell, 0);
    find_neighbors(dim, this_mpi, n_domains, neighbors);

    // Make interface data structures
    faces_on_interface.resize(GeometryInfo<dim>::faces_per_cell, 0);
    faces_on_interface_mortar.resize(GeometryInfo<dim>::faces_per_cell, 0);

    // Label interface faces and count how many of them there are per interface
    mark_interface_faces(triangulation, neighbors, p1, p2, faces_on_interface);
    if (mortar_flag)
      mark_interface_faces(
        triangulation_mortar, neighbors, p1, p2, faces_on_interface_mortar);

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler);

    if (mortar_flag)
      dof_handler_mortar.distribute_dofs(fe_mortar);

    std::vector<types::global_dof_index> dofs_per_component(
      dim * dim + dim + 0.5 * dim * (dim - 1));
    DoFTools::count_dofs_per_component(dof_handler, dofs_per_component);
    unsigned int n_s = 0, n_u = 0, n_p = 0;

    for (unsigned int i = 0; i < dim; ++i)
      {
        n_s += dofs_per_component[i * dim];
        n_u += dofs_per_component[dim * dim + i];

        // Rotation is scalar in 2d and vector in 3d, so this:
        if (dim == 2)
          n_p = dofs_per_component[dim * dim + dim];
        else if (dim == 3)
          n_p += dofs_per_component[dim * dim + dim + i];
      }

    n_stress_interface = n_s;

    BlockDynamicSparsityPattern dsp(3, 3);
    dsp.block(0, 0).reinit(n_s, n_s);
    dsp.block(0, 1).reinit(n_s, n_u);
    dsp.block(0, 2).reinit(n_s, n_p);
    dsp.block(1, 0).reinit(n_u, n_s);
    dsp.block(1, 1).reinit(n_u, n_u);
    dsp.block(1, 2).reinit(n_u, n_p);
    dsp.block(2, 0).reinit(n_p, n_s);
    dsp.block(2, 1).reinit(n_p, n_u);
    dsp.block(2, 2).reinit(n_p, n_p);
    dsp.collect_sizes();
    DoFTools::make_sparsity_pattern(dof_handler, dsp);

    // Initialize system matrix
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    // Reinit solution and RHS vectors
    solution_bar.reinit(3);
    solution_bar.block(0).reinit(n_s);
    solution_bar.block(1).reinit(n_u);
    solution_bar.block(2).reinit(n_p);
    solution_bar.collect_sizes();
    solution_bar = 0;

    // Reinit solution and RHS vectors
    solution_star.reinit(3);
    solution_star.block(0).reinit(n_s);
    solution_star.block(1).reinit(n_u);
    solution_star.block(2).reinit(n_p);
    solution_star.collect_sizes();
    solution_star = 0;

    system_rhs_bar.reinit(3);
    system_rhs_bar.block(0).reinit(n_s);
    system_rhs_bar.block(1).reinit(n_u);
    system_rhs_bar.block(2).reinit(n_p);
    system_rhs_bar.collect_sizes();
    system_rhs_bar = 0;

    system_rhs_star.reinit(3);
    system_rhs_star.block(0).reinit(n_s);
    system_rhs_star.block(1).reinit(n_u);
    system_rhs_star.block(2).reinit(n_p);
    system_rhs_star.collect_sizes();
    system_rhs_star = 0;

    if (mortar_flag)
      {
        std::vector<types::global_dof_index> dofs_per_component_mortar(
          dim * dim + dim + 0.5 * dim * (dim - 1));
        DoFTools::count_dofs_per_component(dof_handler_mortar,
                                           dofs_per_component_mortar);
        unsigned int n_s_mortar = 0, n_u_mortar = 0, n_p_mortar = 0;

        for (unsigned int i = 0; i < dim; ++i)
          {
            n_s_mortar += dofs_per_component_mortar[i * dim];
            n_u_mortar += dofs_per_component_mortar[dim * dim + i];

            // Rotation is scalar in 2d and vector in 3d, so this:
            if (dim == 2)
              n_p_mortar = dofs_per_component_mortar[dim * dim + dim];
            else if (dim == 3)
              n_p_mortar += dofs_per_component_mortar[dim * dim + dim + i];
          }

        n_stress_interface = n_s_mortar;

        solution_bar_mortar.reinit(3);
        solution_bar_mortar.block(0).reinit(n_s_mortar);
        solution_bar_mortar.block(1).reinit(n_u_mortar);
        solution_bar_mortar.block(2).reinit(n_p_mortar);
        solution_bar_mortar.collect_sizes();

        solution_star_mortar.reinit(3);
        solution_star_mortar.block(0).reinit(n_s_mortar);
        solution_star_mortar.block(1).reinit(n_u_mortar);
        solution_star_mortar.block(2).reinit(n_p_mortar);
        solution_star_mortar.collect_sizes();
      }
    pcout << "N stress dofs: " << n_stress_interface << std::endl;
  }


  // MixedElasticityProblemDD - assemble_system
  template <int dim>
  void
  MixedElasticityProblemDD<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "Assemble system");
    system_matrix  = 0;
    system_rhs_bar = 0;

    QGauss<dim>     quadrature_formula(degree + 3);
    QGauss<dim - 1> face_quadrature_formula(qdegree);

    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // std::vector<double>     lambda_values (n_q_points);
    // std::vector<double>     mu_values (n_q_points);

    const LameParameters<dim> lame_function;

    // const LameFirstParameter<dim>         lmbda_function;
    // const LameSecondParameter<dim>        mu_function;
    const RightHandSide<dim>              right_hand_side;
    const DisplacementBoundaryValues<dim> displacement_boundary_values;

    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim));
    std::vector<Vector<double>> boundary_values(n_face_q_points,
                                                Vector<double>(dim));
    std::vector<Vector<double>> lame_parameters_values(n_q_points,
                                                       Vector<double>(2));

    // Rotation variable is either a scalar(2d) or a vector(3d)
    const unsigned int rotation_dim = 0.5 * dim * (dim - 1);
    // Stress DoFs vectors
    std::vector<FEValuesExtractors::Vector> stresses(
      dim, FEValuesExtractors::Vector());
    std::vector<FEValuesExtractors::Scalar> rotations(
      rotation_dim, FEValuesExtractors::Scalar());

    const FEValuesExtractors::Vector stressx(0);
    const FEValuesExtractors::Vector stressy(2);
    const FEValuesExtractors::Scalar rotation(6);
    // Displacement DoFs
    const FEValuesExtractors::Vector displacement(dim * dim);

    for (unsigned int i = 0; i < dim; ++i)
      {
        const FEValuesExtractors::Vector tmp_stress(i * dim);
        stresses[i].first_vector_component = tmp_stress.first_vector_component;
        if (dim == 2 && i == 0)
          {
            const FEValuesExtractors::Scalar tmp_rotation(dim * dim + dim);
            rotations[i].component = tmp_rotation.component;
          }
        else if (dim == 3)
          {
            const FEValuesExtractors::Scalar tmp_rotation(dim * dim + dim + i);
            rotations[i].component = tmp_rotation.component;
          }
      }

    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;

        lame_function.vector_value_list(fe_values.get_quadrature_points(),
                                        lame_parameters_values);
        // lmbda_function.value_list (fe_values.get_quadrature_points(),
        // lambda_values); mu_function.value_list
        // (fe_values.get_quadrature_points(), mu_values);

        right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                // Stress and divergence
                std::vector<Tensor<1, dim>> phi_i_s(dim);
                Tensor<1, dim>              div_phi_i_s;
                for (unsigned int s_i = 0; s_i < dim; ++s_i)
                  {
                    phi_i_s[s_i] = fe_values[stresses[s_i]].value(i, q);
                    div_phi_i_s[s_i] =
                      fe_values[stresses[s_i]].divergence(i, q);
                  }
                // Displacement
                Tensor<1, dim> phi_i_u = fe_values[displacement].value(i, q);
                // Rotations
                Tensor<1, rotation_dim> phi_i_p;
                for (unsigned int r_i = 0; r_i < rotation_dim; ++r_i)
                  phi_i_p[r_i] = fe_values[rotations[r_i]].value(i, q);

                // Make Asigma
                Tensor<2, dim> asigma;
                compliance_tensor(phi_i_s,
                                  lame_parameters_values[q][1],
                                  lame_parameters_values[q][0],
                                  asigma);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    // Stress and divergence
                    std::vector<Tensor<1, dim>> phi_j_s(dim);
                    Tensor<1, dim>              div_phi_j_s;
                    for (unsigned int s_j = 0; s_j < dim; ++s_j)
                      {
                        phi_j_s[s_j] = fe_values[stresses[s_j]].value(j, q);
                        div_phi_j_s[s_j] =
                          fe_values[stresses[s_j]].divergence(j, q);
                      }
                    // Displacement
                    const Tensor<1, dim> phi_j_u =
                      fe_values[displacement].value(j, q);
                    // Rotations
                    Tensor<1, rotation_dim> phi_j_p;
                    for (unsigned int r_j = 0; r_j < rotation_dim; ++r_j)
                      phi_j_p[r_j] = fe_values[rotations[r_j]].value(j, q);

                    Tensor<2, dim>          sigma;
                    Tensor<1, rotation_dim> as_phi_j_s, as_phi_i_s;
                    make_tensor(phi_j_s, sigma);
                    make_asymmetry_tensor(phi_i_s, as_phi_i_s);
                    make_asymmetry_tensor(phi_j_s, as_phi_j_s);

                    local_matrix(i, j) +=
                      (scalar_product(asigma, sigma) +
                       scalar_product(phi_i_u, div_phi_j_s) +
                       scalar_product(phi_j_u, div_phi_i_s) +
                       scalar_product(phi_i_p, as_phi_j_s) +
                       scalar_product(phi_j_p, as_phi_i_s) +
                       cell->diameter() * cell->diameter() * phi_i_u * phi_j_u +
                       cell->diameter() * cell->diameter() * phi_i_p *
                         phi_j_p) *
                      fe_values.JxW(q);
                  }

                for (unsigned d_i = 0; d_i < dim; ++d_i)
                  local_rhs(i) +=
                    -(phi_i_u[d_i] * rhs_values[q][d_i]) * fe_values.JxW(q);
              }
          }
        for (unsigned int face_no = 0;
             face_no < GeometryInfo<dim>::faces_per_cell;
             ++face_no)
          if (cell->at_boundary(face_no) &&
              cell->face(face_no)->boundary_id() ==
                0) // && (cell->face(face_no)->boundary_id() == 1)
            {
              fe_face_values.reinit(cell, face_no);

              displacement_boundary_values.vector_value_list(
                fe_face_values.get_quadrature_points(), boundary_values);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    Tensor<2, dim> sigma;
                    for (unsigned int d_i = 0; d_i < dim; ++d_i)
                      sigma[d_i] = fe_face_values[stresses[d_i]].value(i, q);

                    Tensor<1, dim> sigma_n =
                      sigma * fe_face_values.normal_vector(q);
                    for (unsigned int d_i = 0; d_i < dim; ++d_i)
                      local_rhs(i) +=
                        ((sigma_n[d_i] * boundary_values[q][d_i]) *
                         fe_face_values.JxW(q));
                  }
            }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              local_matrix(i, j));
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          system_rhs_bar(local_dof_indices[i]) += local_rhs(i);
      }
  }

  // MixedElasticityProblemDD - initialize the interface data structure
  template <int dim>
  void
  MixedElasticityProblemDD<dim>::get_interface_dofs()
  {
    TimerOutput::Scope t(computing_timer, "Get interface DoFs");
    interface_dofs.resize(GeometryInfo<dim>::faces_per_cell,
                          std::vector<types::global_dof_index>());

    std::vector<types::global_dof_index> local_face_dof_indices;

    typename DoFHandler<dim>::active_cell_iterator cell, endc;
    unsigned int                                   n_stress = 0;


    if (mortar_flag == 0)
      {
        cell = dof_handler.begin_active(), endc = dof_handler.end();
        local_face_dof_indices.resize(fe.dofs_per_face);
      }
    else
      {
        cell = dof_handler_mortar.begin_active(),
        endc = dof_handler_mortar.end();
        local_face_dof_indices.resize(fe_mortar.dofs_per_face);
      }

    for (; cell != endc; ++cell)
      {
        for (unsigned int face_n = 0;
             face_n < GeometryInfo<dim>::faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n) &&
              cell->face(face_n)->boundary_id() != 0)
            {
              cell->face(face_n)->get_dof_indices(local_face_dof_indices, 0);

              for (auto el : local_face_dof_indices)
                if (el < n_stress_interface)
                  interface_dofs[cell->face(face_n)->boundary_id() - 1]
                    .push_back(el);
            }
      }
  }


  // MixedElasticityProblemDD - assemble RHS of star problems
  template <int dim>
  void
  MixedElasticityProblemDD<dim>::assemble_rhs_star(
    FEFaceValues<dim> &fe_face_values)
  {
    TimerOutput::Scope t(computing_timer, "Assemble RHS star");
    system_rhs_star = 0;

    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;

    Vector<double>                       local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<FEValuesExtractors::Vector> stresses(
      dim, FEValuesExtractors::Vector());

    for (unsigned int d = 0; d < dim; ++d)
      {
        const FEValuesExtractors::Vector tmp_stress(d * dim);
        stresses[d].first_vector_component = tmp_stress.first_vector_component;
      }

    std::vector<std::vector<Tensor<1, dim>>> interface_values(
      dim, std::vector<Tensor<1, dim>>(n_face_q_points));

    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        local_rhs = 0;
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int face_n = 0;
             face_n < GeometryInfo<dim>::faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n) &&
              cell->face(face_n)->boundary_id() != 0)
            {
              fe_face_values.reinit(cell, face_n);

              for (unsigned int d_i = 0; d_i < dim; ++d_i)
                fe_face_values[stresses[d_i]].get_function_values(
                  interface_fe_function, interface_values[d_i]);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    Tensor<2, dim> sigma;
                    Tensor<2, dim> interface_lambda;
                    for (unsigned int d_i = 0; d_i < dim; ++d_i)
                      sigma[d_i] = fe_face_values[stresses[d_i]].value(i, q);

                    Tensor<1, dim> sigma_n =
                      sigma * fe_face_values.normal_vector(q);
                    for (unsigned int d_i = 0; d_i < dim; ++d_i)
                      local_rhs(i) +=
                        fe_face_values[stresses[d_i]].value(i, q) *
                        fe_face_values.normal_vector(q) *
                        interface_values[d_i][q] *
                        get_normal_direction(cell->face(face_n)->boundary_id() -
                                             1) *
                        fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
                  }
            }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          system_rhs_star(local_dof_indices[i]) += local_rhs(i);
      }
  }


  // MixedElasticityProblemDD::solvers
  template <int dim>
  void
  MixedElasticityProblemDD<dim>::solve_bar()
  {
    TimerOutput::Scope t(computing_timer, "Solve bar");

    if (cg_iteration == 0)
      A_direct.initialize(system_matrix);

    pcout << "  ...factorized..."
          << "\n";
    A_direct.vmult(solution_bar, system_rhs_bar);
  }

  template <int dim>
  void
  MixedElasticityProblemDD<dim>::solve_star()
  {
    TimerOutput::Scope t(computing_timer, "Solve star");

    A_direct.vmult(solution_star, system_rhs_star);
  }


  template <int dim>
  void
  MixedElasticityProblemDD<dim>::compute_multiscale_basis()
  {
    TimerOutput::Scope t(computing_timer, "Compute multiscale basis");
    ConstraintMatrix   constraints;
    QGauss<dim - 1>    quad(qdegree);
    FEFaceValues<dim>  fe_face_values(fe,
                                     quad,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    std::vector<size_t> block_sizes{solution_bar_mortar.block(0).size(),
                                    solution_bar_mortar.block(1).size()};
    long                n_interface_dofs = 0;

    for (auto vec : interface_dofs)
      for (auto el : vec)
        n_interface_dofs += 1;

    multiscale_basis.resize(n_interface_dofs);
    BlockVector<double> tmp_basis(solution_bar_mortar);

    interface_fe_function.reinit(solution_bar);

    unsigned int ind = 0;
    for (unsigned int side = 0; side < GeometryInfo<dim>::faces_per_cell;
         ++side)
      for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
        {
          interface_fe_function = 0;
          multiscale_basis[ind].reinit(solution_bar_mortar);
          multiscale_basis[ind] = 0;

          tmp_basis                          = 0;
          tmp_basis[interface_dofs[side][i]] = 1.0;

          project_mortar(P_coarse2fine,
                         dof_handler_mortar,
                         tmp_basis,
                         quad,
                         constraints,
                         neighbors,
                         dof_handler,
                         interface_fe_function);

          interface_fe_function.block(2) = 0;
          assemble_rhs_star(fe_face_values);
          solve_star();

          project_mortar(P_fine2coarse,
                         dof_handler,
                         solution_star,
                         quad,
                         constraints,
                         neighbors,
                         dof_handler_mortar,
                         multiscale_basis[ind]);
          ind += 1;
        }
  }

  template <int dim>
  void
  MixedElasticityProblemDD<dim>::local_cg(const unsigned int &maxiter)
  {
    TimerOutput::Scope t(computing_timer, "Local CG");

    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(mpi_communicator);
    const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    std::vector<std::vector<double>> interface_data_receive(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data_send(n_faces_per_cell);
    std::vector<std::vector<double>> interface_data(n_faces_per_cell);
    std::vector<std::vector<double>> lambda(n_faces_per_cell);

    for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        {
          if (mortar_flag)
            {
              interface_data_receive[side].resize(interface_dofs[side].size(),
                                                  0);
              interface_data_send[side].resize(interface_dofs[side].size(), 0);
              interface_data[side].resize(interface_dofs[side].size(), 0);
            }
          else
            {
              interface_data_receive[side].resize(interface_dofs[side].size(),
                                                  0);
              interface_data_send[side].resize(interface_dofs[side].size(), 0);
              interface_data[side].resize(interface_dofs[side].size(), 0);
            }
        }

    // Extra for projections from mortar to fine grid and RHS assembly
    Quadrature<dim - 1> quad;
    quad = QGauss<dim - 1>(qdegree);


    ConstraintMatrix  constraints;
    FEFaceValues<dim> fe_face_values(fe,
                                     quad,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    // CG structures and parameters
    std::vector<double> alpha_side(n_faces_per_cell, 0),
      alpha_side_d(n_faces_per_cell, 0), beta_side(n_faces_per_cell, 0),
      beta_side_d(n_faces_per_cell, 0);
    std::vector<double> alpha(2, 0), beta(2, 0);

    std::vector<std::vector<double>> r(n_faces_per_cell), p(n_faces_per_cell);
    std::vector<std::vector<double>> Ap(n_faces_per_cell);

    solve_bar();

    interface_fe_function.reinit(solution_bar);

    if (mortar_flag == 1)
      {
        interface_fe_function_mortar.reinit(solution_bar_mortar);
        project_mortar(P_fine2coarse,
                       dof_handler,
                       solution_bar,
                       quad,
                       constraints,
                       neighbors,
                       dof_handler_mortar,
                       solution_bar_mortar);
      }
    else if (mortar_flag == 2)
      {
        interface_fe_function_mortar.reinit(solution_bar_mortar);
        solution_star_mortar = 0;

        // The computation of multiscale basis must necessarilly be after
        // solve_bar() call, as in solve bar we factorize the system matrix into
        // matrix A and clear the system matrix for the sake of memory. Same for
        // solve_star() calls, they should only appear after the solve_bar()
        compute_multiscale_basis();
        project_mortar(P_fine2coarse,
                       dof_handler,
                       solution_bar,
                       quad,
                       constraints,
                       neighbors,
                       dof_handler_mortar,
                       solution_bar_mortar);

        // Instead of solving subdomain problems we compute the response using
        // basis
        unsigned int j = 0;
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            {
              solution_star_mortar.block(0).sadd(
                1.0,
                interface_fe_function_mortar[interface_dofs[side][i]],
                multiscale_basis[j].block(0));
              j += 1;
            }
      }

    double l0 = 0.0;
    // CG with rhs being 0 and initial guess lambda = 0
    for (unsigned side = 0; side < n_faces_per_cell; ++side)
      if (neighbors[side] >= 0)
        {
          // Something will be here to initialize lambda correctly, right now it
          // is just zero
          Ap[side].resize(interface_dofs[side].size(), 0);
          lambda[side].resize(interface_dofs[side].size(), 0);

          r[side].resize(interface_dofs[side].size(), 0);
          std::vector<double> r_receive_buffer(r[side].size());

          // Right now it is effectively solution_bar - A\lambda (0)
          if (mortar_flag)
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              {
                r[side][i] = get_normal_direction(side) *
                               solution_bar_mortar[interface_dofs[side][i]] -
                             get_normal_direction(side) * l0;
              }
          else
            for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
              r[side][i] = get_normal_direction(side) *
                             solution_bar[interface_dofs[side][i]] -
                           get_normal_direction(side) * l0;


          MPI_Send(&r[side][0],
                   r[side].size(),
                   MPI::DOUBLE,
                   neighbors[side],
                   this_mpi,
                   mpi_communicator);
          MPI_Recv(&r_receive_buffer[0],
                   r_receive_buffer.size(),
                   MPI::DOUBLE,
                   neighbors[side],
                   neighbors[side],
                   mpi_communicator,
                   &mpi_status);

          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            {
              r[side][i] += r_receive_buffer[i];
            }
        }

    p = r;

    double normB    = 0;
    double normRold = 0;

    unsigned int iteration_counter = 0;
    while (iteration_counter < maxiter)
      {
        alpha[0] = 0.0;
        alpha[1] = 0.0;
        beta[0]  = 0.0;
        beta[1]  = 0.0;

        iteration_counter++;
        interface_data = p;

        if (mortar_flag == 1)
          {
            for (unsigned int side = 0; side < n_faces_per_cell; ++side)
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                interface_fe_function_mortar[interface_dofs[side][i]] =
                  interface_data[side][i];

            project_mortar(P_coarse2fine,
                           dof_handler_mortar,
                           interface_fe_function_mortar,
                           quad,
                           constraints,
                           neighbors,
                           dof_handler,
                           interface_fe_function);

            interface_fe_function.block(2) = 0;

            assemble_rhs_star(fe_face_values);
            solve_star();
          }
        else if (mortar_flag == 2)
          {
            solution_star_mortar = 0;

            unsigned int j = 0;
            for (unsigned int side = 0; side < n_faces_per_cell; ++side)
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                {
                  solution_star_mortar.block(0).sadd(1.0,
                                            interface_data[side][i],
                                            multiscale_basis[j].block(0));
                  j += 1;
                }
          }
        else
          {
            for (unsigned int side = 0; side < n_faces_per_cell; ++side)
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                interface_fe_function[interface_dofs[side][i]] =
                  interface_data[side][i];

            interface_fe_function.block(2) = 0;
            assemble_rhs_star(fe_face_values);
            solve_star();
          }

        cg_iteration++;

        if (mortar_flag == 1)
          project_mortar(P_fine2coarse,
                         dof_handler,
                         solution_star,
                         quad,
                         constraints,
                         neighbors,
                         dof_handler_mortar,
                         solution_star_mortar);

        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          if (neighbors[side] >= 0)
            {
              alpha_side[side]   = 0;
              alpha_side_d[side] = 0;
              beta_side[side]    = 0;
              beta_side_d[side]  = 0;

              // Create vector of u\dot n to send
              if (mortar_flag)
                for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                  interface_data_send[side][i] =
                    get_normal_direction(side) *
                    solution_star_mortar[interface_dofs[side][i]];
              else
                for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                  interface_data_send[side][i] =
                    get_normal_direction(side) *
                    solution_star[interface_dofs[side][i]];

              MPI_Send(&interface_data_send[side][0],
                       interface_dofs[side].size(),
                       MPI::DOUBLE,
                       neighbors[side],
                       this_mpi,
                       mpi_communicator);
              MPI_Recv(&interface_data_receive[side][0],
                       interface_dofs[side].size(),
                       MPI::DOUBLE,
                       neighbors[side],
                       neighbors[side],
                       mpi_communicator,
                       &mpi_status);

              // Compute Ap and with it compute alpha
              for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                {
                  Ap[side][i] = -(interface_data_send[side][i] +
                                  interface_data_receive[side][i]);

                  alpha_side[side] += r[side][i] * r[side][i];
                  alpha_side_d[side] += p[side][i] * Ap[side][i];
                }
            }

        // Fancy some lambdas, huh?
        std::for_each(alpha_side.begin(), alpha_side.end(), [&](double n) {
          alpha[0] += n;
        });
        std::for_each(alpha_side_d.begin(), alpha_side_d.end(), [&](double n) {
          alpha[1] += n;
        });
        std::vector<double> alpha_buffer(2, 0);

        MPI_Allreduce(&alpha[0],
                      &alpha_buffer[0],
                      2,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);

        alpha = alpha_buffer;

        if (cg_iteration == 1)
          normB = alpha[0];

        normRold = alpha[0];

        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          if (neighbors[side] >= 0)
            {
              for (unsigned int i = 0; i < interface_data[side].size(); ++i)
                {
                  lambda[side][i] += (alpha[0] * p[side][i]) / alpha[1];
                  r[side][i] -= (alpha[0] * Ap[side][i]) / alpha[1];
                }

              for (unsigned int i = 0; i < interface_data[side].size(); ++i)
                beta_side[side] += r[side][i] * r[side][i];
            }

        pcout << "\r  ..." << cg_iteration
              << " iterations completed, (residual = " << fabs(alpha[0] / normB)
              << ")..." << std::flush;
        // Exit criterion
        if (fabs(alpha[0]) / normB < tolerance)
          {
            pcout << "\n  CG converges in " << cg_iteration << " iterations!\n";
            break;
          }

        std::for_each(beta_side.begin(), beta_side.end(), [&](double n) {
          beta[0] += n;
        });
        double beta_buffer = 0;

        MPI_Allreduce(
          &beta[0], &beta_buffer, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

        beta[0] = beta_buffer;
        beta[1] = normRold;

        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          {
            if (neighbors[side] >= 0)
              for (unsigned int i = 0; i < interface_data[side].size(); ++i)
                p[side][i] = r[side][i] + (beta[0] / beta[1]) * p[side][i];

            interface_data_receive[side].resize(interface_dofs[side].size(), 0);
            interface_data_send[side].resize(interface_dofs[side].size(), 0);

            Ap.resize(n_faces_per_cell);
          }
      }

    if (mortar_flag)
      {
        interface_data = lambda;
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            interface_fe_function_mortar[interface_dofs[side][i]] =
              interface_data[side][i];

        project_mortar(P_coarse2fine,
                       dof_handler_mortar,
                       interface_fe_function_mortar,
                       quad,
                       constraints,
                       neighbors,
                       dof_handler,
                       interface_fe_function);
        interface_fe_function.block(2) = 0;
      }
    else
      {
        interface_data = lambda;
        for (unsigned int side = 0; side < n_faces_per_cell; ++side)
          for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
            interface_fe_function[interface_dofs[side][i]] =
              interface_data[side][i];
      }

    assemble_rhs_star(fe_face_values);
    solve_star();

    solution.reinit(solution_bar);
    solution = solution_bar;
    solution.sadd(1.0, solution_star);

    solution_star.sadd(1.0, solution_bar);
  }



  // MixedElasticityProblemDD::compute_interface_error
  template <int dim>
  double
  MixedElasticityProblemDD<dim>::compute_interface_error(
    Function<dim> &exact_solution)
  {
    system_rhs_star = 0;

    QGauss<dim - 1>   quad(qdegree);
    QGauss<dim - 1>   project_quad(qdegree);
    FEFaceValues<dim> fe_face_values(fe,
                                     quad,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int dofs_per_cell_mortar = fe_mortar.dofs_per_cell;

    Vector<double>                       local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<FEValuesExtractors::Vector> stresses(
      dim, FEValuesExtractors::Vector());
    for (unsigned int d = 0; d < dim; ++d)
      {
        const FEValuesExtractors::Vector tmp_stress(d * dim);
        stresses[d].first_vector_component = tmp_stress.first_vector_component;
      }

    std::vector<std::vector<Tensor<1, dim>>> interface_values(
      dim, std::vector<Tensor<1, dim>>(n_face_q_points));
    std::vector<std::vector<Tensor<1, dim>>> solution_values(
      dim, std::vector<Tensor<1, dim>>(n_face_q_points));
    std::vector<Vector<double>> displacement_values(n_face_q_points,
                                                    Vector<double>(dim));

    // Assemble rhs for star problem with data = u - lambda_H on interfaces
    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        local_rhs = 0;
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int face_n = 0;
             face_n < GeometryInfo<dim>::faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n) &&
              cell->face(face_n)->boundary_id() != 0)
            {
              fe_face_values.reinit(cell, face_n);

              for (unsigned int d_i = 0; d_i < dim; ++d_i)
                fe_face_values[stresses[d_i]].get_function_values(
                  interface_fe_function, interface_values[d_i]);

              exact_solution.vector_value_list(
                fe_face_values.get_quadrature_points(), displacement_values);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    Tensor<2, dim> sigma;
                    Tensor<2, dim> interface_lambda;
                    for (unsigned int d_i = 0; d_i < dim; ++d_i)
                      fe_face_values[stresses[d_i]].get_function_values(
                        interface_fe_function, interface_values[d_i]);

                    Tensor<1, dim> sigma_n =
                      sigma * fe_face_values.normal_vector(q);
                    for (unsigned int d_i = 0; d_i < dim; ++d_i)
                      local_rhs(i) +=
                        fe_face_values[stresses[d_i]].value(i, q) *
                        fe_face_values.normal_vector(q) *
                        (displacement_values[q][d_i] -
                         interface_values[d_i][q] *
                           get_normal_direction(
                             cell->face(face_n)->boundary_id() - 1) *
                           fe_face_values.normal_vector(q)) *
                        fe_face_values.JxW(q);
                  }
            }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          system_rhs_star(local_dof_indices[i]) += local_rhs(i);
      }

    // Solve star problem with data given by p - lambda_h
    solve_star();

    // Project the solution to the mortar space
    ConstraintMatrix constraints;
    project_mortar(P_fine2coarse,
                   dof_handler,
                   solution_star,
                   project_quad,
                   constraints,
                   neighbors,
                   dof_handler_mortar,
                   solution_star_mortar);

    double res = 0;

    FEFaceValues<dim> fe_face_values_mortar(fe_mortar,
                                            quad,
                                            update_values |
                                              update_normal_vectors |
                                              update_quadrature_points |
                                              update_JxW_values);

    // Compute the discrete interface norm
    cell = dof_handler_mortar.begin_active(), endc = dof_handler_mortar.end();
    for (; cell != endc; ++cell)
      {
        for (unsigned int face_n = 0;
             face_n < GeometryInfo<dim>::faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n) &&
              cell->face(face_n)->boundary_id() != 0)
            {
              fe_face_values_mortar.reinit(cell, face_n);

              for (unsigned int d_i = 0; d_i < dim; ++d_i)
                {
                  fe_face_values_mortar[stresses[d_i]].get_function_values(
                    solution_star_mortar, solution_values[d_i]);
                  fe_face_values_mortar[stresses[d_i]].get_function_values(
                    interface_fe_function_mortar, interface_values[d_i]);
                }

              exact_solution.vector_value_list(
                fe_face_values_mortar.get_quadrature_points(),
                displacement_values);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                for (unsigned int d_i = 0; d_i < dim; ++d_i)
                  res += fabs(fe_face_values_mortar.normal_vector(q) *
                              solution_values[d_i][q] *
                              (displacement_values[q][d_i] -
                               fe_face_values_mortar.normal_vector(q) *
                                 interface_values[d_i][q] *
                                 get_normal_direction(
                                   cell->face(face_n)->boundary_id() - 1)) *
                              fe_face_values_mortar.JxW(q));
            }
      }

    return sqrt(res);
  }


  // MixedElasticityProblemDD::compute_errors
  template <int dim>
  void
  MixedElasticityProblemDD<dim>::compute_errors(const unsigned int &cycle)
  {
    TimerOutput::Scope t(computing_timer, "Compute Errors");

    const ComponentSelectFunction<dim> rotation_mask(dim * dim + dim,
                                                     dim * dim + dim +
                                                       0.5 * dim * (dim - 1));
    const ComponentSelectFunction<dim> displacement_mask(
      std::make_pair(dim * dim, dim * dim + dim),
      dim * dim + dim + 0.5 * dim * (dim - 1));
    const ComponentSelectFunction<dim> stress_mask(std::make_pair(0, dim * dim),
                                                   dim * dim + dim +
                                                     0.5 * dim * (dim - 1));
    ExactSolution<dim>                 exact_solution;

    // Vectors to temporarily store cellwise errros
    Vector<double> cellwise_errors(triangulation.n_active_cells());
    Vector<double> cellwise_norms(triangulation.n_active_cells());

    // Vectors to temporarily store cellwise componentwise div errors
    Vector<double> cellwise_div_errors(triangulation.n_active_cells());
    Vector<double> cellwise_div_norms(triangulation.n_active_cells());

    // Define quadrature points to compute errors at
    QGauss<dim> quadrature(degree + 5);

    // This is used to show superconvergence at midcells
    QGauss<dim> quadrature_super(1);

    // Since we want to compute the relative norm
    BlockVector<double> zerozeros(1, solution_star.size());

    // Rotation error and norm
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &rotation_mask);
    const double p_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference(dof_handler,
                                      zerozeros,
                                      exact_solution,
                                      cellwise_norms,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &rotation_mask);
    const double p_l2_norm = cellwise_norms.l2_norm();

    // Displacement error and norm
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &displacement_mask);
    const double u_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference(dof_handler,
                                      zerozeros,
                                      exact_solution,
                                      cellwise_norms,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &displacement_mask);
    const double u_l2_norm = cellwise_norms.l2_norm();

    // Displacement error and norm at midcells
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature_super,
                                      VectorTools::L2_norm,
                                      &displacement_mask);
    const double u_l2_mid_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference(dof_handler,
                                      zerozeros,
                                      exact_solution,
                                      cellwise_norms,
                                      quadrature_super,
                                      VectorTools::L2_norm,
                                      &displacement_mask);
    const double u_l2_mid_norm = cellwise_norms.l2_norm();

    // Stress L2 error and norm
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &stress_mask);
    const double s_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference(dof_handler,
                                      zerozeros,
                                      exact_solution,
                                      cellwise_norms,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &stress_mask);

    const double s_l2_norm = cellwise_norms.l2_norm();

    // Stress Hdiv seminorm
    cellwise_errors = 0;
    cellwise_norms  = 0;
    for (int i = 0; i < dim; ++i)
      {
        const ComponentSelectFunction<dim> stress_component_mask(
          std::make_pair(i * dim, (i + 1) * dim),
          dim * dim + dim + 0.5 * dim * (dim - 1));

        VectorTools::integrate_difference(dof_handler,
                                          solution,
                                          exact_solution,
                                          cellwise_div_errors,
                                          quadrature,
                                          VectorTools::Hdiv_seminorm,
                                          &stress_component_mask);
        cellwise_errors += cellwise_div_errors;

        VectorTools::integrate_difference(dof_handler,
                                          zerozeros,
                                          exact_solution,
                                          cellwise_div_norms,
                                          quadrature,
                                          VectorTools::Hdiv_seminorm,
                                          &stress_component_mask);
        cellwise_norms += cellwise_div_norms;
      }

    const double s_hd_error = cellwise_errors.l2_norm();
    const double s_hd_norm  = cellwise_norms.l2_norm();

    double l_int_error = 1, l_int_norm = 1;

    if (mortar_flag)
      {
        DisplacementBoundaryValues<dim> displ_solution;
        l_int_error = compute_interface_error(displ_solution);

        interface_fe_function        = 0;
        interface_fe_function_mortar = 0;
        l_int_norm                   = compute_interface_error(displ_solution);
      }

    double send_buf_num[6] = {s_l2_error,
                              s_hd_error,
                              u_l2_error,
                              u_l2_mid_error,
                              p_l2_error,
                              l_int_error};
    double send_buf_den[6] = {
      s_l2_norm, s_hd_norm, u_l2_norm, u_l2_mid_norm, p_l2_norm, l_int_norm};

    double recv_buf_num[6] = {0, 0, 0, 0, 0, 0};
    double recv_buf_den[6] = {0, 0, 0, 0, 0, 0};

    MPI_Reduce(&send_buf_num[0],
               &recv_buf_num[0],
               6,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               mpi_communicator);
    MPI_Reduce(&send_buf_den[0],
               &recv_buf_den[0],
               6,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               mpi_communicator);

    for (unsigned int i = 0; i < 6; ++i)
      recv_buf_num[i] = recv_buf_num[i] / recv_buf_den[i];

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        convergence_table.add_value("cycle", cycle);
        convergence_table.add_value("# CG", cg_iteration);
        convergence_table.add_value("Stress,L2", recv_buf_num[0]);
        convergence_table.add_value("Stress,Hdiv", recv_buf_num[1]);
        convergence_table.add_value("Displ,L2", recv_buf_num[2]);
        convergence_table.add_value("Displ,L2mid", recv_buf_num[3]);
        convergence_table.add_value("Rotat,L2", recv_buf_num[4]);

        if (mortar_flag)
          convergence_table.add_value("Lambda,Int", recv_buf_num[6]);
      }
  }


  // MixedElasticityProblemDD::output_results
  template <int dim>
  void
  MixedElasticityProblemDD<dim>::output_results(const unsigned int &cycle,
                                                const unsigned int &refine,
                                                const std::string & name)
  {
    TimerOutput::Scope t(computing_timer, "Output results");
    unsigned int       n_processes =
      Utilities::MPI::n_mpi_processes(mpi_communicator);
    unsigned int this_mpi = Utilities::MPI::this_mpi_process(mpi_communicator);


    std::vector<std::string> solution_names;
    std::string              rhs_name = "rhs";

    switch (dim)
      {
        case 2:
          solution_names.push_back("s11");
          solution_names.push_back("s12");
          solution_names.push_back("s21");
          solution_names.push_back("s22");
          solution_names.push_back("u");
          solution_names.push_back("v");
          solution_names.push_back("p");
          break;

        case 3:
          solution_names.push_back("s11");
          solution_names.push_back("s12");
          solution_names.push_back("s13");
          solution_names.push_back("s21");
          solution_names.push_back("s22");
          solution_names.push_back("s23");
          solution_names.push_back("s31");
          solution_names.push_back("s32");
          solution_names.push_back("s33");
          solution_names.push_back("u");
          solution_names.push_back("v");
          solution_names.push_back("w");
          solution_names.push_back("p1");
          solution_names.push_back("p2");
          solution_names.push_back("p3");
          break;

        default:
          Assert(false, ExcNotImplemented());
      }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim * dim + dim + 0.5 * dim * (dim - 1) - 1,
        DataComponentInterpretation::component_is_part_of_vector);

    switch (dim)
      {
        case 2:
          data_component_interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);
          break;

        case 3:
          data_component_interpretation.push_back(
            DataComponentInterpretation::component_is_part_of_vector);
          break;

        default:
          Assert(false, ExcNotImplemented());
          break;
      }

    DataOut<dim> data_out_star;
    data_out_star.add_data_vector(dof_handler,
                                  solution,
                                  solution_names,
                                  data_component_interpretation);
    data_out_star.build_patches(degree);
    std::ofstream output("solution" + name + "_p" +
                         Utilities::to_string(this_mpi) + "-" +
                         Utilities::to_string(cycle) + ".vtu");
    data_out_star.write_vtu(output);


    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        convergence_table.set_precision("Stress,L2", 3);
        convergence_table.set_precision("Stress,Hdiv", 3);
        convergence_table.set_precision("Displ,L2", 3);
        convergence_table.set_precision("Displ,L2mid", 3);
        convergence_table.set_precision("Rotat,L2", 3);

        convergence_table.set_scientific("Stress,L2", true);
        convergence_table.set_scientific("Stress,Hdiv", true);
        convergence_table.set_scientific("Displ,L2", true);
        convergence_table.set_scientific("Displ,L2mid", true);
        convergence_table.set_scientific("Rotat,L2", true);

        convergence_table.set_tex_caption("# CG", "\\# cg");
        convergence_table.set_tex_caption(
          "Stress,L2", "$ \\|\\sigma - \\sigma_h\\|_{L^2} $");
        convergence_table.set_tex_caption(
          "Stress,Hdiv", "$ \\|\\nabla\\cdot(\\sigma - \\sigma_h)\\|_{L^2} $");
        convergence_table.set_tex_caption("Displ,L2",
                                          "$ \\|u - u_h\\|_{L^2} $");
        convergence_table.set_tex_caption("Displ,L2mid",
                                          "$ \\|Qu - u_h\\|_{L^2} $");
        convergence_table.set_tex_caption("Rotat,L2",
                                          "$ \\|p - p_h\\|_{L^2} $");

        convergence_table.evaluate_convergence_rates(
          "# CG", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "Stress,L2", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "Stress,Hdiv", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "Displ,L2", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "Displ,L2mid", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "Rotat,L2", ConvergenceTable::reduction_rate_log2);

        if (mortar_flag)
          {
            convergence_table.set_precision("Lambda,Int", 3);
            convergence_table.set_scientific("Lambda,Int", true);
            convergence_table.set_tex_caption("Lambda,Int",
                                              "$ \\|p - \\lambda_H\\|_{d_H} $");
            convergence_table.evaluate_convergence_rates(
              "Lambda,Int", ConvergenceTable::reduction_rate_log2);
          }

        if (cycle == refine - 1)
          {
            std::ofstream error_table_file(
              "error" + name +
              std::to_string(
                Utilities::MPI::n_mpi_processes(mpi_communicator)) +
              "domains.tex");
            convergence_table.write_text(std::cout);
            convergence_table.write_tex(error_table_file);
          }
      }
  }


  // MixedElasticityProblemDD::run
  template <int dim>
  void
  MixedElasticityProblemDD<dim>::run(
    const unsigned int                            refine,
    const std::vector<std::vector<unsigned int>> &reps,
    double                                        tol,
    std::string                                   name,
    unsigned int                                  maxiter,
    unsigned int                                  quad_degree)
  {
    tolerance = tol;
    qdegree   = quad_degree;

    const unsigned int this_mpi =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(mpi_communicator);

    Assert(reps[0].size() == dim, ExcDimensionMismatch(reps[0].size(), dim));

    if (mortar_flag)
      {
        Assert(n_processes > 1,
               ExcMessage("Mortar MFEM is impossible with 1 subdomain"));
        Assert(reps.size() >= n_processes + 1,
               ExcMessage("Some of the mesh parameters were not provided"));
      }

    for (unsigned int cycle = 0; cycle < refine; ++cycle)
      {
        cg_iteration = 0;
        interface_dofs.clear();

        if (cycle == 0)
          {
            // Partitioning into subdomains (simple bricks)
            find_divisors<dim>(n_processes, n_domains);

            // Dimensions of the domain (unit hypercube)
            std::vector<double> subdomain_dimensions(dim);
            for (unsigned int d = 0; d < dim; ++d)
              subdomain_dimensions[d] = 1.0 / double(n_domains[d]);

            get_subdomain_coordinates(
              this_mpi, n_domains, subdomain_dimensions, p1, p2);

            if (mortar_flag)
              GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                        reps[this_mpi],
                                                        p1,
                                                        p2);
            else
              GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                        reps[0],
                                                        p1,
                                                        p2);

            if (mortar_flag)
              GridGenerator::subdivided_hyper_rectangle(triangulation_mortar,
                                                        reps[n_processes],
                                                        p1,
                                                        p2);
          }
        else
          {
            if (mortar_flag == 0)
              triangulation.refine_global(1);
            else if (mortar_degree <= 2)
              triangulation.refine_global(1);
            else if (mortar_degree > 2)
              triangulation.refine_global(2);

            if (mortar_flag)
              {
                triangulation_mortar.refine_global(1);
                pcout << "Mortar mesh has "
                      << triangulation_mortar.n_active_cells() << " cells"
                      << std::endl;
              }
          }

        pcout << "Making grid and DOFs..."
              << "\n";
        make_grid_and_dofs();
        pcout << "Assembling system..."
              << "\n";
        assemble_system();

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
          {
            solve_bar();
            solution = solution_bar;

            compute_errors(cycle);

            computing_timer.print_summary();
            computing_timer.reset();

            output_results(cycle, refine, name);
          }
        else
          {
            get_interface_dofs();
            pcout << "Starting CG iterations..."
                  << "\n";
            local_cg(maxiter);
            compute_errors(cycle);

            computing_timer.print_summary();
            computing_timer.reset();

            output_results(cycle, refine, name);
          }
      }

    triangulation.clear();
    dof_handler.clear();
    convergence_table.clear();
    faces_on_interface.clear();
    faces_on_interface_mortar.clear();
    interface_dofs.clear();
    interface_fe_function = 0;

    if (mortar_flag)
      {
        triangulation_mortar.clear();
        P_fine2coarse.reset();
        P_coarse2fine.reset();
      }
    dof_handler_mortar.clear();
  }

  template class MixedElasticityProblemDD<2>;
  template class MixedElasticityProblemDD<3>;
} // namespace dd_elasticity