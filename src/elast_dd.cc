/* ---------------------------------------------------------------------
 * This program implements multiscale mortar mixed finite element
 * method for linear elasticity model. The elasticity system is
 * written in a three-field form, with stress, displacement and
 * rotation as variables. The domain decomposition procedure
 * is then obtained by matching the normal components of stresses
 * across the interface.
 *
 * This implementation allows for non-matching grids by utilizing
 * the mortar finite element space on the interface. To speed things
 * up a little, the multiscale stress basis is also available for
 * the cases when the mortar grid is much coarser than the subdomain
 * ones.
 * ---------------------------------------------------------------------
 *
 * Author: Eldar Khattatov, University of Pittsburgh, 2016 - 2017
 */

// Utilities, data, etc.
#include <map>
#include "../inc/elasticity_mfedd.h"

static void show_usage(std::string name, dealii::ConditionalOStream &pcout)
{
    pcout << "Usage: " << name << " <option(s)> \n"
          << "Options:\n"
          << "\t-h,--help\t\tShow the usage format\n"
          << "\t-m,--mortar MORTAR_DEGREE\tSpecify the mortar degree (use 0 for no mortar)\n"
          << "\t-r,--refine REFINEMENT_CYCLES\tSpecify the number of refinements\n"
          << "\t-d,--dim DIMENSION\tSpecify the physical dimension\n"
          << "\t-o,--order ORDER\tSpecify the finite element order"
          << std::endl;
}

static bool check_arguments(int argc, char *argv[], dealii::ConditionalOStream &pcout)
{
    if (argc != 9 && (std::string(argv[1]) != "--help" and std::string(argv[1]) != "-h"))
      { 
        show_usage(argv[0], pcout);
        return 1;
      }
    else
	return 0;
}

// Main function is simple here
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace dd_elasticity;

      MultithreadInfo::set_thread_limit(4);
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    
      ConditionalOStream pcerr(std::cerr, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
      ConditionalOStream pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      if (check_arguments(argc, argv, pcerr))
	return 1;

      std::map<std::string,int> options;
      for (unsigned int i=1; i<argc; i+=2)
        options[std::string(argv[i])] = std::stoi(argv[i+1]);

      unsigned int mortar_degree;
      unsigned int refinement_cycles;
      unsigned int dimension;
      unsigned int order;

      for (const auto &el : options)
        {
          if (el.first == "--mortar" || el.first == "-m")
              mortar_degree = el.second;
          else if (el.first == "--refine" || el.first == "-r")
              refinement_cycles = el.second;
          else if (el.first == "--dim" || el.first == "-d")
              dimension = el.second;
          else if (el.first == "--order" || el.first == "-o")
              order = el.second;
          else if (el.first == "--help" || el.first == "-h")
              show_usage(argv[0],pcout);
          else
            {
              std::cerr << "Unsupported option " << el.first << std::endl;
              return 1;
            }
        }

      pcout << std::boolalpha;
      pcout << "Running mixed elasticity problem with the following parameters:\n"
                << "\t FE order = " << order << std::endl
                << "\t dimension = " << dimension << std::endl
                << "\t with mortars = " << bool(mortar_degree) << ", of order " << mortar_degree << std::endl
                << "\t number of refinements = " << refinement_cycles << std::endl
                << "\t with MSB = " << (mortar_degree > 2) << std::endl
		<< "==============================================================\n";

      if (dimension == 2)
        {
          std::vector<std::string> names {"M0_2d", "M1_2d", "M1_2d", "M2_2d"};

          //TODO: generalize for any nprocs
          std::vector<std::vector<unsigned int>> mesh_m2d(8);
          mesh_m2d[0] = {2, 2};
          mesh_m2d[1] = {1, 1};
          mesh_m2d[2] = {4, 4};
          mesh_m2d[3] = {3, 3};
          mesh_m2d[4] = {2, 1};
          mesh_m2d[5] = {1, 1};
          mesh_m2d[6] = {2, 2};
          mesh_m2d[7] = {1, 2};

          if (mortar_degree == 0)
            {
              MixedElasticityProblemDD<2> no_mortars(order);
              no_mortars.run(refinement_cycles, mesh_m2d, 1.e-14, names[mortar_degree], 500);
            }
          else
            {
              MixedElasticityProblemDD<2> with_mortars(order, (mortar_degree <= 2) ? 1 : 2, mortar_degree);
              with_mortars.run (refinement_cycles, mesh_m2d, 1.e-14, names[mortar_degree], 500, 51);
            }
        }
      else
        {
          std::vector<std::string> names {"M0_3d", "M1_3d", "M1_3d", "M2_3d"};

          //TODO: generalize for any nprocs
          std::vector<std::vector<unsigned int>> mesh_m3d(16);
          mesh_m3d[0] = {2, 2, 2};
          mesh_m3d[1] = {3, 3, 3};
          mesh_m3d[2] = {3, 3, 3};
          mesh_m3d[3] = {2, 2, 2};
          mesh_m3d[4] = {3, 3, 3};
          mesh_m3d[5] = {2, 2, 2};
          mesh_m3d[6] = {2, 2, 2};
          mesh_m3d[7] = {3, 3, 3};
          for (unsigned int i=8; i<16; ++i)
            mesh_m3d[i] = {1, 1, 1};

          if (mortar_degree == 0)
          {
            MixedElasticityProblemDD<3> no_mortars(order);
            no_mortars.run(refinement_cycles, mesh_m3d, 1.e-14, names[mortar_degree], 500);
          }
          else
          {
            MixedElasticityProblemDD<3> with_mortars(order, (mortar_degree <= 2) ? 1 : 2, mortar_degree);
            with_mortars.run (refinement_cycles, mesh_m3d, 1.e-14, names[mortar_degree], 500, 71);
          }
        }      
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  return 0;
}
