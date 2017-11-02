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
#include "../inc/elasticity_mfedd.h"

// Main function is simple here
int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace dd_elasticity;

        MultithreadInfo::set_thread_limit(4);
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        // Mortar mesh parameters (non-matching checkerboard)
        std::vector<std::vector<unsigned int>> mesh_m2d(5);
        mesh_m2d[0] = {2,2};
        mesh_m2d[1] = {3,3};
        mesh_m2d[2] = {3,3};
        mesh_m2d[3] = {2,2};
        mesh_m2d[4] = {1,1};

        std::vector<std::vector<unsigned int>> mesh_m3d(9);
        mesh_m3d[0] = {2,2,2};
        mesh_m3d[1] = {3,3,3};
        mesh_m3d[2] = {3,3,3};
        mesh_m3d[3] = {2,2,2};
        mesh_m3d[4] = {3,3,3};
        mesh_m3d[5] = {2,2,2};
        mesh_m3d[6] = {2,2,2};
        mesh_m3d[7] = {3,3,3};
        mesh_m3d[8] = {1,1,1};

        MixedElasticityProblemDD<2> no_mortars(1);
//        MixedElasticityProblemDD<2> lin_mortars(1,1,1);
//        MixedElasticityProblemDD<2> quad_mortars(1,1,2);
//        MixedElasticityProblemDD<2> cubic_mortars(1,2,3);

        std::string name1("M0");
        std::string name2("M1");
        std::string name3("M2");
        std::string name4("M3");

//        MixedElasticityProblemDD<3> no_mortars_3d(1);
//        MixedElasticityProblemDD<3> lin_mortars3d(1,1,1);
//        MixedElasticityProblemDD<3> quad_mortars_2_3d(1,1,1);
//        MixedElasticityProblemDD<3> cubic_mortars_1_3d(1,1,2);
//        MixedElasticityProblemDD<3> cubic_mortars_2_3d(1,1,2);

//        std::string name13("M0_3d");
//        std::string name23("M1_3d");
//        std::string name33("M2_3d");

        // 2d cases
        no_mortars.run (4, mesh_m2d, 1.e-14, name1, 500);
//        lin_mortars.run (7, mesh_m2d, 1.e-14, name2, 500, 51);
//        quad_mortars.run (7, mesh_m2d, 1.e-14, name3, 500, 61);
//        cubic_mortars.run (4, mesh_m2d, 1.e-14, name4, 500, 71);

//        // 3d cases
//        no_mortars_3d.run (2, mesh_matching3d, 1.e-10, name13, 500);
//        lin_mortars3d.run (2, mesh_m3d, 1.e-12, name23, 500, 15);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
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
        std::cerr << std::endl << std::endl
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