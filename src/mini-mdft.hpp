#ifndef MINI_MDFT_HPP
#define MINI_MDFT_HPP

#include <memory>
#include "MDFT_System.hpp"
#include "MDFT_Grid.hpp"
#include "MDFT_Solute.hpp"
#include "MDFT_Thermo.hpp"
#include "MDFT_Solvent.hpp"
#include "MDFT_Energy.hpp"
#include "MDFT_OrientationProjectionTransform.hpp"
#include "MDFT_Convolution.hpp"
#include "IO/MDFT_Commandline_Utils.hpp"

namespace MDFT {
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
class Solver {
    private:
    using SettingsType    = Settings<ScalarType>;
    using SpatialGridType = SpatialGrid<ExecutionSpace, ScalarType>;
    using AngularGridType = AngularGrid<ExecutionSpace, ScalarType>;
    using SoluteType      = Solute<ExecutionSpace, ScalarType>;
    using ThermoType      = Thermo<ScalarType>;
    using SolventsType     = Solvents<ExecutionSpace, ScalarType>;
    using EnergyType      = Energy<ScalarType>;
    using OPMapType       = OrientationProjectionMap<ExecutionSpace, ScalarType>;
    using OPTransformType = OrientationProjectionTransform<ExecutionSpace, ScalarType>;
    using ConvolutionType = Convolution<ExecutionSpace, ScalarType>;

    using View3DType = Kokkos::View<ScalarType***, ExecutionSpace>;
    using View4DType = Kokkos::View<ScalarType****, ExecutionSpace>;
    using View5DType = Kokkos::View<ScalarType*****, ExecutionSpace>;

    using ComplexView2DType = Kokkos::View<Kokkos::complex<ScalarType>**, ExecutionSpace>;
    using ComplexView4DType = Kokkos::View<Kokkos::complex<ScalarType>****, ExecutionSpace>;

    ExecutionSpace m_exec_space;

    // System settings
    std::unique_ptr<SettingsType> m_settings;
    std::unique_ptr<SpatialGridType> m_spatial_grid;
    std::unique_ptr<AngularGridType> m_angular_grid;
    std::unique_ptr<SoluteType> m_solute;
    std::unique_ptr<ThermoType> m_thermo;
    std::unique_ptr<SolventsType> m_solvents;
    std::unique_ptr<OPMapType> m_op_map;
    std::unique_ptr<OPTransformType> m_op_transform;
    std::unique_ptr<ConvolutionType> m_conv;

    // Functional to minimize
    ScalarType m_f;
    View4DType m_df;
    EnergyType m_energy;

    View4DType m_delta_rho;
    View4DType m_vexc;
    ComplexView4DType m_delta_rho_p;

        //Grid grid;
    public:
        
        explicit Solver(const ExecutionSpace& exec_space) : m_exec_space(exec_space) {}
        Solver() = delete;
        ~Solver() = default;

        void initialize(int *argc, char ***argv) {
          // Load args
          auto kwargs = MDFT::IO::parse_args(*argc, *argv);
          std::string input_file = MDFT::IO::get_arg(kwargs, "filename", "dft2.json");
          std::string solute_filename = MDFT::IO::get_arg(kwargs, "solute", "solute.json");

          m_settings = std::make_unique<SettingsType>(input_file);

          // Initialize grids
          init_grid(*m_settings, m_spatial_grid, m_angular_grid);

          // Initialize solute
          m_solute = std::make_unique<SoluteType>(*m_spatial_grid, *m_settings, solute_filename);

          // Initialize thermo
          m_thermo = std::make_unique<ThermoType>(*m_settings);

          // Initialize solvents
          m_solvents = std::make_unique<SolventsType>(*m_spatial_grid, *m_angular_grid, *m_settings, *m_thermo);

          // Initialize orientation projection map
          m_op_map = std::make_unique<OPMapType>(*m_angular_grid);

          // Initialize orientation projection transform
          m_op_transform = std::make_unique<OPTransformType>(*m_spatial_grid, *m_angular_grid, *m_op_map);

          // Initialize convolution
          m_conv = std::make_unique<ConvolutionType>(*m_spatial_grid, *m_angular_grid, *m_op_map);
        }
        void run(){
          energy_cproj_mrso(m_energy.m_exc_cproj, m_df);
          m_f += m_energy.m_exc_cproj;
          m_energy.m_tot = m_f;
        }
        void finalize(){}

    private:
      void energy_cproj_mrso(ScalarType& ff, View4DType& df) {
        auto solvent = m_solvents->m_solvents.at(0);
        
        // 1 get Δρ(r,ω)
        // (nx, ny, nz, theta * phi * psi) -> (nx * ny * nz, theta, phi, psi)
        get_delta_rho(m_exec_space, solvent.m_xi, m_delta_rho, solvent.m_rho0);

        // 2. projection
        // (nx * ny * nz, theta, phi, psi) -> (nm * nmup * nmu, nx, ny, nz)
        m_op_transform->angl2proj(m_delta_rho, m_delta_rho_p);

        // 3. FFT3D-C2C
        // 4. rotate to q frame
        // 5. OZ
        // 6. rotate back to fixed frame
        // 7. FFT3D-C2C
        // (nm * nmup * nmu, nx, ny, nz) -> (nm * nmup * nmu, nx, ny, nz)
        m_conv->execute(m_delta_rho_p);

        // 8. gather projections
        // (np, nx, ny, nz) -> (nx * ny * nz, theta, phi, psi)
        m_op_transform->proj2angl(m_delta_rho_p, m_vexc);
        
        // (nx, ny, nz, theta * phi * psi) -> (nx * ny * nz, theta, phi, psi)
        get_delta_f(m_exec_space, solvent.m_xi, m_vexc, m_angular_grid->m_w, df, ff, solvent.m_rho0, m_solvents->m_prefactor); 
      }
};  // class Solver 
    
}



#endif // MINI_MDFT_HPP
