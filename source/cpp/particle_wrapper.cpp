// Author: Jakob Maljaars and Chris Richardson
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <dolfin/fem/Form.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/MeshFunction.h>

#include "adddelete.h"
#include "advect_particles.h"
#include "formutils.h"
#include "l2projection.h"
#include "particles.h"
#include "pdestaticcondensation.h"
#include "stokesstaticcondensation.h"

PYBIND11_MODULE(particle_wrapper, m)
{
  m.doc() = "example";

  py::class_<dolfin::particles>(m, "particles")
      .def(py::init<
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>,
           const std::vector<unsigned int>&, const dolfin::Mesh&>())
      .def("interpolate", &dolfin::particles::interpolate)
      .def("increment", (void (dolfin::particles::*)(const dolfin::Function&,
                                                     const dolfin::Function&,
                                                     const std::size_t))
                            & dolfin::particles::increment)
      .def("increment",
           (void (dolfin::particles::*)(
               const dolfin::Function&, const dolfin::Function&,
               Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>,
               const double, const std::size_t))
               & dolfin::particles::increment)
      .def("positions", &dolfin::particles::positions)
      .def("push_particle", &dolfin::particles::push_particle)
      .def("get_property", &dolfin::particles::get_property)
      .def("property", &dolfin::particles::property)
      .def("set_property", &dolfin::particles::set_property)
      .def("num_properties", &dolfin::particles::num_properties)
      .def("num_cell_particles", &dolfin::particles::num_cell_particles)
      .def("mesh", &dolfin::particles::mesh)
      .def("ptemplate", &dolfin::particles::ptemplate)
      .def("add_particle", &dolfin::particles::add_particle)
      .def("delete_particle", &dolfin::particles::delete_particle)
      .def("set_empty_cell_default_values", &dolfin::particles::set_empty_cell_default_values)
      .def("relocate",
           (void (dolfin::particles::*)()) & dolfin::particles::relocate);

  py::class_<dolfin::advect_particles>(m, "advect_particles")
      .def(py::init<dolfin::particles&, dolfin::FunctionSpace&,
                    std::function<const dolfin::Function&(int, double)>, const std::string>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const std::string,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def("do_step", &dolfin::advect_particles::do_step)
      .def("do_stepLPT", &dolfin::advect_particles::do_stepLPT) 
           //std::array<std::array<float,6>,1> LPTParameters)
      //.def("do_step_LP", &dolfin::advect_particles::do_step_LP)
      .def("update_facets_info", &dolfin::advect_particles::update_facets_info);

  py::class_<dolfin::advect_rk2>(m, "advect_rk2")
      .def(py::init<dolfin::particles&, dolfin::FunctionSpace&,
                    std::function<const dolfin::Function&(int, double)>, const std::string>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const std::string,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def("do_step", &dolfin::advect_rk2::do_step)
      .def("update_facets_info", &dolfin::advect_rk2::update_facets_info);

  py::class_<dolfin::advect_rk3>(m, "advect_rk3")
      .def(py::init<dolfin::particles&, dolfin::FunctionSpace&,
                    std::function<const dolfin::Function&(int, double)>, const std::string>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const std::string,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def("do_step", &dolfin::advect_rk3::do_step)
      .def("update_facets_info", &dolfin::advect_rk3::update_facets_info);

  py::class_<dolfin::advect_rk4>(m, "advect_rk4")
      .def(py::init<dolfin::particles&, dolfin::FunctionSpace&,
                    std::function<const dolfin::Function&(int, double)>, const std::string>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const std::string,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def(py::init<
           dolfin::particles&, dolfin::FunctionSpace&,
           std::function<const dolfin::Function&(int, double)>, const dolfin::MeshFunction<std::size_t>&,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>,
           Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>>())
      .def("do_step", &dolfin::advect_rk4::do_step)
      .def("update_facets_info", &dolfin::advect_rk4::update_facets_info);

  py::class_<dolfin::l2projection>(m, "l2projection")
      .def(py::init<dolfin::particles&, dolfin::FunctionSpace&,
                    const std::size_t>())
      .def("project", (void (dolfin::l2projection::*)(dolfin::Function&))
                          & dolfin::l2projection::project)
      .def("project", (void (dolfin::l2projection::*)(
                          dolfin::Function&, const double, const double))
                          & dolfin::l2projection::project)
      .def("project_cg", &dolfin::l2projection::project_cg);

  py::class_<dolfin::StokesStaticCondensation>(m, "StokesStaticCondensation")
      .def(py::init<std::shared_ptr<const dolfin::Mesh>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>>())
      .def(py::init<std::shared_ptr<const dolfin::Mesh>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::vector<std::shared_ptr<const dolfin::DirichletBC>>>())
      .def(py::init<std::shared_ptr<const dolfin::Mesh>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>>())
      .def(py::init<std::shared_ptr<const dolfin::Mesh>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>,
                    std::vector<std::shared_ptr<const dolfin::DirichletBC>>>())
      .def("assemble_global",
           &dolfin::StokesStaticCondensation::assemble_global)
      .def("assemble_global_lhs",
           &dolfin::StokesStaticCondensation::assemble_global_lhs)
      .def("assemble_global_rhs",
           &dolfin::StokesStaticCondensation::assemble_global_rhs)
      .def("assemble_global_system",
           &dolfin::StokesStaticCondensation::assemble_global_system)
      .def("apply_boundary", &dolfin::StokesStaticCondensation::apply_boundary)
      .def("solve_problem",
           (void (dolfin::StokesStaticCondensation::*)(
               dolfin::Function&, dolfin::Function&, const std::string,
               const std::string))
               & dolfin::StokesStaticCondensation::solve_problem)
      .def("get_global_lhs_matrix", &dolfin::StokesStaticCondensation::get_global_lhs_matrix)
      .def("get_global_rhs_vector", &dolfin::StokesStaticCondensation::get_global_rhs_vector)
      .def("backsubstitute", &dolfin::StokesStaticCondensation::backsubstitute);

  py::class_<dolfin::PDEStaticCondensation>(m, "PDEStaticCondensation")
      .def(
          py::init<std::shared_ptr<const dolfin::Mesh>, dolfin::particles&, std::shared_ptr<const dolfin::Form>,
                   std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                   std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                   std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                   std::shared_ptr<const dolfin::Form>, const std::size_t>())
      .def(py::init<std::shared_ptr<const dolfin::Mesh>, dolfin::particles&,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::shared_ptr<const dolfin::Form>, std::shared_ptr<const dolfin::Form>,
                    std::vector<std::shared_ptr<const dolfin::DirichletBC>>,
                    const std::size_t>())
      .def("assemble", &dolfin::PDEStaticCondensation::assemble)
      .def("assemble_state_rhs",
           &dolfin::PDEStaticCondensation::assemble_state_rhs)
      .def("solve_problem", (void (dolfin::PDEStaticCondensation::*)(
                                dolfin::Function&, dolfin::Function&,
                                const std::string, const std::string))
                                & dolfin::PDEStaticCondensation::solve_problem)
      .def("solve_problem",
           (void (dolfin::PDEStaticCondensation::*)(
               dolfin::Function&, dolfin::Function&, dolfin::Function&,
               const std::string, const std::string))
               & dolfin::PDEStaticCondensation::solve_problem)
      .def("apply_boundary", &dolfin::PDEStaticCondensation::apply_boundary);

  py::class_<dolfin::AddDelete>(m, "AddDelete")
      .def(py::init<dolfin::particles&, std::size_t, std::size_t,
                    std::vector<std::shared_ptr<const dolfin::Function>>>())
      .def(py::init<dolfin::particles&, std::size_t, std::size_t,
                    std::vector<std::shared_ptr<const dolfin::Function>>,
                    std::vector<std::size_t>, std::vector<double>>())
      .def("do_sweep", &dolfin::AddDelete::do_sweep)
      .def("do_sweep_weighted", &dolfin::AddDelete::do_sweep_weighted)
      .def("do_sweep_failsafe", &dolfin::AddDelete::do_sweep_failsafe);
}
