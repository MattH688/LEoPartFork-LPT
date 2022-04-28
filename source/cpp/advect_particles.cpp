// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Vertex.h>
//#include "vtkMath.h"

#include "advect_particles.h"
#include "utils.h"

// #include <math.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
advect_particles::advect_particles(particles& P, FunctionSpace& U,
                                   std::function<const Function&(int, double)> uhi, const std::string type1)
    : _P(&P), uh(uhi), _element(U.element())
{
  // Following types are distinguished:
  // "open"       --> open boundary
  // "periodic"   --> periodic bc (additional info on extent required)
  // "closed"     --> closed boundary

  // This constructor cant take periodic nor bounded:
  assert(type1 != "periodic");
  assert(type1 != "bounded");

  // Set facet info
  update_facets_info();

  // Set all external facets to type1
  set_bfacets(type1);

  // Set some other useful info
  _space_dimension = _element->space_dimension();
  _value_size_loc = 1;
  for (std::size_t i = 0; i < _element->value_rank(); i++)
    _value_size_loc *= _element->value_dimension(i);

  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
// Using delegate constructors here
advect_particles::advect_particles(
    particles& P, FunctionSpace& U, std::function<const Function&(int, double)> uhi, const std::string type1,
    Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits)
    : advect_particles::advect_particles(P, U, uhi, type1)
{
  std::size_t gdim = _P->mesh()->geometry().dim();

  // Then the only thing to do: check if type1 was "periodic"
  if (type1 == "periodic")
  {
    // Check if it has the right size, always has to come in pairs
    // TODO: do provided values make sense?
    if ((pbc_limits.size() % (gdim * 4)) != 0)
    {
      dolfin_error("advect_particles.cpp::advect_particles",
                   "construct periodic boundary information",
                   "Incorrect shape of pbc_limits provided?");
    }

    std::size_t num_rows = pbc_limits.size() / (gdim * 2);
    for (std::size_t i = 0; i < num_rows; i++)
    {
      std::vector<double> pbc_helper(gdim * 2);
      for (std::size_t j = 0; j < gdim * 2; j++)
        pbc_helper[j] = pbc_limits[i * gdim * 2 + j];

      pbc_lims.push_back(pbc_helper);
    }
    pbc_active = true;
  }
  else if (type1 == "bounded")
  {
    // Check if it has the right size. [xmin, xmax, ymin, ymax, zmin, zmax]
    if ((pbc_limits.size() % (2 * gdim)) != 0)
    {
      dolfin_error("advect_particles.cpp::advect_particles",
                   "construct periodic boundary information",
                   "Incorrect shape of pbc_limits provided?");
    }

    std::size_t num_rows = pbc_limits.size() / gdim;
    for (std::size_t i = 0; i < num_rows; i++)
    {
      std::vector<double> bounded_domain_lims_helper(2);
      bounded_domain_lims_helper[0] = pbc_limits[2*i];
      bounded_domain_lims_helper[1] = pbc_limits[2*i + 1];

      bounded_domain_lims.push_back(bounded_domain_lims_helper);
    }
    bounded_domain_active = true;
  }
  else
  {
    dolfin_error("advect_particles.cpp::advect_particles",
                 "could not set pbc_lims",
                 "Did you provide limits for a non-periodic BC?");
  }

  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
advect_particles::advect_particles(particles& P, FunctionSpace& U, std::function<const Function&(int, double)> uhi,
                 const MeshFunction<std::size_t>& mesh_func)
    : _P(&P), uh(uhi), _element(U.element())
{
  // Confirm that mesh_func contains no periodic boundary values (3)
  if (std::find(mesh_func.values(), mesh_func.values()+mesh_func.size(), 3)
        != mesh_func.values()+mesh_func.size())
    dolfin_error("advect_particles.cpp::advect_particles",
                 "construct advect_particles class",
                 "Periodic boundary value encountered in facet MeshFunction");

  // Set facet info
  update_facets_info();

  // Set facets information
  set_bfacets(mesh_func);

  // Set some other useful info
  _space_dimension = _element->space_dimension();
  _value_size_loc = 1;
  for (std::size_t i = 0; i < _element->value_rank(); i++)
    _value_size_loc *= _element->value_dimension(i);

  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
advect_particles::advect_particles(particles& P, FunctionSpace& U, std::function<const Function&(int, double)> uhi,
                                   const MeshFunction<std::size_t>& mesh_func,
                                   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits)
    : _P(&P), uh(uhi), _element(U.element())
{
  // Confirm that mesh_func does contain periodic boundary values?

  // Set facet info
  update_facets_info();

  // Set facets information
  set_bfacets(mesh_func);

  // Set periodic boundary info
  std::size_t gdim = _P->mesh()->geometry().dim();

  // Check if it has the right size, always has to come in pairs
  // TODO: do provided values make sense?
  if ((pbc_limits.size() % (gdim * 4)) != 0)
  {
    dolfin_error("advect_particles.cpp::advect_particles",
                 "construct periodic boundary information",
                 "Incorrect shape of pbc_limits provided?");
  }

  std::size_t num_rows = pbc_limits.size() / (gdim * 2);
  for (std::size_t i = 0; i < num_rows; i++)
  {
    std::vector<double> pbc_helper(gdim * 2);
    for (std::size_t j = 0; j < gdim * 2; j++)
      pbc_helper[j] = pbc_limits[i * gdim * 2 + j];

    pbc_lims.push_back(pbc_helper);
  }
  pbc_active = true;

  // Set some other useful info
  _space_dimension = _element->space_dimension();
  _value_size_loc = 1;
  for (std::size_t i = 0; i < _element->value_rank(); i++)
    _value_size_loc *= _element->value_dimension(i);

  update_particle_template();
  init_weights();
}
//-----------------------------------------------------------------------------
advect_particles::advect_particles(particles& P, FunctionSpace& U, std::function<const Function&(int, double)> uhi,
                                   const MeshFunction<std::size_t>& mesh_func,
                                   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                                   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> bounded_limits)
    : advect_particles(P, U, uhi, mesh_func, pbc_limits)
{
  std::size_t gdim = _P->mesh()->geometry().dim();
  // Check if it has the right size. [xmin, xmax, ymin, ymax, zmin, zmax]
  if ((bounded_limits.size() % (2 * gdim)) != 0)
  {
    dolfin_error("advect_particles.cpp::advect_particles",
                 "construct periodic boundary information",
                 "Incorrect shape of bounded_limits provided?");
  }

  std::size_t num_rows = bounded_limits.size() / gdim;
  for (std::size_t i = 0; i < num_rows; i++)
  {
    std::vector<double> bounded_domain_lims_helper(2);
    bounded_domain_lims_helper[0] = bounded_limits[2*i];
    bounded_domain_lims_helper[1] = bounded_limits[2*i + 1];

    bounded_domain_lims.push_back(bounded_domain_lims_helper);
  }
  bounded_domain_active = true;
}
//-----------------------------------------------------------------------------
void advect_particles::update_facets_info()
{
  // Cache midpoint, and normal of each facet in mesh
  // Note that in DOLFIN simplicial cells, Facet f_i is opposite Vertex v_i,
  // etc.

  const Mesh* mesh = _P->mesh();
  std::size_t tdim = mesh->topology().dim();
  const std::size_t num_cell_facets = mesh->type().num_entities(tdim - 1);

  // Information for each facet of the mesh
  facets_info.resize(mesh->num_entities(tdim - 1));

  for (FacetIterator fi(*mesh); !fi.end(); ++fi)
  {
    // Get and store facet normal and facet midpoint
    Point facet_n = fi->normal();
    Point facet_mp = fi->midpoint();
    std::vector<bool> outward_normal;

    // FIXME: could just look at first cell only, simplifies code

    int i = 0;
    for (CellIterator ci(*fi); !ci.end(); ++ci)
    {
      const unsigned int* cell_facets = ci->entities(tdim - 1);

      // Find which facet this is in the cell
      const std::size_t local_index
          = std::find(cell_facets, cell_facets + num_cell_facets, fi->index())
            - cell_facets;
      assert(local_index < num_cell_facets);

      // Get cell vertex opposite facet
      Vertex v(*mesh, ci->entities(0)[local_index]);

      // Take vector from facet midpoint to opposite vertex
      // and compare to facet normal.
      const Point q = v.point() - facet_mp;
      const double dir = q.dot(facet_n);
      assert(std::abs(dir) > 1e-10);
      bool outward_pointing = (dir < 0);

      // Make sure that the facet normal is always outward pointing
      // from Cell 0.
      if (!outward_pointing and i == 0)
      {
        facet_n *= -1.0;
        outward_pointing = true;
      }

      // Store outward normal bool for safety check (below)
      outward_normal.push_back(outward_pointing);
      ++i;
    }

    // Safety check
    if (fi->num_entities(tdim) == 2)
    {
      if (outward_normal[0] == outward_normal[1])
      {
        dolfin_error(
            "advect_particles.cpp::update_facets_info",
            "get correct facet normal direction",
            "The normal cannot be of same direction for neighboring cells");
      }
    }

    // Store info in facets_info array
    const std::size_t index = fi->index();
    facets_info[index].midpoint = facet_mp;
    facets_info[index].normal = facet_n;
  } // End facet iterator
}
//-----------------------------------------------------------------------------
void advect_particles::set_bfacets(std::string btype)
{

  // Type of external facet to set on all external facets
  facet_t external_facet_type;
  if (btype == "closed")
    external_facet_type = facet_t::closed;
  else if (btype == "open")
    external_facet_type = facet_t::open;
  else if (btype == "periodic")
    external_facet_type = facet_t::periodic;
  else if (btype == "bounded")
    external_facet_type = facet_t::bounded;
  else
  {
    dolfin_error("advect_particles.cpp", "set external facet type",
                 "Invalid value: %s", btype.c_str());
  }

  const Mesh* mesh = _P->mesh();
  const std::size_t tdim = mesh->topology().dim();
  for (FacetIterator fi(*mesh); !fi.end(); ++fi)
  {
    if (fi->num_global_entities(tdim) == 1)
      facets_info[fi->index()].type = external_facet_type;
    else
      facets_info[fi->index()].type = facet_t::internal;
  }
}
//-----------------------------------------------------------------------------
void advect_particles::set_bfacets(const MeshFunction<std::size_t>& mesh_func)
{
  const Mesh* mesh = _P->mesh();
  const std::size_t tdim = mesh->topology().dim();

  // Check if size matches number of facets in mesh
  assert(mesh_func.size() == mesh->num_facets());

  // Loop over facets to determine type
  for (FacetIterator fi(*mesh); !fi.end(); ++fi)
  {
    if (fi->num_global_entities(tdim) == 1)
    {
      if (mesh_func[fi->index()] == 1)
        facets_info[fi->index()].type = facet_t::closed;
      else if (mesh_func[fi->index()] == 2)
        facets_info[fi->index()].type = facet_t::open;
      else if (mesh_func[fi->index()] == 3)
        facets_info[fi->index()].type = facet_t::periodic;
      else if (mesh_func[fi->index()] == 4)
        facets_info[fi->index()].type = facet_t::bounded;
      else
        dolfin_error("advect_particles.cpp", "set external facet type",
                     "Invalid value, must be 1, 2, 3 or 4");
    }
    else
    {
      assert(mesh_func[fi->index()] == 0);
      facets_info[fi->index()].type = facet_t::internal;
    }
  }
}

//-----------------------------------------------------------------------------
// float prob_prop[10] = {DynVisc, Density, P.Diameter}
// - Add old particle velocity to particle slot 2
Point advect_particles::do_stepLPT(double dt, Point& up, Point& up_1,
  Point& up1, Point& pPos,
  // Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>> LPTParameters) 
  Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> LPTParameters) 
{

    // Initial setup
  const Mesh* mesh = _P->mesh();
  const std::size_t gdim = mesh->geometry().dim();


  double particleDiameter = LPTParameters[0];
  double particleDensity = LPTParameters[1];
  double flowDensity = LPTParameters[2];
  double flowDynamicViscosity = LPTParameters[3];
  double MicroH = LPTParameters[4];
  double MicroW = LPTParameters[5];
  double DisableLift = LPTParameters[6];

  // Set up in terms of dt
  up = up * dt;
  up_1 = up_1 * dt;
  Point upLPT = up;

  
  std::cout << "Flow Velocity B4: " << up << std::endl;
  // std::cout << "Flow up: " << up_v << std::endl;
  // std::cout << "Flow Velocity1: " << up1 << std::endl;

  // Lagrangian particle function based on Drag and bouyancy
  //
  // 1) Access prior particle velocity
  // 2) Get current mesh velocity
  // 3) Calculate lagrangian movement


  // Calculate drag, relax and Reynolds for LPT particles
  double reynolds = this->cal_reynolds(
    flowDynamicViscosity, particleDiameter, flowDensity, up[0], up_1[0]);
  double drag = this->cal_drag(reynolds);
  double relax = this->cal_relax(
    flowDynamicViscosity, particleDiameter, particleDensity);




  // If final dimension (Y (2D) or Z (3D)), apply buoyancy term
  // const double G = 9.8; // Gravity
  // double AddG = (particleDensity - flowDensity);
  // AddG *= G;
  // AddG /= particleDensity;

  // double uMax = 0.8;
  // double H = (MicroW * MicroH);
  // H *= 2;
  // H /= (MicroW + MicroH);



  // Loop particle values for LPT
  for (unsigned int ii = 0; ii < gdim; ii++)
  {
    // if (ii == 2)
    // {
    //   int s = smallestZ / MicroW;
    // } else
    // {
    //   int s = smallestXY / MicroH;
    // }
    // int s = 0;
    // std::cout << "Distance: " << s << std::endl;

    // Calculate Wall lift for LPT particles - Dependent on axis
    double lift = this->cal_WallLiftSq(
      flowDynamicViscosity, particleDiameter, flowDensity, reynolds, ii, up,
      up1, pPos, gdim, MicroH, MicroW, dt);

    // Calculate particle velocity within flow in axis direction
    //  Less than 0 (negative) implies particle has changed direction
    //      Leave as 0 m / s as could be dramatic flow drop relative
    //      to particle velocity. For example 3 - 10
    double particleVelocity = (up[ii] - up_1[ii]);
    // double particleVelocity = (std:abs(up[ii]) - std:abs(up_1[ii]));
    // Save forward moving force, Fluid u
    // double uF = up[ii];
    // double uMax = 0.7;


    // if (0 > (std::abs(up[ii]) - std::abs(up_1[ii])))
    // {
    //     particleVelocity = 0;
    // }
    
    // _P->set_property(ci->index(), i, 1, up);
    // std::cout << "particleAcceleration: " << particleVelocity << std::endl;
    // Calculating Force Balance term (drag and relax)
    double ForceBal = ((drag * reynolds) / 24);
    // std::cout << "ForceBal1: " << ForceBal << std::endl;
    ForceBal *= (relax);

    // std::cout << "ParticleReynolds: " << reynolds << std::endl;
    // std::cout << "Drag: " << drag << std::endl;
    // std::cout << "relax: " << relax << std::endl;

    // std::cout << "ForceBal2: " << ForceBal << std::endl;
    // ForceBal *= particleVelocity;
    // // std::cout << "ForceBal3: " << ForceBal << std::endl;
    
    // Calculate lift (Based upon microfluidic walls)
    //ForceBal += lift;

    // Set particleVelcity to up[ii]
    //up[ii] = ForceBal;


    // Convert flow velocity to position as currently du/dt
    //  Currently has dt/Relax
    //  v = v0 x e^(-t/Relax)
    //  v_p^(n+1) = v + (v - v_p)e^(-t/Relax) + Relax*Rp(1 - e^(t/Relax))
    //    Where Rp is the accelerations due to all other forces except
    //      drag forces such as grvaity, rotation effects

    // Add drag to particle
    upLPT[ii] = (particleVelocity) * exp(-dt/ForceBal);
    // up[ii] += (particleVelocity) * exp(-dt/ForceBal);

    // Calculating the reflection, 'r', to find if it is +ve/-ve movement
    // double d = (-1 * up[ii]); // Need to invert the particle direction
    // double r = d - ( 2 * (d * normFlow ) * normFlow ); // normFlow (-1 * up_1[ii])
    // r = r * 2;

    // Assume the particle is neutrally bouyant
    // up[ii] -= ((lift) * ForceBal * (exp(-dt/ForceBal) - 1));

    // Treat as neutrally bouyant
    

    // std::cout << "Particle Velocity w/o lift: " << up[ii] << std::endl;
    std::cout << "Added lift: " << (lift * ForceBal * (exp(-dt/ForceBal) - 1)) << std::endl;
    
    // up[ii] -= lift * 100;
    std::cout << "Lift Force: " << lift << std::endl;
    std::cout << "(exp(-dT/t) - 1): " << std::abs(exp(-dt/ForceBal) - 1) << std::endl;
    std::cout << "Force Balance: " << ForceBal << std::endl;

    // Lift should be positive / negative defined in WallLiftSq function
    if (DisableLift == 0)
    {
      upLPT[ii] += ((lift * ForceBal) * std::abs(exp(-dt/ForceBal) - 1));
    } else
    {
      std::cout << "Lift Disabled: " << std::endl;
    }
  }
  
  std::cout << "Flow Velocity AF: " << upLPT << std::endl;

  return upLPT;
}

//-----------------------------------------------------------------------------
double advect_particles::cal_drag(double reynolds)
//  Schiller-Naumann Model & Stokes Drag
//  Drag coefficent determines the amount of drag acting upon the particle. 
//    Above a Particle Reynolds number of 1000, flow is turbulent and drag is
//    approximately 0.44. Above a Particle Reynolds number of 1,
//    Schiller-Naumann Model is appropriate otherwise Stokes Drag.
{

  // if (dynVisc == 0)
  // {
  //   return -1.0 * std::numeric_limits<double>::infinity();
  // }

  // Relative Reynolds Number
  // double reynolds = cal_reynolds(dynVisc, particleDiameter, flowDensity, up, up_1);

  if (reynolds < 1000)
  {
    double ans = (24 / reynolds);
    if (reynolds < 1)
    {
      // Return Stokes Drag
      std::cout << "Stokes Drag: " << ans << std::endl;
      return ans;
    }
    double rhs = pow(reynolds, 0.687);
    rhs *= 0.15;
    rhs += 1.0;
    
    ans *= rhs;
    std::cout << "Stokes Drag: " << ans << std::endl;
    std::cout << "Schiller-Naumann Drag: " << (ans * rhs) << std::endl;
    return ans;
    //return (24 / reynolds)*(1.0 + 0.15 * pow(reynolds, 0.687));
  }
  else
  {
    return 0.44;
  }

}

//-----------------------------------------------------------------------------
double advect_particles::DEFINE_DPM_TIMESTEP(double dt, Point& up, Point& up_1,
  Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> LPTParameters,
  Point& pPos, Point& dPoint)
  // dt particle relax time is (Drag * ReP) / (FlowDensity * Pdia^2)
  //   Based upon ANSYS DEFINE_DPM_TIMESTEP which uses the particle relaxation time
  //   to move particles if the timestep, dT, is too large.
  //    https://www.afs.enea.it/project/neptunius/docs/fluent/html/udf/node79.htm
{
  double particleDiameter = LPTParameters[0];
  double particleDensity = LPTParameters[1];
  // double flowDensity = LPTParameters[2];
  double flowDynamicViscosity = LPTParameters[3];
  // double MicroH = LPTParameters[4];
  // double MicroW = LPTParameters[5];
  double EnableDTstep = LPTParameters[6];

  if (EnableDTstep == 0)
  {
    // Calculate drag, relax and Reynolds for LPT particles
    // double drag = cal_drag( 
    //   flowDynamicViscosity, particleDiameter, flowDensity, up, up_1);
    // double relax = this->cal_relax(
    //   flowDynamicViscosity, particleDiameter, particleDensity);
    // double reynolds = cal_reynolds(
    //   flowDynamicViscosity, particleDiameter, flowDensity, up, up_1);

    // Particle Response Time
    double dt1 = (pow(particleDiameter,2) * particleDensity);
    dt1 /= (18.0 * flowDynamicViscosity); 
    // (drag * flowDynamicViscosity); //(drag * reynolds);

    // std::cout << "dt: " << (dt) << std::endl;
    // if (dt > (dt1 / 5.))
    // {
    //   dt = (dt1 / 5.);
    //   std::cout << "dT (dt1/5.): " << dt << std::endl;
    // }

    double h = 0.00016; // Channel height
    double w = dPoint[1]; // Channel width
    double sT = dPoint[0]; // Particle distance from boundary

    int s = ((sT / w) * 100);
    //  Assumed is Z is a constant height
    double zMin = 0.0110585018148766;
    double sTT = (pPos[2] - zMin);

    sTT /= h;
    sTT *= 100;
    int sZ = sTT;

    if (sZ < s)
    {
      s = sZ;
    }

    double sStep = (dt - dt1) / 10;
    if (s < 11)
    {
      dt1 += ( ( s - 1 ) * sStep );
      dt = dt1;
      std::cout << "dT: " << dt << std::endl;
    }
  }

  // std::cout << "dt: " << (dt) << std::endl;

  // Point relativeVelocity = (up + up_1);

  // double relativeSpeed = relativeVelocity.norm();
  // double particleSpeed = up_1.norm();
  // double uMax = 0.8;
  // uMax /= 3;

  // std::cout << "Particle Speed: " << particleSpeed << std::endl;

  // if (particleSpeed < uMax)
  // {
  //   dt = 5e-5;
  //   std::cout << "dT 5e-5" << std::endl;
  // }

  return dt;
}

//-----------------------------------------------------------------------------
double advect_particles::cal_WallCorrection(double particleDiameter, double distance)
//  Wall correction is a force which correct particles' drag against a wall
//    This is similar to the Comsol Wall Correction in the particles drag function
//
//  Based upon 7-4.28 (pg 327) and 7-4.39 (pg 330) within 
//    Happell, J. (1981) Low Reynolds number hydrodynamics,
//    Sphere Moving Relative to Plabne Walls,
//    Chapter 7 - Wall Effects on Motion of a Single Particle,
//    DOI: 10.1007/978-94-009-8352-6
//    
//  Essentially Part one is parallel to flow
//              Part two is perpendicular to flow
{
  double particleRadius = particleDiameter / 2;
  double ka = (particleRadius) / (distance);

  double part1 = 1;
  part1 -= (9.0/16.0) * ka;
  part1 += (1.0/8.0) * pow( ka , 3 );
  part1 -= (45.0/256.0) * pow( ka , 4 );
  part1 -= (1.0/16.0) * pow( ka , 5 );
  
  double part2 = 1;
  part2 -= (9.0/8.0) * ka;
  part2 += (1.0/2.0) * pow ( ka , 3 );

  // std::cout << "Drag WC P1: " << part1 << std::endl;
  // std::cout << "Drag WC P2: " << part2 << std::endl;

  double ans = ( (1.0 + (1.0 / part1))* -1.0) + ( (1.0 / part2)* 1.0 );
  // std::cout << "Ans: " << ans << std::endl;

  ans = abs(ans);
  return ans;

}

//-----------------------------------------------------------------------------
double advect_particles::cal_relax(double dynVisc, double diameter, double density)
//  Relax calculation determines how much a particle will "follow" flow streamlines
//    or migrate away due to particle drag (particle density, particle size and 
//    fluid viscosity)
//
{
    // std::cout << "dynVisc: " << (dynVisc) << std::endl;
    // std::cout << "diameter: " << (diameter) << std::endl;
    // std::cout << "density: " << (density) << std::endl;
    double top = 18.0;
    top *= dynVisc;
    double bottom = pow(diameter,2);
    bottom *= density;
    
    return ((top) / (bottom));
  //return ((18.0 * dynVisc) / (density * diameter * diameter));

}

//-----------------------------------------------------------------------------
double advect_particles::cal_reynolds(double dynVisc, 
  double particleDiameter, double flowDensity, double FlowVel, double PartVel)
// Reynolds number calculation requires average speed to calcualte
//    laminar flow (<1) or turbulent flow (>1).
//
{ 
  // Point relativeVelocity;

  // for (int i = 0; i < 3; i++)
  // {
  //   // Particle Velocity - Flow Velocity
  //   // relativeVelocity[i] = (up_1[i] - up[i]);
  //   // Flow Velocity - Particle Velocity
  //   relativeVelocity[i] = std::abs(up[i] - up_1[i])
  //   // std::cout << relativeVelocity[i] << std::endl;
  // }

  // Flow Velocity - Particle Velocity
  // relativeVelocity = (up - up_1);


  // Calculate relative speed of particle
  // double relativeSpeed = relativeVelocity.norm();
  double relativeSpeed = FlowVel - PartVel;

  double ans = (relativeSpeed * particleDiameter);
  ans *= flowDensity;
  ans /= dynVisc;

  // Reynolds is always postive hence absolute return
  return std::abs(ans);
  //return std::abs((flowDensity * relativeSpeed * particleDiameter) / dynVisc);
}

//-----------------------------------------------------------------------------
// double advect_particles::cal_WallLiftSq(double dynVisc, 
//   double particleDiameter, double flowDensity, int i, Point& up, Point& up_1,
//   Point& pp, const Mesh* mesh, double h, double w)
//
//      Calculates net wall lift
double advect_particles::cal_WallLiftSq(double dynVisc, 
  double particleDiameter, double flowDensity, double reynolds,
  int i, Point& up, Point& up1, Point& pPos, int gdim,
  double h, double w, double dt)
{

  // Initialise G1 and G2 for lift constants
  //  GSpot is based upon
  //    Ho, B. P., & Leal, L. G. (1974).
  //      Inertial migration of rigid spheres in
  //      two-dimensional unidirectional flows.
  //      Journal of Fluid Mechanics, 65(2), 365–400.
  //    https://doi.org/10.1017/S0022112074001431
  //  G1 in slot 0, G2 in slot 1
  //  The index is 0.01 step, index start at 0
  //    i.e. 49 is 0.50
  double GSpot[2][50] =
  { 
    { 0,
      0.0419,
      0.0837,
      0.1254,
      0.1669,
      0.208,
      0.2489,
      0.2894,
      0.3293,
      0.3688,
      0.4077,
      0.4459,
      0.4834,
      0.52,
      0.556,
      0.591,
      0.626,
      0.659,
      0.691,
      0.723,
      0.753,
      0.782,
      0.81,
      0.836,
      0.861,
      0.885,
      0.907,
      0.927,
      0.945,
      0.96,
      0.973,
      0.982,
      0.988,
      0.99,
      0.988,
      0.981,
      0.971,
      0.957,
      0.943,
      0.931,
      0.927,
      0.94,
      0.982,
      1.07,
      1.23,
      1.5,
      1.93,
      2.58,
      3.59,
      5.33
      },
    { 1.072,
      1.07,
      1.068,
      1.066,
      1.062,
      1.056,
      1.05,
      1.042,
      1.033,
      1.023,
      1.012,
      1,
      0.987,
      0.972,
      0.956,
      0.94,
      0.922,
      0.902,
      0.882,
      0.861,
      0.838,
      0.815,
      0.79,
      0.765,
      0.738,
      0.711,
      0.683,
      0.654,
      0.625,
      0.596,
      0.566,
      0.536,
      0.506,
      0.477,
      0.448,
      0.42,
      0.393,
      0.368,
      0.345,
      0.324,
      0.306,
      0.292,
      0.282,
      0.278,
      0.28,
      0.291,
      0.315,
      0.354,
      0.414,
      0.505
    }
  };


  // double distance = cal_ParticleDistFromBoundary(pp, mesh);
  // std::cout << "P distance from boundary: " << distance << std::endl;

  // // Adding wall lift force
  // // Particle Velocity - Flow Velocity
  // double relativeSpeed = (up[i] - 0.77); // Per axis
  // // double relativeSpeed = (up[i] - up_1[i]); // Per axis
  // std::cout << "Relative Speed P: " << relativeSpeed << std::endl;

  // Calculate relative speed of particle and flow
  //    Require for stress rate and shear gradient
  // double pNorm = up_1.norm();
  // double uNorm = up.norm();
  double uNorm1 = up1.norm();

  // 
  // double relativeSpeed = pNorm - uNorm;

  // double w = 0.0005; // X or Y axis width
  // double h = 0.000160; // Z axis height
  //double H = (2 * w * h) / (w + h); // Hydrodynamic Diametre
  
  // double H = (w * h);
  // H *= 2;
  // H /= (w + h);
  double H = 0.0005; // 500 um width

  // Calculate fl co-efficient based on H^2 / ( pd^2 * sqrtroot(Reynolds))
  // double Fl_temp = pow(particleDiameter,2);
  // Fl_temp *= sqrt(reynolds);
  // double Fl = pow(H, 2) / Fl_temp;
  // std::cout << "Flow Lift denominator: " << Fl_temp << std::endl;
  // std::cout << "Relative Flow Lift: " << Fl << std::endl;
  // double fl = 0.5;

  // // Calculate Saffman version of lift (Fl) based on ANSYS
  // double Fl = pow(particleDiameter,4);
  // Fl *= pow(relativeSpeed,2);
  // Fl *= flowDensity;
  // Fl *= fl; // 0.5
  // Fl /= pow(H, 2);

  // // Requires net wall lift where Fl feeds into Flnl based on Asmolov(1999)
  // //  Requires Particle Reynolds Number
  // // double reynoldsP = cal_reynolds(dynVisc, particleDiameter, flowDensity, up, up_1);
  // double Erp = pow(reynolds, 0.5);
  // //  Requires channel Reynolds Number
  // double reynoldsC = flowDensity;
  // reynoldsC *= H;
  // reynoldsC *= relativeSpeedU; // Hard coded mean velocity of channel (m / s)
  // reynoldsC /= dynVisc;
  // double Flnl = Erp * Fl ;
  // Flnl *= pow(reynoldsC, 0.5);
  // Flnl *= (1 / pow(Erp, 3) );
  // Flnl *= -1;


  double uMax = 0.8; // Needs to be imported from problem parameters
  int s = 0;

  // 4.0 * Umax * x[1] * (0.41 - x[1]) / pow(0.41, 2)
  // up[ii] = up[ii] * ( LPTVecT / upNorm);
  
  // s is the particle position from the walls important for lift
  //  This is generalised on a 2D parabolic and does not indicate
  //    closeness to a particular wall
  // x is uNorm, uMax is maximum flow rate
  // Function is divided by H to make percentage and times by 100
  //    to make it accessable "int" for GSpot
  // uNorm1 is the centre line of the channel for velocities, expected to be
  //    parabolic and avoid issues whe(pPos[2] - 0.0110585)
  if (s == 0)
  {
    s = (((H/2 - (H/2 * sqrt(1 - (1/uMax) * uNorm1))) / H) * 100);
    // double sT = (((H/2 - (H/2 * sqrt(1 - (1/uMax) * uNorm1))) / H) * 100);
    // std::cout << "uMax: " << uMax << std::endl;
    // std::cout << "uNorm1: " << uNorm1 << std::endl;
    // std::cout << "sT: " << sT << std::endl;
    // std::cout << "H: " << H << std::endl;
  }

  // If Z axis, use Z axis height and P position
  //  Assumed is Z is a constant height
  if (i == gdim-1)
  {
    H = 0.00016; // Set Z to boundary height, not hydraulic
    uMax = 0.0022;//0.000005; // 0.0022 // Bouyancy acting upon particle
    // Point pPos = _P->x(ci->index(), 1);
    // s = ( Z particle position minus zMin ) / Zrange
    // std::cout << "pPos[2]: " << pPos[2] << std::endl;
    // std::cout << "Za axis S value: " << ((pPos[2] - 0.0110585) / 0.00016) << std::endl;
    // s needs to be out of 100, not decimal
    // s = ((pPos[2] - 0.0110585) / 0.00016) * 100;
    double zMin = 0.0110585018148766;
    double sT = (pPos[2] - zMin);
    sT /= H;
    sT *= 100;
    s = sT;
    // Set Z axis to something reasonable?
    // uNorm = std::abs(up[2]); // uMax * 0.01;//std::abs(up[2]);
  
  }

  // particles.set_property(c, pi, 6, dPoint)
  // particles.set_property(c, pi, 7, P4)

  // Shear rate - Based on velocity norm at particle position
  // double Lander = ( uNorm / (s * 0.01 * H) );//(-8 * uNorm * (s * 0.01 * H) ) / pow(H,2);

  // // Shear gradient
  // double Beta = dynVisc * Lander;

  // Based on paper Ho 1974 - eq. 3.17a & b
  // double k = ( particleDiameter / 2 ) / H;
  // double Lander = -4 * uMax * pow(k, 2);
  // double Beta = 4 * uMax * ( 1 - (2 * s) ) * k;

  // Comsol implementation
  // double k = ( particleDiameter / 2 ) / H;

  // Stopped the fail due to s = 0 causing inf bug
  if (s < 1)
  {
    s = 1;
  } else if (s > 99)
  {
    s = 99;
  }
  
  // double Lander = ( uNorm / (s * 0.01 * H) ); //uNorm * pow(k, 2);
  
  // double Beta = dynVisc * Lander; //Beta = uNorm * k;


  // If, for some reason, s is larger than 50.
  //  Function should always make it sub 50 due to the square root
  //    plus and minus
  //  Also set the equation to inverse if above 50 for the Z axis.
  int sU = 1;
  if (s > 50)
  {
    s = 100 - s;
    sU = -1;
  }

  double Lander = ( uMax / (s * 0.01 * H) ); //uNorm * pow(k, 2);
  // double Lander = 4 * uMax;

  double Beta = dynVisc * Lander; //Beta = uNorm * k;
  // double Beta = 4 * uMax * ( 1 - (2 * (s * 0.01) ) );

  // std::cout << "Flow uNorm: " << uNorm << std::endl;
  std::cout << "P distance from boundary: " << s << std::endl;

  // Invert for the GSpot as index 50 is 0 distance from wall
  s = 50 - s;
  // std::cout << "Lander: " << Lander << std::endl;
  // std::cout << "Beta: " << Beta << std::endl;
  // std::cout << "G1 Spot(50-s): " << GSpot[0][s] << std::endl;
  // std::cout << "G2 Spot(50-s): " << GSpot[1][s] << std::endl;

  // std::cout << "uMax: " << uMax << std::endl;
  // std::cout << "uNorm: " << uNorm << std::endl;
  // std::cout << "Hydraulic Diameter: " << H << std::endl;

  // Lift force
  //  Using the defined G1 and G2 earlier
  //  & shear rate and shear gradient
  double CL = ( pow( Beta, 2) * GSpot[0][s] ) + (Beta * Lander * GSpot[1][s] );
  // double CL = ( 36 * pow((1 - (2 * (s * 0.01) )), 2) * GSpot[0][s] );
  // CL -= ( 36 * (1 - (2 * (s * 0.01) ) ) * GSpot[1][s] );
  // CL1 = CL1 * 2;
  // double Cs = 1L = ( pow( H, 2) ) / ((particleDiameter / 2) * sqrt(reynolds));
  // std::cout << "Lift co-efficient (CL): " << CL << std::endl;
  // double CL1 = G1[0,s];
  // CL1 *= pow(Beta, 2);
  // double CL2 = Gspot

  // Method using G1 G2
  // Flnl = k^2 * Reynolds * CL
  // k^2 = (particle radius / H)^2
  ////
  double Flnl = pow( ( (particleDiameter / 2) / H ) , 2);
  Flnl *= (flowDensity * uMax * H) / dynVisc; // Reynolds;
  Flnl *= CL;
  Flnl = std::abs(Flnl);

  // Method using Di Carlo
  // double Flnl = flowDensity;
  if (i != gdim-1) // Z axis based on pPos using sU
  {
    Flnl = flowDensity;
    Flnl = 0.5;
    Flnl *= pow( uMax, 2 );
    Flnl *= pow( (particleDiameter / 2), 6 );
    Flnl /= pow( H, 4 );
  }
  // G1G2 flow for 
  //
  // k^2 = (particle radius / H)^2
  // double Flnl = pow(( (particleDiameter / 2) / H  ), 2); 
  // Flnl *= pow(uMax,2); // Vmax^2 = Flow max velocity 
  // Flnl *= flowDensity; // Po = Flow density
  // Flnl *= CL; // G1 and G2
  // Flnl *= pow( (particleDiameter / 2), 2); // a^2 = (Particle Radius)^2
  // Flnl = std::abs(Flnl);
  // Flnl *= -10;


  if (i == gdim-1) // Z axis based on pPos using sU
  {
    Flnl *= sU;
  } else { // Add lift to XY
    // Add lift if above 1% uMax to opposite direction (XY)
    // if ( (uMax * 0.01) < std::abs(up[i]) )
    // {
    int iP;
    if (i == 1) // y axis
    {
      iP = 0; // to change x axis
    }
    else // x axis
    {
      iP = 1; // to change y axis
    }
    // Add ratio of X:Y vectors
    double ratioU = 1;
    if (gdim == 3){
      // Where y / y would be 1, therefore a 1:1 ratio
      // ratio = x / y
      ratioU += std::abs(up1[0]) / std::abs(up1[1]);
    }
    
    // std::cout << "up1 Add lift: " << up1 << std::endl;
    // up1 is the direction along the X or Y axis the particle is moving
    //  This assumes a parabolic where flow drops off towards the boundaries
    //  Therefore, particle movement should be in the opposite direction to movement
    Flnl *= (std::abs(up1[iP]) / ratioU);
    if (up1[i] < 0) // Positive along axis
    {
      std::cout << "Flnl: " << up1[i] << std::endl;
      // up[ii] += ( lift * (std::abs(up[iP]) / ratioU) );// * 10000;
      // up[ii] += std::abs((lift * (std::abs(up[iP]) / ratioU) ) * ForceBal * (exp(-dt/ForceBal) - 1));
      Flnl *= -1;
    }
    // } else // R negative.
    // {
    //   // up[ii] -= ( lift * (std::abs(up[iP]) / ratioU) );// * 10000;
    //   // up[ii] -= std::abs((lift * (std::abs(up[iP]) / ratioU) ) * ForceBal * (exp(-dt/ForceBal) - 1));
    //   Flnl *= (std::abs(up[iP]) / ratioU);
    // }
    // } else // Apply no lift force
    // {
    //   Flnl = 0;
    // }
  }


  return Flnl;// * dt;
  // Make it tiem depenent otherwise far too large!
  //return 1;
  //return (0.5 * flowDensity * pow(relativeSpeed,2) * pow(particleDiameter,4)) / pow(H, 2);
}


//-----------------------------------------------------------------------------
// double advect_particles::cal_WallLiftSq(double dynVisc, 
//   double particleDiameter, double flowDensity, int i, Point& up, Point& up_1,
//   Point& pp, const Mesh* mesh, double h, double w)
//
//      Calculates net wall lift
// double advect_particles::cal_WallLiftSq(double dynVisc, 
//   double particleDiameter, double particleDensity, double flowDensity,
//   Point& zParam, double uMax, int i, int gdim, Point& up,
//   Point& pPos, Point& dPoint, Point& P4)
// {

//   // Initialise G1 and G2 for lift constants
//   //  GSpot is based upon
//   //    Ho, B. P., & Leal, L. G. (1974).
//   //      Inertial migration of rigid spheres in
//   //      two-dimensional unidirectional flows.
//   //      Journal of Fluid Mechanics, 65(2), 365–400.
//   //    https://doi.org/10.1017/S0022112074001431
//   //  G1 in slot 0, G2 in slot 1
//   //  The index is 0.01 step, index start at 0
//   //    i.e. 49 is 0.50
//   double GSpot[2][50] =
//   { 
//     { 0,
//       0.0419,
//       0.0837,
//       0.1254,
//       0.1669,
//       0.208,
//       0.2489,
//       0.2894,
//       0.3293,
//       0.3688,
//       0.4077,
//       0.4459,
//       0.4834,
//       0.52,
//       0.556,
//       0.591,
//       0.626,
//       0.659,
//       0.691,
//       0.723,
//       0.753,
//       0.782,
//       0.81,
//       0.836,
//       0.861,
//       0.885,
//       0.907,
//       0.927,
//       0.945,
//       0.96,
//       0.973,
//       0.982,
//       0.988,
//       0.99,
//       0.988,
//       0.981,
//       0.971,
//       0.957,
//       0.943,
//       0.931,
//       0.927,
//       0.94,
//       0.982,
//       1.07,
//       1.23,
//       1.5,
//       1.93,
//       2.58,
//       3.59,
//       5.33
//       },
//     { 1.072,
//       1.07,
//       1.068,
//       1.066,
//       1.062,
//       1.056,
//       1.05,
//       1.042,
//       1.033,
//       1.023,
//       1.012,
//       1,
//       0.987,
//       0.972,
//       0.956,
//       0.94,
//       0.922,
//       0.902,
//       0.882,
//       0.861,
//       0.838,
//       0.815,
//       0.79,
//       0.765,
//       0.738,
//       0.711,
//       0.683,
//       0.654,
//       0.625,
//       0.596,
//       0.566,
//       0.536,
//       0.506,
//       0.477,
//       0.368,
//       0.345,
//       0.324,
//       0.306,
//       0.292,
//       0.282,
//       0.278,
//       0.28,
//       0.291,
//       0.315,
//       0.354,
//       0.414,
//       0.505
//     }
//   };

//   // Calculate relative speed of particle and flow
//   //    Require for stress rate and shear gradient
//   // Point P4 = (_P->property(ci->index(), i, 7));
//   // Point dPoint = (_P->property(ci->index(), i, 6));
//   double h = zParam[0]; // Channel height
//   double w = dPoint[1]; // Channel width
//   double sT = dPoint[0]; // Particle distance from boundary
//   // double uMax = 0.76;//1; // Needs to be imported from problem parameters
//   double H = w;
//   // double uMax = 1.8e-5;
//   // double uMax = up.norm(); // double uNorm = up.norm();
//   //// uM is Mean velocity in Ho and Leal is assumed to be
//   ////    2/3 max flow velocity
//   // double uM = 2.0 * uMax ;
//   // uM /= 3.0;
//   double uM = uMax;
//   // double particleRadius = particleDiameter / 2.0;
  

//   int s = ((sT / w) * 100.0);



//   // If Z axis, use Z axis height and P position
//   //  Assumed is Z is a constant height
//   if (i == gdim-1)
//   {
//     H = h;
//     double zMin = zParam[1];
//     double sTT = (pPos[2] - zMin);
//     // k = (particleRadius) / (H);
//     // if ( (sTT > (H - (particleDiameter / 2) ) ) || (sTT < (particleDiameter / 2) ) )
//     //   {
//     //     sTT = particleDiameter / 10;
//     //   }
//     sTT /= h;
//     sTT *= 100.0;
//     s = sTT;
//     std::cout << "pPos[2]: " << pPos[2] << std::endl;
//     std::cout << "Za axis S value: " << ((pPos[2] - 0.0110585) / 0.00016) << std::endl;
//     std::cout << "Za axis S value: " << sTT << std::endl;
//     // uMax = 0.022;//0.0022;//0.000005; // Bouyancy acting upon particle
//     // uMax = 0.000005;
//     // uMax = 200; // uMax = 0.8; 1000 100
//     // sT = (s * 0.01 * H);
//   }

//   // Hydraulic Diameter of rectangle channel
//   H = (w * h);
//   H *= 2;
//   H /= (w + h);
//   // double k = (particleRadius) / (H);
//   double k = (particleDiameter) / (H);



//   // Stopped the fail due to s = 0 causing inf bug
//   if (s < 1)
//   {
//     s = 1;
//   } else if (s > 99)
//   {
//     s = 99;
//   }
  
//   int sU = 1;
//   if (s > 50)
//   {
//     s = 100 - s;
//     sU = -1;
//   }

//   //// Shear Gradient (rate of change of Shear Rate)
//   // double Lander = ( uMax / (sT) );
//   // double Lander = ( uMax / (s * 0.01 * H) ); //uNorm * pow(k, 2);
//   // double Lander = -8 * uMax;
//   // double Lander = (pow( H, 2 ) / 2 ) * 8 * uMax;
//   // double Lander = dynVisc * Beta;

//   // double Lander = 0.0;
//   // double Lander = -6.0;
//   double Lander = -8.0;

//   // double Lander = ( pow( H , 2) / 2 ) * uMax;
//   // if (i != gdim-1)
//   // {
//   //   Lander = ( uMax / (sT) );
//   // }

//   //// Shear Rate
//   // double Beta = uMax * H;
//   // double Beta = dynVisc * Lander; //Beta = uNorm * k;
//   // double Beta = ( uMax / (s * 0.01 * H) );
//   // double Lander = dynVisc * Beta;
//   // double Beta = 4 * uMax * ( 1 - (s * 0.01) ) * H;

//   double Beta = 4.0 * ( 1.0 - (2.0 * (s * 0.01) ) );
//   // double Beta = ( 1.0 - (2.0 * (s * 0.01) ) );

//   // double Beta = 4 * uMax * H * ( 1 - (2 * (s * 0.01) ) );

//   // std::cout << "Flow uNorm: " << uNorm << std::endl;
//   std::cout << "P distance from boundary: " << s << std::endl;
//   std::cout << "k: " << k << std::endl;
//   // Invert for the GSpot as index 50 is 0 distance from wall
//   s = 50 - s;
//   // std::cout << "S value for G: " << s << std::endl;
//   // std::cout << "G1: " << GSpot[0][s] << std::endl;
//   // std::cout << "G2: " << GSpot[1][s] << std::endl;

//   //// Lift force
//   //  Using the defined G1 and G2 earlier
//   //  & shear rate and shear gradient
//   double CL = ( pow( Beta, 2) * GSpot[0][s] ) + (Beta * Lander * GSpot[1][s] );
//   // double CL = 36 * (( pow( Beta, 2) * GSpot[0][s] ) + (Beta * GSpot[1][s] ));
//   // double CL = 0.5;
//   // Method using G1 G2
//   // Flnl = k^2 * Reynolds * CL
//   // k^2 = (particle radius / H)^2
//   //// Ho and Leal 1974 5.24
//   // double Flnl = pow( ( (particleDiameter) / H ) , 2);
//   // Flnl *= (flowDensity * uMax * H) / dynVisc; // Reynolds;
//   // // Flnl *= (flowDensity * uMax * particleDiameter) / dynVisc; // Reynolds;
//   // Flnl *= CL;
//   // Flnl = std::abs(Flnl);

//   //// Ho and Leal 1974 5.27
//   double Flnl = pow( uM , 2);
//   // Flnl *= pow( (particleRadius) , 2); // a
//   Flnl *= pow( (particleDiameter) , 2); // a
//   Flnl *= pow( ( k ) , 2);
//   // Flnl *= pow( ( k ) , 3);
//   // Flnl *= H;
//   // double Flnl = pow( (particleDiameter / 2) , 4);
//   // Flnl /= pow( H , 2 );
//   // Flnl *= pow( uMax , 2);
//   Flnl *= flowDensity; // Reynolds;
//   Flnl *= CL;
//   // Flnl *= 5000;
//   // Flnl = std::abs(Flnl);

//   //// Comsol (same as Ho and Leal 1974 5.27)
//   // double Flnl = pow( (particleDiameter) , 4);
//   // Flnl /= pow( H , 2 );
//   // Flnl *= pow( uMax , 2);
//   // Flnl *= flowDensity; // Reynolds;
//   // Flnl *= CL;
//   // Flnl = std::abs(Flnl);

//   //// Lateral Velocity Movement
//   //// Ho and Leal 1974 6.1
//   // Unlikely to be relevant due to acceleration req., not velocity
//   // double PI = 3.14159265358979323846;
//   // //
//   // double Flnl = flowDensity;
//   // Flnl *= pow( uM , 2);
//   // // std::cout << "Flnl fD * uM " << Flnl << std::endl;
//   // Flnl *= H;
//   // // std::cout << "Flnl b4 divide " << Flnl << std::endl;
//   // // Flnl *= 6;
//   // // Flnl /= dynVisc * PI;
//   // Flnl /= dynVisc * PI * 6.0;
//   // Flnl *= pow( k , 3 );
//   // Flnl *= CL;
//   // // Re * Uz value
//   // Flnl /= (flowDensity * uMax * H) / dynVisc;//reynolds
//   // Flnl = std::abs(Flnl);

//   // // std::cout << "Flnl AF CL " << Flnl << std::endl;

//   if (i == gdim-1) // Z axis based on pPos using sU
//   {
//     Flnl *= sU;
//     // Flnl = 0;
//     // std::cout << "Flnl(z): " << Flnl << std::endl;
//   } else { // Add lift to XY
//     // int iP;
//     // if (i == 1) // y axis
//     // {
//     //   iP = 0; // to change x axis
//     // }
//     // else // x axis
//     // {
//     //   iP = 1; // to change y axis
//     // }
//     // Add ratio of X:Y vectors
//     // double ratioU = 1;
//     // Total magnitude of XY is...
//     // double MagXY = sqrt(pow( up[0], 2) + pow( up[1], 2));
//     // Total magnitude of XY (face norm) is...
//     double MagXY = sqrt(pow( P4[0], 2) + pow( P4[1], 2));//P4[i]
  
    
//     // up1 is the direction along the X or Y axis the particle is moving
//     //  This assumes a parabolic where flow drops off towards the boundaries
//     //  Therefore, particle movement should be in the opposite direction to movement
//     // Flnl *= (std::abs(up[iP]) / MagXY);
//     Flnl *= (std::abs(P4[i]) / MagXY);
//     if (P4[i] < 0) // Negative along axis
//     {
//       // std::cout << "Flnl: " << Flnl << std::endl;
//       // up[ii] += ( lift * (std::abs(up[iP]) / ratioU) );// * 10000;
//       // up[ii] += std::abs((lift * (std::abs(up[iP]) / ratioU) ) * ForceBal * (exp(-dt/ForceBal) - 1));
//       Flnl *= -1;
//     }
//   }

//   //// Convert force to acceleration
//   // F = m*a == a = F/m
//   // m = volume(Sphere) * density == 1/6 pi D^3 * Particle Density
//   double PI = 3.14159265358979323846;
//   double mass = (PI * pow( particleDiameter, 3 )); // Volume
//   // std::cout << "mass1: " << mass << std::endl;
//   mass /= 6.0;
//   // std::cout << "mass2: " << mass << std::endl;
//   mass *= particleDensity; // mass
//   // std::cout << "mass3: " << mass << std::endl;
//   Flnl /= mass; // F/m = Acceleration

//   // // Ho & Leal (1974) Lateral Particle Velocity Eq. 6.1
//   // double mass = (6 * PI * dynVisc * particleRadius);
//   // double mass = 6 * PI * dynVisc;
//   // Flnl /= mass; // F/m = Acceleration
//   // // Calculate Particle Reynolds (using dimensionless Um)
//   // double Reynolds = particleDensity * uMax * particleRadius * k;
//   // Reynolds /= dynVisc;
//   // Flnl /= Reynolds; // Usz * Re = Flnl

//   return Flnl;
// }


//-----------------------------------------------------------------------------
double advect_particles::cal_NetLiftHoLeal1974(double dynVisc, 
  double particleDiameter, double particleDensity, double flowDensity,
  Point& zParam, double uMax, int i, int gdim, Point& up,
  Point& pPos, Point& dPoint, Point& P4, Point C_U_G = Point(0.0,0.0,0.0),
  Point C_V_G = Point(0.0,0.0,0.0), Point C_W_G = Point(0.0,0.0,0.0))
{

  // Initialise G1 and G2 for lift constants
  //  GSpot is based upon
  //    Ho, B. P., & Leal, L. G. (1974).
  //      Inertial migration of rigid spheres in
  //      two-dimensional unidirectional flows.
  //      Journal of Fluid Mechanics, 65(2), 365–400.
  //    https://doi.org/10.1017/S0022112074001431
  //  G1 in slot 0, G2 in slot 1
  //  The index is 0.01 step, index start at 0
  //    i.e. 49 is 0.50
  double GSpot[2][50] =
  { 
    { 0, 0.0419, 0.0837, 0.1254, 0.1669, 0.208, 0.2489, 0.2894, 0.3293, 0.3688,
      0.4077, 0.4459, 0.4834, 0.52, 0.556, 0.591, 0.626, 0.659, 0.691, 0.723,
      0.753, 0.782, 0.81, 0.836, 0.861, 0.885, 0.907, 0.927, 0.945, 0.96, 0.973,
      0.982, 0.988, 0.99, 0.988, 0.981, 0.971, 0.957, 0.943, 0.931, 0.927, 0.94,
      0.982, 1.07, 1.23, 1.5, 1.93, 2.58, 3.59, 5.33
    },
    { 1.072, 1.07, 1.068, 1.066, 1.062, 1.056, 1.05, 1.042, 1.033, 1.023, 1.012,
      1.0, 0.987, 0.972, 0.956, 0.94, 0.922, 0.902, 0.882, 0.861, 0.838, 0.815,
      0.79, 0.765, 0.738, 0.711, 0.683, 0.654, 0.625, 0.596, 0.566, 0.536, 0.506,
      0.477, 0.448, 0.420, 0.393, 0.368, 0.345, 0.324, 0.306, 0.292, 0.282, 0.278,
      0.28, 0.291, 0.315, 0.354, 0.414, 0.505
    }
  };

  // Calculate relative speed of particle and flow
  //    Require for stress rate and shear gradient
  // Point P4 = (_P->property(ci->index(), i, 7));
  // Point dPoint = (_P->property(ci->index(), i, 6));
  double h = zParam[0]; // Channel height
  // double w = dPoint[1]; // Channel width
  double w = 0.0005;
  double sT = dPoint[0]; // Particle distance from boundary
  double H = w;
  double uM = uMax;
  double particleRadius = particleDiameter / 2.0;
  

  int s = ((sT / w) * 100.0);



  // If Z axis, use Z axis height and P position
  //  Assumed is Z is a constant height
  if (i == gdim-1)
  {
    H = h;
    double zMin = zParam[1]; // Z minimum height
    // Normalise Z position
    double sTT = (pPos[2] - zMin);
    sTT /= h;
    sTT *= 100.0;
    s = sTT;
    std::cout << "pPos[2]: " << pPos[2] << std::endl;
    std::cout << "Za axis S value: " << ((pPos[2] - 0.0110585) / 0.00016) << std::endl;
    std::cout << "Za axis S value: " << sTT << std::endl;
  }

  // Hydraulic Diameter of rectangle channel
  H = (w * h);
  H *= 2;
  H /= (w + h);
  double k = (particleRadius) / (H);



  // Stopped the fail due to s = 0 causing inf bug
  if (s < 1)
  {
    s = 1;
  } else if (s > 99)
  {
    s = 99;
  }
  
  int sU = 1;
  if (s > 50)
  {
    s = 100 - s;
    sU = -1;
  }

  // std::cout << "Flow uNorm: " << uNorm << std::endl;
  std::cout << "P distance from boundary: " << s << std::endl;
  
  // Invert for the GSpot as index 50 is 0 distance from wall
  s = 50 - s;

  
  //// Ho and Leal 1974 5.27
  double Flnl = pow( (particleRadius) , 2); // a
  Flnl *= pow( ( k ) , 2);
  Flnl *= flowDensity; // Reynolds;
  Flnl *= pow( uM , 2); // Max Velocity


  // Add ratio of X:Y vectors
  // double ratioU = 1;
  // Total magnitude of XY is...
  double MagXY = sqrt(pow( up[0], 2) + pow( up[1], 2));
  // Total magnitude of XY (face norm) is...
  // double MagXY = sqrt(pow( P4[0], 2) + pow( P4[1], 2));//P4[i]

  if (i == 0) // Direction lift to X
  {    
    // up1 is the direction along the X or Y axis the particle is moving
    //  This assumes a parabolic where flow drops off towards the boundaries
    //  Therefore, particle movement should be in the opposite direction to movement
    Flnl *= (std::abs(up[1]) / MagXY);

    if (P4[0] < 0)//(P4[i] < 0) // Negative along axis
    {
      Flnl *= -1;
    }
  } else if (i == 1) // Direction lift to Y
  {
    // up1 is the direction along the X or Y axis the particle is moving
    //  This assumes a parabolic where flow drops off towards the boundaries
    //  Therefore, particle movement should be in the opposite direction to movement
    Flnl *= (std::abs(up[0]) / MagXY);

    if (P4[i] < 0)//(P4[i] < 0) // Negative along axis
    {
      Flnl *= -1;
    }
  } else // Direction lift to Z
  { 
    Flnl *= sU;
  }


  //// Ho & Leal 1974 Lift Coefficient parabolic 2D flow
  //  Using the defined G1 and G2 earlier
  //  & shear rate and shear gradient
  double beta, Lander = 0;

  if (C_U_G.x() == 0) // Calculate Shear Gradient and Rate of Change
  {                   //  based upon Ho & Leal 1974
    // beta = 4.0 * (1.0 - (2.0 * (s * 0.01))); // * uM
    beta = (50.0 - s) * 0.01;
    beta *= 2.0;
    beta = 1.0 - beta;
    beta *= 4.0;
    Lander = -4.0;
  } else // Calculate Shear Gradient and Rate of Change from fluid
  {
    double Gxy = C_U_G[1];
    double Gxz = C_U_G[2];

    double Gyx = C_V_G[0];
    double Gyz = C_V_G[2];

    double Gzx = C_W_G[0];
    double Gzy = C_W_G[1];

    double U, G = 0;
    if (i == 0) // Add lift to X
    {
      U = sqrt(pow(up[1], 2) + pow(up[2], 2));
      // double G = sqrt(pow( (Gxz), 2) + pow( (Gxy), 2));
      G = sqrt(pow( (Gzx), 2) + pow( (Gyx), 2));
      U *= C_U_G[0] / std::abs(C_U_G[0]);
    } else if (i == 1) // Add lift to Y
    {
      U = sqrt(pow(up[0], 2) + pow(up[2], 2));
      // double G = sqrt(pow(1/Gxy, 2) + pow( (Gyz), 2));
      G = sqrt(pow(Gxy, 2) + pow( (Gzy), 2));
      U *= C_V_G[1] / std::abs(C_V_G[1]);
    } else
    {
      U = sqrt(pow(up[0], 2) + pow(up[1], 2));
      // double G = sqrt(pow(1/Gxz, 2) + pow(1/Gyz, 2));
      G = sqrt(pow(Gxz, 2) + pow(Gyz, 2));
    }
    //// Shear Gradient - Velocity perpendicular over a distance
    beta = std::abs(G * (H / uM));
    // double beta = G * (H / uM);
  
    //// Shear rate - Velocity perpendicular to fluid flow
    Lander = pow(U, 2) / pow(uM, 2);
  }

  
  double CL = ( (( pow( beta, 2) * GSpot[0][s] ))
                + (( beta * Lander * GSpot[1][s] )) );
  Flnl *= CL;


  //// Convert force to acceleration
  // F = m*a == a = F/m
  // m = volume(Sphere) * density == 1/6 pi D^3 * Particle Density
  double PI = 3.14159265358979323846;
  double mass = (PI * pow( particleDiameter, 3 )); // Volume
  // std::cout << "mass1: " << mass << std::endl;
  mass /= 6.0;
  // std::cout << "mass2: " << mass << std::endl;
  mass *= particleDensity; // mass
  // std::cout << "mass3: " << mass << std::endl;
  Flnl /= mass; // F/m = Acceleration
  std::cout << "Flnl: " << Flnl << std::endl;

  return Flnl;
}

//-----------------------------------------------------------------------------
std::tuple<Point, Point, Point> advect_particles::cal_ShearGradient(Point& up,
                        Point& pPos, double gdim, Eigen::MatrixXd basis_mat,
                        const dolfin::Cell ci, Eigen::Map<Eigen::VectorXd> exp_coeffs,
                        std::shared_ptr<const FiniteElement> _element)
{
  Point C_U_G, C_V_G, C_W_G = up * 0;
  // Point C_V_G = up * 0;
  // Point C_W_G = up * 0;
  double pStep = 1e-5; // 10 micrometre

  //// Calculate flow gradient for axis
  // Based on the C_U_G it would suggest shear of U velocity
  //  on the X, Y and Z axis
  // Due to axis implemented as pPos[iI], where iI = 0 to 2
  //  signifying X, Y, Z position, iI will use axis not velocity
  for (std::size_t iI = 0; (iI < gdim); iI++)
  {
    // Reset point
    Point pPos1 = pPos;
    pPos1[iI] += pStep; // Add to axis

    // Retrieve velocity at new point
    Utils::return_basis_matrix(basis_mat.data(), pPos1, ci,
                                _element);
    Eigen::VectorXd u_p1 = basis_mat * exp_coeffs;

    Point up1(gdim, u_p1.data());

    //// Velocity gradient vector

    // Caluclate individual axis gradient force
    // double G = std::abs(up1[iI]) - std::abs(up[iI]);
    std::cout << "up1: " << std::abs(up1[iI]) << std::endl;
    std::cout << "up: " << std::abs(up[iI]) << std::endl;

    C_U_G[iI] = (up1[0] - up[0]) / pStep;
    C_V_G[iI] = (up1[1] - up[1]) / pStep;
    C_W_G[iI] = (up1[2] - up[2]) / pStep;
  }

  return {C_U_G, C_V_G, C_W_G};
}
// //-----------------------------------------------------------------------------
// double advect_particles::cal_WallLiftSq(double dynVisc, 
//   double particleDiameter, double particleDensity, double flowDensity,
//   Point& zParam, double uMax, int i, int gdim, Point& up,
//   Point& pPos, Point& dPoint, Point& P4, Point& C_U_G, Point& C_V_G,
//   Point& C_W_G)
// {

//   // Initialise G1 and G2 for lift constants
//   //  GSpot is based upon
//   //    Ho, B. P., & Leal, L. G. (1974).
//   //      Inertial migration of rigid spheres in
//   //      two-dimensional unidirectional flows.
//   //      Journal of Fluid Mechanics, 65(2), 365–400.
//   //    https://doi.org/10.1017/S0022112074001431
//   //  G1 in slot 0, G2 in slot 1
//   //  The index is 0.01 step, index start at 0
//   //    i.e. 49 is 0.50
//   double GSpot[2][50] =
//   { 
//     { 0,
//       0.0419,
//       0.0837,
//       0.1254,
//       0.1669,
//       0.208,
//       0.2489,
//       0.2894,
//       0.3293,
//       0.3688,
//       0.4077,
//       0.4459,
//       0.4834,
//       0.52,
//       0.556,
//       0.591,
//       0.626,
//       0.659,
//       0.691,
//       0.723,
//       0.753,
//       0.782,
//       0.81,
//       0.836,
//       0.861,
//       0.885,
//       0.907,
//       0.927,
//       0.945,
//       0.96,
//       0.973,
//       0.982,
//       0.988,
//       0.99,
//       0.988,
//       0.981,
//       0.971,
//       0.957,
//       0.943,
//       0.931,
//       0.927,
//       0.94,
//       0.982,
//       1.07,
//       1.23,
//       1.5,
//       1.93,
//       2.58,
//       3.59,
//       5.33
//       },
//     { 1.072,
//       1.07,
//       1.068,
//       1.066,
//       1.062,
//       1.056,
//       1.05,
//       1.042,
//       1.033,
//       1.023,
//       1.012,
//       1.0,
//       0.987,
//       0.972,
//       0.956,
//       0.94,
//       0.922,
//       0.902,
//       0.882,
//       0.861,
//       0.838,
//       0.815,
//       0.79,
//       0.765,
//       0.738,
//       0.711,
//       0.683,
//       0.654,
//       0.625,
//       0.596,
//       0.566,
//       0.536,
//       0.506,
//       0.477,
//       0.448,
//       0.420,
//       0.393,
//       0.368,
//       0.345,
//       0.324,
//       0.306,
//       0.292,
//       0.282,
//       0.278,
//       0.28,
//       0.291,
//       0.315,
//       0.354,
//       0.414,
//       0.505
//     }
//   };

//   // Calculate relative speed of particle and flow
//   //    Require for stress rate and shear gradient
//   // Point P4 = (_P->property(ci->index(), i, 7));
//   // Point dPoint = (_P->property(ci->index(), i, 6));
//   double h = zParam[0]; // Channel height
//   // double w = dPoint[1]; // Channel width
//   double w = 0.0005;
//   double sT = dPoint[0]; // Particle distance from boundary
//   double H = w;
//   // double uMax = up.norm(); // double uNorm = up.norm();
//   //// uM is Mean velocity in Ho and Leal is assumed to be
//   ////    2/3 max flow velocity
//   // double uM = 2.0 * uMax ;
//   // uM /= 3.0;
//   double uM = uMax;
//   double particleRadius = particleDiameter / 2.0;
  

//   int s = ((sT / w) * 100.0);



//   // If Z axis, use Z axis height and P position
//   //  Assumed is Z is a constant height
//   if (i == gdim-1)
//   {
//     H = h;
//     double zMin = zParam[1];
//     double sTT = (pPos[2] - zMin);
//     // k = (particleRadius) / (H);
//     // if ( (sTT > (H - (particleDiameter / 2) ) ) || (sTT < (particleDiameter / 2) ) )
//     //   {
//     //     sTT = particleDiameter / 10;
//     //   }
//     sTT /= h;
//     sTT *= 100.0;
//     s = sTT;
//     std::cout << "pPos[2]: " << pPos[2] << std::endl;
//     std::cout << "Za axis S value: " << ((pPos[2] - 0.0110585) / 0.00016) << std::endl;
//     std::cout << "Za axis S value: " << sTT << std::endl;
//     // uMax = 0.022;//0.0022;//0.000005; // Bouyancy acting upon particle
//     // uMax = 0.000005;
//     // uMax = 200; // uMax = 0.8; 1000 100
//     // sT = (s * 0.01 * H);
//   }

//   // Hydraulic Diameter of rectangle channel
//   // double A = w * h; // channel area
// 	// double P = (w * 2) + (h * 2); // channel perimeter

// 	// H = 4 * A / P; // hydraulic
// 	// double Re = flowDensity * Um * Dh / (dynVisc * A);
// 	// double kappa = particle_diameter / H; // blockage ratio
//   H = (w * h);
//   H *= 2;
//   H /= (w + h);
//   double k = (particleRadius) / (H);
//   // double k = (particleDiameter) / (H);



//   // Stopped the fail due to s = 0 causing inf bug
//   if (s < 1)
//   {
//     s = 1;
//   } else if (s > 99)
//   {
//     s = 99;
//   }
  
//   int sU = 1;
//   if (s > 50)
//   {
//     s = 100 - s;
//     sU = -1;
//   }

//   // std::cout << "Flow uNorm: " << uNorm << std::endl;
//   std::cout << "P distance from boundary: " << s << std::endl;
//   // std::cout << "k: " << k << std::endl;
//   // Invert for the GSpot as index 50 is 0 distance from wall
//   s = 50 - s;


//   //// Lift force
//   // CL fitting parameter, determined to be 0.1 experimentally
//   // double CL = 0.1;

  
//   //// Ho and Leal 1974 5.27
//   // double Flnl = pow( (particleDiameter) , 4); // a
//   // Flnl *= pow( (particleDiameter) , 2); // a
//   double Flnl = pow( (particleRadius) , 2); // a
//   Flnl *= pow( ( k ) , 2);
//   Flnl *= flowDensity; // Reynolds;
//   Flnl *= pow( uM , 2); // Max Velocity
//   // Flnl *= CL;
//   // Flnl /= (H * Um);
//   // cal_CL(W/H, Re, kappa, s, &CL, i)
  
//   //// Shear rates based on Gab,
//   //  a is inc. in pPos
//   //  b is vel. vector 

//   double Gxy = C_U_G[1];
//   double Gxz = C_U_G[2];

//   double Gyx = C_V_G[0];
//   double Gyz = C_V_G[2];

//   double Gzx = C_W_G[0];
//   double Gzy = C_W_G[1];

//   double U, G = 0;


//   // Add ratio of X:Y vectors
//   // double ratioU = 1;
//   // Total magnitude of XY is...
//   double MagXY = sqrt(pow( up[0], 2) + pow( up[1], 2));
//   // Total magnitude of XY (face norm) is...
//   // double MagXY = sqrt(pow( P4[0], 2) + pow( P4[1], 2));//P4[i]

//   if (i == 0) // Add lift to X
//   {
//     U = sqrt(pow(up[1], 2) + pow(up[2], 2));
//     // double G = sqrt(pow( (Gxz), 2) + pow( (Gxy), 2));
//     G = sqrt(pow( (Gzx), 2) + pow( (Gyx), 2));
//     U *= C_U_G[0] / std::abs(C_U_G[0]);
//     // Flnl *= ( (G * pow(U, 2)) / (uM * H));
    
//     // up1 is the direction along the X or Y axis the particle is moving
//     //  This assumes a parabolic where flow drops off towards the boundaries
//     //  Therefore, particle movement should be in the opposite direction to movement
//     Flnl *= (std::abs(up[1]) / MagXY);
//     // Flnl *= (std::abs(P4[i]) / MagXY);
    
//     // Use G for flow gradient?

//     if (P4[0] < 0)//(P4[i] < 0) // Negative along axis
//     {
//       // std::cout << "Flnl: " << Flnl << std::endl;
//       // up[ii] += ( lift * (std::abs(up[iP]) / ratioU) );// * 10000;
//       // up[ii] += std::abs((lift * (std::abs(up[iP]) / ratioU) ) * ForceBal * (exp(-dt/ForceBal) - 1));
//       Flnl *= -1;
//     }
//   } else if (i == 1) // Add lift to Y
//   {
//     U = sqrt(pow(up[0], 2) + pow(up[2], 2));
//     // double G = sqrt(pow(1/Gxy, 2) + pow( (Gyz), 2));
//     G = sqrt(pow(Gxy, 2) + pow( (Gzy), 2));
//     U *= C_V_G[1] / std::abs(C_V_G[1]);
//     // Flnl *= ( (G * pow(U, 2)) / (uM * H));

//     //  Using the defined G1 and G2 earlier
//     //  & shear rate and shear gradient
//     // double CL = ( ( pow( G, 2) * GSpot[0][s] ) + (G * U * GSpot[1][s] ) ) / (uM * H) ;
//     // Flnl *= CL;

//     // Add ratio of X:Y vectors
//     // double ratioU = 1;
//     // Total magnitude of XY is...
//     // double MagXY = sqrt(pow( up[0], 2) + pow( up[1], 2));
//     // Total magnitude of XY (face norm) is...
//     // double MagXY = sqrt(pow( P4[0], 2) + pow( P4[1], 2));//P4[i]
    
//     // up1 is the direction along the X or Y axis the particle is moving
//     //  This assumes a parabolic where flow drops off towards the boundaries
//     //  Therefore, particle movement should be in the opposite direction to movement
//     Flnl *= (std::abs(up[0]) / MagXY);
//     // Flnl *= (std::abs(P4[i]) / MagXY);
//     if (P4[1] < 0)//(P4[i] < 0) // Negative along axis
//     {
//       // std::cout << "Flnl: " << Flnl << std::endl;
//       // up[ii] += ( lift * (std::abs(up[iP]) / ratioU) );// * 10000;
//       // up[ii] += std::abs((lift * (std::abs(up[iP]) / ratioU) ) * ForceBal * (exp(-dt/ForceBal) - 1));
//       Flnl *= -1;
//     }
//   } else { // Add lift to Z
//     U = sqrt(pow(up[0], 2) + pow(up[1], 2));
//     // double G = sqrt(pow(1/Gxz, 2) + pow(1/Gyz, 2));
//     G = sqrt(pow(Gxz, 2) + pow(Gyz, 2));
//     // U *= C_W_G[2] / std::abs(C_W_G[2]);
//     // Flnl *= ( (G * pow(U, 2)) / (uM * H));

//     // //  Using the defined G1 and G2 earlier
//     // //  & shear rate and shear gradient
//     // double CL = ( ( pow( G, 2) * GSpot[0][s] ) + (G * U * GSpot[1][s] ) ) / (uM * H) ;
//     // Flnl *= CL;

//     Flnl *= sU;
//     // if (P4[2] < 0)//(P4[i] < 0) // Negative along axis
//     // {
//       // std::cout << "Flnl: " << Flnl << std::endl;
//       // up[ii] += ( lift * (std::abs(up[iP]) / ratioU) );// * 10000;
//       // up[ii] += std::abs((lift * (std::abs(up[iP]) / ratioU) ) * ForceBal * (exp(-dt/ForceBal) - 1));
//       // Flnl *= -1;
//     // }
//     // Flnl = 0;
//     // std::cout << "Flnl(z): " << Flnl << std::endl;
//   }

//   //  Using the defined G1 and G2 earlier
//   //  & shear rate and shear gradient
//   // double CL = ( ( ( pow( G, 2) * GSpot[0][s] ) / pow(uM,2) )
//   //               + ( (G * (U * H) * GSpot[1][s] ) / (uM) ) );
//   // double beta = U * G * H;
//   //// Shear Gradient - Velocity perpendicular over a distance
//   // double beta = std::abs(G * (H / uM));
//   double beta = G * (H / uM);
  
//   // //// Shear rate - Velocity perpendicular to fluid flow
//   double Lander = pow(U, 2) / pow(uM, 2);

//   // Check negative / positive velocity gradient
//   //    important for lift across channel

//   // double Lander = pow(beta, 2); 
//   // double Lander = U / uM;

  

//   //// Ho & Leal 1974 Lift Coefficient parabolic 2D flow
//   // beta = 4.0 * (1.0 - (2.0 * (s * 0.01))); // * uM

//   beta = (50.0 - s) * 0.01;
//   beta *= 2.0;
//   // std::cout << "beta: " << beta << std::endl;
//   beta = 1.0 - beta;
//   // std::cout << "beta: " << beta << std::endl;
//   beta *= 4.0;
//   Lander = -4.0; // * uM;
//   // Lander = 0;

//   // std::cout << "beta: " << beta << std::endl;
//   // std::cout << "Lander: " << Lander << std::endl;
//   // std::cout << "G1: " << GSpot[0][s] << std::endl;
//   // std::cout << "G2: " << GSpot[1][s] << std::endl;

//   // Lander *= U / std::abs(U); // Find Lander direction
  
//   // double CL = ( (( pow( beta, 2) * GSpot[0][s] ))
//   //               - (( beta * Lander * GSpot[1][s] )) );

//   double CL = ( (( pow( beta, 2) * GSpot[0][s] ))
//                 + (( beta * Lander * GSpot[1][s] )) );
//   Flnl *= CL;

//   // Flnl *= G * pow(U, 2);
//   std::cout << "CL: " << CL << std::endl;

//   //// Convert force to acceleration
//   // F = m*a == a = F/m
//   // m = volume(Sphere) * density == 1/6 pi D^3 * Particle Density
//   double PI = 3.14159265358979323846;
//   double mass = (PI * pow( particleDiameter, 3 )); // Volume
//   // std::cout << "mass1: " << mass << std::endl;
//   mass /= 6.0;
//   // std::cout << "mass2: " << mass << std::endl;
//   mass *= particleDensity; // mass
//   // std::cout << "mass3: " << mass << std::endl;
//   Flnl /= mass; // F/m = Acceleration
//   std::cout << "Flnl: " << Flnl << std::endl;

//   // // Ho & Leal (1974) Lateral Particle Velocity Eq. 6.1
//   // double mass = (6 * PI * dynVisc * particleRadius);
//   // double mass = 6 * PI * dynVisc;
//   // Flnl /= mass; // F/m = Acceleration
//   // // Calculate Particle Reynolds (using dimensionless Um)
//   // double Reynolds = particleDensity * uMax * particleRadius * k;
//   // Reynolds /= dynVisc;
//   // Flnl /= Reynolds; // Usz * Re = Flnl

//   return Flnl;
// }

//-----------------------------------------------------------------------------
// Point advect_particles::VirtualMass(double flowDensity, double particleDensity,
//                                     Point u, Point up, double dT)
// {
//   // Virtual mass accounts for the fluid velocity around a particle

//   double VM = 0.5 * (flowDensity / particleDensity) * ((u - up) / dT);

//   return VM;
// }

//-----------------------------------------------------------------------------
// Point advect_particles::cal_ParticleDistToBoundary(const Mesh bmesh, Point pPos,
//   double Zmid)
// {

//   Zmid *= 100;

//   // Create boundary mesh
//   // mesh bmesh = BoundaryMesh(mesh, "exterior");
//   // Mesh bmesh = BoundaryMesh::BoundaryMesh();
//   // Mesh bmesh = BoundaryMesh(mesh,"exterior");

//   // // Scale mesh Z axis
//   // std::vector<double> x = bmesh.coordinates();
//   // x[:, 2] *= 100;

//   // Creating bonding box tree of mesh
//   //     where Z axis is very large
//   BoundingBoxTree bbtree;
//   bbtree.build(bmesh);

//   // Store particle position
//   Point P2 = Point(pPos[0], pPos[1], Zmid);

//   // Distance to closest XY exterior boundary 
//   // std::pair< unsigned int, double >
//   std::pair<int, double> d = bbtree.compute_closest_entity(P2);
//   std::cout << "P distance from boundary: " << d.second << std::endl;

//   // std::vector<double>& all_cells = bmesh.cells();

//   std::cout << "bmesh cells: " << all_cells << std::endl;

//   // Cell_ID
//   // std::vector<double> all_cells = bmesh.Cell(d.first).get_vertex_coordinates

//   // Find vertices from mesh to find co-ordinate position of closet entity
//   // std::vector<double> all_cells = bmesh.coordinates(); Will not be accessable

//   // std::vector<unsigned int> all_cells = bmesh.cells();
//   // std::vector<double>& all_cells = bmesh.cells();

//   //bmesh.cells()[d.first];

//   // std::size_t d1 = d.first;

//   // std::vector<double> closest_cell = all_cells(d1);
//   // vertices_of_closest_cell = mesh.Vertex(d.first).coordinate()
//   // vertices_of_closest_cell = bmesh.coordinates();

//   // Temporary D1 distance
//   double D1 = 100;

//   // P3 Particle position without Z axis
//   Point P3 = Point(P2.x(), P2.y(), 0)

//   // for i in vertices_of_closest_cell:
//   for (unsigned int i = 0; i < vertices_of_closest_cell; i++)
//     {
//     // P1 verticies without Z axis
//     P1 = Point(i[0], i[1], 0);
//     // Calculate nearest point
//     D = P3.distance(P1);

//     if D1 > D:
//       D1 = D;
//       // Save distance as vector
//       P4 = P3 - P1;
        
//   // print(P4.x(), P4.y())

//   // Add distance to point to current point
//   P3 += P4;

//   P3 = Point(P3.x(), P3.y(), Zmid);

//   d1, distance1 = bbtree.compute_closest_entity(P3);

//   Point dPoint = Point((distance), ((distance * 2) + distance1), 0);
//   // print("dPoint: ", dPoint.x(), dPoint.y(), dPoint.z())

//   // print("Distance: ", distance)
//   // print("Distance 1: ", (distance + distance1))
//   // print("Total Distance: ", ((distance * 2) + distance1))

//   return dPoint;
// }

//-----------------------------------------------------------------------------
void advect_particles::do_step(double dt,
  Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> LPTParameters)
{
  init_weights();

  const Mesh* mesh = _P->mesh();
  const MPI_Comm mpi_comm = mesh->mpi_comm();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology().dim();

  std::size_t num_processes = MPI::size(mpi_comm);

  // Needed for local reloc
  std::vector<std::array<std::size_t, 3>> reloc;

  const Function& uh_step = uh(0, dt);

  for (CellIterator ci(*mesh); !ci.end(); ++ci)
  {
    std::vector<double> coeffs;
    // Restrict once per cell, once per timestep
    Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);

    // Loop over particles in cell
    for (unsigned int i = 0; i < _P->num_cell_particles(ci->index()); i++)
    {
      // FIXME: It might be better to use 'pointer iterator here' as we need to
      // erase from cell2part vector now we decrement iterator int when needed

      Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
      Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                 _element);

      // Compute value at point using expansion coeffs and basis matrix, first
      // convert to Eigen matrix
      Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), _space_dimension);
      Eigen::VectorXd u_p = basis_mat * exp_coeffs;


      Point up(gdim, u_p.data());

      std::cout << "P Pos: " <<  _P->x(ci->index(), i) << std::endl;
      std::cout << "Flow Vel: " <<  up << std::endl;

      // Current Particle velocity
      Point up_1 = (_P->property(ci->index(), i, 1));
      std::cout << "P Vel Curr: " << up_1 << std::endl;


      double relax = this->cal_relax(
        LPTParameters[3], LPTParameters[0], LPTParameters[1]);

      // Cal Particle Reynolds
      double reynolds = cal_reynolds(LPTParameters[3],
        LPTParameters[0], LPTParameters[2], up[0], up_1[0]);

      double drag = this->cal_drag(reynolds);

      // Drag force balance for LPT
      double ForceBalance1 = ((drag * reynolds) / 24);
      ForceBalance1 *= (relax);
      ForceBalance1 = 1 / ForceBalance1;

      Point Acceleration = up;
      Point ForceBalance = up;
      Point pPos = _P->x(ci->index(), i);

      // Particle positive or negative along axis (ind. as vector)
      Point P4 = (_P->property(ci->index(), i, 4));
      // Particle distance from boundary, Boundary distance, 0
      Point dPoint = (_P->property(ci->index(), i, 3));

      Acceleration *= 0;
      ForceBalance *= 0;
      ForceBalance *= ForceBalance1;
      for (std::size_t iI = 0; (iI < gdim); iI++)
      {
        if (LPTParameters[6] == 0)
        {
          Point zParam = Acceleration;
          zParam[0] = LPTParameters[4]; // channel height
          zParam[1] = LPTParameters[5]; // minimum z
          // double uMax = LPTParameters[6];
          // if (LPTParameters[6] == 0)
          // {
            // pPos = cal_ParticleDistToBoundary(bmeshXY, Zmid)
            // Accelerations due to all other forces except drag force
            // Acceleration[iI] = this->cal_WallLiftSq(LPTParameters[3],
            //   LPTParameters[0], LPTParameters[2], reynolds, iI, up, up1,
            //   pPos, gdim, LPTParameters[4], LPTParameters[5], dt);


          // Acceleration[iI] = this->cal_WallLiftSq(LPTParameters[3],
          //   LPTParameters[0], LPTParameters[1], LPTParameters[2],
          //   zParam, uMax, iI, gdim, up, pPos, dPoint, P4);
          Acceleration[iI] = 0;
        }
      }

      // if (LPTParameters[6] == 0)
      // {
      //   // Add s values from Wall Induced Lift Force
      //   dt = DEFINE_DPM_TIMESTEP(dt, up, up_1, LPTParameters, pPos, dPoint);
      // }

      std::cout << "P Acceleration: " << Acceleration << std::endl;
      std::cout << "P ForceBalance: " << ForceBalance1 << std::endl;
    

      std::size_t cidx_recv = ci->index();
      double dt_rem = dt;

      while (dt_rem > 1E-15)
      {
        // Returns facet which is intersected and the time it takes to do so
        std::tuple<std::size_t, double> intersect_info
            = time2intersect(cidx_recv, dt_rem, _P->x(ci->index(), i), up);
        const std::size_t target_facet = std::get<0>(intersect_info);
        const double dt_int = std::get<1>(intersect_info);

        if (target_facet == std::numeric_limits<unsigned int>::max())
        {
          // Then remain within cell, finish time step
          // _P->push_particle(dt_rem, up, ci->index(), i);
          _P->push_particleLPT(dt_rem, up, ci->index(), i, ForceBalance,
            Acceleration, up_1);
          // _P->push_particleLPT(dt_rem, up, ci->index(), i, upLPT);
          // _P->push_particleLPT(dt_rem, up, ci->index(), i, ForceBalance,
          //   Acceleration);
          dt_rem = 0.0;

          if (bounded_domain_active)
            bounded_domain_violation(ci->index(), i);

          // TODO: if step == last tstep: update particle position old to most
          // recent value If cidx_recv != ci->index(), particles crossed facet
          // and hence need to be relocated
          if (cidx_recv != ci->index())
            reloc.push_back({ci->index(), i, cidx_recv});
        }
        else
        {
          const Facet f(*mesh, target_facet);
          const unsigned int* facet_cells = f.entities(tdim);

          // Two options: if internal (==2) else if boundary
          if (f.num_entities(tdim) == 2)
          {
            // Then we cross facet which has a neighboring cell
            // _P->push_particle(dt_int, up, ci->index(), i);
            _P->push_particleLPT(dt_rem, up, ci->index(), i, ForceBalance,
              Acceleration, up_1);

            cidx_recv = (facet_cells[0] == cidx_recv) ? facet_cells[1]
                                                      : facet_cells[0];

            // Update remaining time
            dt_rem -= dt_int;
            if (dt_rem < 1E-15)
            {
              // Then terminate
              dt_rem = 0.0;
              if (cidx_recv != ci->index())
                reloc.push_back({ci->index(), i, cidx_recv});
            }
          }
          else if (f.num_entities(tdim) == 1)
          {
            const facet_t ftype = facets_info[target_facet].type;
            // Then we hit a boundary, but which type?
            if (f.num_global_entities(tdim) == 2)
            {
              assert(ftype == facet_t::internal);
              // Then it is an internal boundary
              // Do a full push
              // _P->push_particle(dt_rem, up, ci->index(), i);
              _P->push_particleLPT(dt_rem, up, ci->index(), i, ForceBalance,
                Acceleration, up_1);
              // _P->push_particleLPT(dt_rem, up, ci->index(), i, ForceBalance,
              //   Acceleration);
              dt_rem *= 0.;

              if (pbc_active)
                pbc_limits_violation(ci->index(),
                                     i); // Check on sequence crossing internal
                                         // bc -> crossing periodic bc

              if (bounded_domain_active)
                bounded_domain_violation(ci->index(), i);

              // TODO: do same for closed bcs to handle (unlikely event):
              // internal bc-> closed bc

              // Go to the particle communicator
              reloc.push_back(
                  {ci->index(), i, std::numeric_limits<unsigned int>::max()});
            }
            else if (ftype == facet_t::open)
            {
              // Particle leaves the domain. Simply erase!
              // FIXME: additional check that particle indeed leaves domain
              // (u\cdotn > 0)
              // Send to "off process" (should just disappear)
              //
              // Issue 12 Work around: do a full push to make sure that
              // particle is pushed outside domain
              // _P->push_particle(dt_rem, up, ci->index(), i);
              _P->push_particleLPT(dt_rem, up, ci->index(), i, ForceBalance,
                Acceleration, up_1);
              // _P->push_particleLPT(dt_rem, up, ci->index(), i, ForceBalance,
              //   Acceleration);

              // Then push back to relocate
              reloc.push_back(
                  {ci->index(), i, std::numeric_limits<unsigned int>::max()});
              dt_rem = 0.0;
            }
            else if (ftype == facet_t::closed)
            {
              // Closed BC
              apply_closed_bc(dt_int, up, ci->index(), i, target_facet);
              dt_rem -= dt_int;
            }
            else if (ftype == facet_t::periodic)
            {
              // Then periodic bc
              apply_periodic_bc(dt_rem, up, ci->index(), i, target_facet);
              if (num_processes > 1) // Behavior in parallel
                reloc.push_back(
                    {ci->index(), i, std::numeric_limits<unsigned int>::max()});
              else
              {
                // Behavior in serial
                std::size_t cell_id = _P->mesh()
                                          ->bounding_box_tree()
                                          ->compute_first_entity_collision(
                                              _P->x(ci->index(), i));
                reloc.push_back({ci->index(), i, cell_id});
              }
              dt_rem = 0.0;
            }
            else if (ftype == facet_t::bounded)
            {
              std::cout << "Hit bounded facet " << std::endl;
              // Then bounded bc
              apply_bounded_domain_bc(dt_rem, up, ci->index(), i, target_facet);

              if (num_processes > 1) // Behavior in parallel
                reloc.push_back(
                    {ci->index(), i, std::numeric_limits<unsigned int>::max()});
              else
              {
                // Behavior in serial
                std::size_t cell_id = _P->mesh()
                                          ->bounding_box_tree()
                                          ->compute_first_entity_collision(
                                              _P->x(ci->index(), i));
                reloc.push_back({ci->index(), i, cell_id});
              }
              dt_rem = 0.0;
            }
            else
            {
              dolfin_error("advect_particles.cpp::do_step",
                           "encountered unknown boundary",
                           "Only internal boundaries implemented yet");
            }
          }
          else
          {
            dolfin_error("advect_particles.cpp::do_step",
                         "found incorrect number of facets (<1 or > 2)",
                         "Unknown");
          }
        } // end else
      }   // end while
    }     // end for
  }       // end for

  // Relocate local and global
  _P->relocate(reloc);
}
//-----------------------------------------------------------------------------
void advect_particles::do_step(double dt)
{
  init_weights();

  const Mesh* mesh = _P->mesh();
  const MPI_Comm mpi_comm = mesh->mpi_comm();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology().dim();

  std::size_t num_processes = MPI::size(mpi_comm);

  // Needed for local reloc
  std::vector<std::array<std::size_t, 3>> reloc;

  const Function& uh_step = uh(0, dt);

  for (CellIterator ci(*mesh); !ci.end(); ++ci)
  {
    std::vector<double> coeffs;
    // Restrict once per cell, once per timestep
    Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);

    // Loop over particles in cell
    for (unsigned int i = 0; i < _P->num_cell_particles(ci->index()); i++)
    {
      // FIXME: It might be better to use 'pointer iterator here' as we need to
      // erase from cell2part vector now we decrement iterator int when needed

      Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
      Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                 _element);

      // Compute value at point using expansion coeffs and basis matrix, first
      // convert to Eigen matrix
      Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), _space_dimension);
      Eigen::VectorXd u_p = basis_mat * exp_coeffs;

      // Convert velocity to point
      Point up(gdim, u_p.data());

      std::size_t cidx_recv = ci->index();
      double dt_rem = dt;

      while (dt_rem > 1E-15)
      {
        // Returns facet which is intersected and the time it takes to do so
        std::tuple<std::size_t, double> intersect_info
            = time2intersect(cidx_recv, dt_rem, _P->x(ci->index(), i), up);
        const std::size_t target_facet = std::get<0>(intersect_info);
        const double dt_int = std::get<1>(intersect_info);

        if (target_facet == std::numeric_limits<unsigned int>::max())
        {
          // Then remain within cell, finish time step
          _P->push_particle(dt_rem, up, ci->index(), i);
          dt_rem = 0.0;

          if (bounded_domain_active)
            bounded_domain_violation(ci->index(), i);

          // TODO: if step == last tstep: update particle position old to most
          // recent value If cidx_recv != ci->index(), particles crossed facet
          // and hence need to be relocated
          if (cidx_recv != ci->index())
            reloc.push_back({ci->index(), i, cidx_recv});
        }
        else
        {
          const Facet f(*mesh, target_facet);
          const unsigned int* facet_cells = f.entities(tdim);

          // Two options: if internal (==2) else if boundary
          if (f.num_entities(tdim) == 2)
          {
            // Then we cross facet which has a neighboring cell
            _P->push_particle(dt_int, up, ci->index(), i);

            cidx_recv = (facet_cells[0] == cidx_recv) ? facet_cells[1]
                                                      : facet_cells[0];

            // Update remaining time
            dt_rem -= dt_int;
            if (dt_rem < 1E-15)
            {
              // Then terminate
              dt_rem = 0.0;
              if (cidx_recv != ci->index())
                reloc.push_back({ci->index(), i, cidx_recv});
            }
          }
          else if (f.num_entities(tdim) == 1)
          {
            const facet_t ftype = facets_info[target_facet].type;
            // Then we hit a boundary, but which type?
            if (f.num_global_entities(tdim) == 2)
            {
              assert(ftype == facet_t::internal);
              // Then it is an internal boundary
              // Do a full push
              _P->push_particle(dt_rem, up, ci->index(), i);
              dt_rem *= 0.;

              if (pbc_active)
                pbc_limits_violation(ci->index(),
                                     i); // Check on sequence crossing internal
                                         // bc -> crossing periodic bc

              if (bounded_domain_active)
                bounded_domain_violation(ci->index(), i);

              // TODO: do same for closed bcs to handle (unlikely event):
              // internal bc-> closed bc

              // Go to the particle communicator
              reloc.push_back(
                  {ci->index(), i, std::numeric_limits<unsigned int>::max()});
            }
            else if (ftype == facet_t::open)
            {
              // Particle leaves the domain. Simply erase!
              // FIXME: additional check that particle indeed leaves domain
              // (u\cdotn > 0)
              // Send to "off process" (should just disappear)
              //
              // Issue 12 Work around: do a full push to make sure that
              // particle is pushed outside domain
              _P->push_particle(dt_rem, up, ci->index(), i);

              // Then push back to relocate
              reloc.push_back(
                  {ci->index(), i, std::numeric_limits<unsigned int>::max()});
              dt_rem = 0.0;
            }
            else if (ftype == facet_t::closed)
            {
              // Closed BC
              apply_closed_bc(dt_int, up, ci->index(), i, target_facet);
              dt_rem -= dt_int;
            }
            else if (ftype == facet_t::periodic)
            {
              // Then periodic bc
              apply_periodic_bc(dt_rem, up, ci->index(), i, target_facet);
              if (num_processes > 1) // Behavior in parallel
                reloc.push_back(
                    {ci->index(), i, std::numeric_limits<unsigned int>::max()});
              else
              {
                // Behavior in serial
                std::size_t cell_id = _P->mesh()
                                          ->bounding_box_tree()
                                          ->compute_first_entity_collision(
                                              _P->x(ci->index(), i));
                reloc.push_back({ci->index(), i, cell_id});
              }
              dt_rem = 0.0;
            }
            else if (ftype == facet_t::bounded)
            {
              std::cout << "Hit bounded facet " << std::endl;
              // Then bounded bc
              apply_bounded_domain_bc(dt_rem, up, ci->index(), i, target_facet);

              if (num_processes > 1) // Behavior in parallel
                reloc.push_back(
                    {ci->index(), i, std::numeric_limits<unsigned int>::max()});
              else
              {
                // Behavior in serial
                std::size_t cell_id = _P->mesh()
                                          ->bounding_box_tree()
                                          ->compute_first_entity_collision(
                                              _P->x(ci->index(), i));
                reloc.push_back({ci->index(), i, cell_id});
              }
              dt_rem = 0.0;
            }
            else
            {
              dolfin_error("advect_particles.cpp::do_step",
                           "encountered unknown boundary",
                           "Only internal boundaries implemented yet");
            }
          }
          else
          {
            dolfin_error("advect_particles.cpp::do_step",
                         "found incorrect number of facets (<1 or > 2)",
                         "Unknown");
          }
        } // end else
      }   // end while
    }     // end for
  }       // end for

  // Relocate local and global
  _P->relocate(reloc);
}
//-----------------------------------------------------------------------------
std::tuple<std::size_t, double>
advect_particles::time2intersect(std::size_t cidx, double dt, const Point xp,
                                 const Point up)
{
  // Time to facet intersection
  const Mesh* mesh = _P->mesh();
  const std::size_t tdim = mesh->topology().dim();
  double dt_int = std::numeric_limits<double>::max();
  std::size_t target_facet = std::numeric_limits<unsigned int>::max();

  Cell c(*mesh, cidx);
  for (unsigned int i = 0; i < c.num_entities(tdim - 1); ++i)
  {
    std::size_t fidx = c.entities(tdim - 1)[i];
    Facet f(*mesh, fidx);

    Point normal = facets_info[fidx].normal;

    // Normal points outward from Cell 0, so reverse if this is Cell 1 of the
    // Facet
    if (f.entities(tdim)[0] != cidx)
      normal *= -1.0;

    // Compute distance to point. For procedure, see Haworth (2010). Though it
    // is slightly modified
    double h = f.distance(xp);

    // double dtintd = std::max(0., h / (up.dot(normal)) ); //See Haworth
    double denom = up.dot(normal);
    if (denom > 0. && denom < 1e-8)
      denom *= -1.; // If up orth to normal --> blows up timestep

    double dtintd = h / denom;
    // TODO: is this robust for: 1) very small h? 2) infinite number?
    if ((dtintd < dt_int && dtintd > 0. && h > 1E-10)
        || (h < 1E-10 && denom > 0.))
    {
      dt_int = dtintd;
      // Then hit a face or located exactly at a face with non-zero velocity in
      // outward normal direction
      if (dt_int <= dt)
      {
        target_facet = fidx;
      }
    }
  }
  // Store and return intersect info in tuple
  std::tuple<std::size_t, double> intersect_info(target_facet, dt_int);
  return intersect_info;
}
//-----------------------------------------------------------------------------
void advect_particles::apply_open_bc(std::size_t cidx, std::size_t pidx)
{
  _P->delete_particle(cidx, pidx);
}
//-----------------------------------------------------------------------------
void advect_particles::apply_closed_bc(double dt, Point& up, std::size_t cidx,
                                       std::size_t pidx, std::size_t fidx)
{
  // First push particle
  _P->push_particle(dt, up, cidx, pidx);
  // Mirror velocity
  Point normal = facets_info[fidx].normal;
  up -= 2 * (up.dot(normal)) * normal;
  
  // Set rebound velocity vector for particle vector
  _P->set_property(cidx, pidx, 1, up );
  
  std::cout << "Rebound True: " << up << std::endl;
}
//-----------------------------------------------------------------------------
void advect_particles::apply_periodic_bc(double dt, Point& up, std::size_t cidx,
                                         std::size_t pidx, std::size_t fidx)
{
  const std::size_t gdim = _P->mesh()->geometry().dim();
  Point midpoint = facets_info[fidx].midpoint;
  std::size_t row_match = std::numeric_limits<unsigned int>::max();
  std::size_t row_friend;
  std::size_t component;
  bool hit = false;
  for (std::size_t i = 0; i < pbc_lims.size(); i++)
  {
    for (std::size_t j = 0; j < gdim; j++)
    {
      if (std::abs(midpoint[j] - pbc_lims[i][j * 2]) < 1E-10
          && std::abs(midpoint[j] - pbc_lims[i][j * 2 + 1]) < 1E-10)
      {
        // Then we most likely found a match, but check if midpoint coordinates
        // are in between the limits for the other coordinate directions
        hit = true;
        for (std::size_t k = 0; k < gdim; k++)
        {
          if (k == j)
            continue;
          // New formulation
          if (midpoint[k] <= pbc_lims[i][k * 2]
              || midpoint[k] >= pbc_lims[i][k * 2 + 1])
            hit = false;
        }
        if (hit)
        {
          row_match = i;
          component = j;
          goto break_me;
        }
      }
    }
  }

break_me:
  // Throw an error if rowmatch not set at this point
  if (row_match == std::numeric_limits<unsigned int>::max())
    dolfin_error("advect_particles.cpp::apply_periodic_bc",
                 "find matching periodic boundary info", "Unknown");
  // Column and matchin column come in pairs
  if (row_match % 2 == 0)
  {
    // Find the uneven friend
    row_friend = row_match + 1;
  }
  else
  {
    // Find the even friend
    row_friend = row_match - 1;
  }

  // For multistep/multistage (!?) schemes, you may need to copy the old
  // position before doing the actual push
  _P->push_particle(dt, up, cidx, pidx);

  // Point formulation
  Point x = _P->x(cidx, pidx);
  x[component] += pbc_lims[row_friend][component * 2]
                  - pbc_lims[row_match][component * 2];

  // Corners can be tricky, therefore include this test
  for (std::size_t i = 0; i < gdim; i++)
  {
    if (i == component)
      continue; // Skip this
    if (x[i] < pbc_lims[row_match][i * 2])
    {
      // Then we push the particle to the other end of domain
      x[i] += (pbc_lims[row_friend][i * 2 + 1] - pbc_lims[row_match][i * 2]);
    }
    else if (x[i] > pbc_lims[row_match][i * 2 + 1])
    {
      x[i] -= (pbc_lims[row_match][i * 2 + 1] - pbc_lims[row_friend][i * 2]);
    }
  }
  _P->set_property(cidx, pidx, 0, x);
}
//-----------------------------------------------------------------------------
void advect_particles::pbc_limits_violation(std::size_t cidx, std::size_t pidx)
{
  // This method guarantees that particles can cross internal bc -> periodic bc
  // in one time step without being deleted.
  // FIXME: more efficient implementation??
  // FIXME: can give troubles when domain decomposition results in one cell in
  // domain corner Check if periodic bcs are violated somewhere, if so, modify
  // particle position
  std::size_t gdim = _P->mesh()->geometry().dim();

  Point x = _P->x(cidx, pidx);

  for (std::size_t i = 0; i < pbc_lims.size() / 2; i++)
  {
    for (std::size_t j = 0; j < gdim; j++)
    {
      if (std::abs(pbc_lims[2 * i][2 * j] - pbc_lims[2 * i][2 * j + 1]) < 1E-13)
      {
        if (x[j] > pbc_lims[2 * i][2 * j] && x[j] > pbc_lims[2 * i + 1][2 * j])
        {
          x[j] -= (std::max(pbc_lims[2 * i][2 * j], pbc_lims[2 * i + 1][2 * j])
                   - std::min(pbc_lims[2 * i][2 * j],
                              pbc_lims[2 * i + 1][2 * j]));
          // Check whether the other bounds are violated, to handle corners
          // FIXME: cannot handle cases where domain of friend in one direction
          // is different from match, reason: looping over periodic bc pairs
          for (std::size_t k = 0; k < gdim; k++)
          {
            if (k == j)
              continue;
            if (x[k] < pbc_lims[2 * i][2 * k])
            {
              x[k] += (pbc_lims[2 * i + 1][2 * k + 1] - pbc_lims[2 * i][2 * k]);
            }
            else if (x[k] > pbc_lims[2 * i][2 * k + 1])
            {
              x[k] -= (pbc_lims[2 * i][2 * k + 1] - pbc_lims[2 * i + 1][2 * k]);
            }
          }
        }
        else if (x[j] < pbc_lims[2 * i][2 * j]
                 && x[j] < pbc_lims[2 * i + 1][2 * j])
        {
          x[j] += (std::max(pbc_lims[2 * i][2 * j], pbc_lims[2 * i + 1][2 * j])
                   - std::min(pbc_lims[2 * i][2 * j],
                              pbc_lims[2 * i + 1][2 * j]));
          // Check wheter the other bounds are violated, to handle corners
          for (std::size_t k = 0; k < gdim; k++)
          {
            if (k == j)
              continue;
            if (_P->x(cidx, pidx)[k] < pbc_lims[2 * i][2 * k])
            {
              x[k] += (pbc_lims[2 * i + 1][2 * k + 1] - pbc_lims[2 * i][2 * k]);
            }
            else if (x[k] > pbc_lims[2 * i][2 * k + 1])
            {
              x[k] -= (pbc_lims[2 * i][2 * k + 1] - pbc_lims[2 * i + 1][2 * k]);
            }
          }
        } // else do nothing
      }
    }
  }
  _P->set_property(cidx, pidx, 0, x);
}
//-----------------------------------------------------------------------------
void advect_particles::apply_bounded_domain_bc(
    double dt, Point& up, std::size_t cidx,
    std::size_t pidx, std::size_t fidx)
{
  // First push particle
  _P->push_particle(dt, up, cidx, pidx);

  Point x = _P->x(cidx, pidx);
  for (std::size_t i = 0; i < _P->mesh()->geometry().dim(); ++i)
  {
    const double xmin = bounded_domain_lims[i][0];
    const double xmax = bounded_domain_lims[i][1];
    x[i] = std::min(std::max(x[i], xmin), xmax);
  }

  _P->set_property(cidx, pidx, 0, x);
}
//-----------------------------------------------------------------------------
void advect_particles::bounded_domain_violation(
    std::size_t cidx, std::size_t pidx)
{
  // This method guarantees that particles can cross internal bc -> bounded bc
  // in one time step without being deleted.
  Point x = _P->x(cidx, pidx);
  for (std::size_t i = 0; i < _P->mesh()->geometry().dim(); ++i)
  {
    const double xmin = bounded_domain_lims[i][0];
    const double xmax = bounded_domain_lims[i][1];
    x[i] = std::min(std::max(x[i], xmin), xmax);
  }

  _P->set_property(cidx, pidx, 0, x);
}
//-----------------------------------------------------------------------------
void advect_particles::do_substep(
    double dt, Point& up, const std::size_t cidx, std::size_t pidx,
    std::size_t& step, const std::size_t num_steps,
    const std::size_t xp0_idx, const std::size_t up0_idx,
    std::vector<std::array<std::size_t, 3>>& reloc)
{
  double dt_rem = dt;

  const Mesh* mesh = _P->mesh();
  const std::size_t mpi_size = MPI::size(mesh->mpi_comm());
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology().dim();

  std::size_t cidx_recv = std::numeric_limits<unsigned int>::max();

  if (step == 0)
    cidx_recv = cidx;
  else
  {
    // The reason for doing this step is:
    // for the multistep (RK) schemes, the carried old position may not be the
    // same as the cell where the particle lives newest position is always
    // carried
    // TODO: Can we think of smarter implementation?
    cidx_recv = mesh->bounding_box_tree()->compute_first_entity_collision(
        _P->x(cidx, pidx));

    // One alternative might be:
    // Cell cell(*(_P->_mesh), cidx);
    // bool contain = cell.contains(_P->_cell2part[cidx][pidx][0])
    // If true  cidx_recv = cidx; and continue
    // if not: do entity collision

    // FIXME: this approach is robust for the internal points multistep schemes,
    // but what about multistage schemes and near closed/periodic bc's?
    if (cidx_recv == std::numeric_limits<unsigned int>::max())
    {
      _P->push_particle(dt_rem, up, cidx, pidx);
      if (pbc_active)
        pbc_limits_violation(cidx, pidx);

      if (bounded_domain_active)
        bounded_domain_violation(cidx, pidx);

      if (step == (num_steps - 1))
      {
        // Copy current position to old position
        // so something like
        _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));
      }
      // Apparently, this always lead to a communicate, but why?
      reloc.push_back({cidx, pidx, std::numeric_limits<unsigned int>::max()});
      return; // Stop right here
    }
  }

  // std::cout << "xp0_idx: " << xp0_idx << std::endl;

  bool hit_cbc = false; // Hit closed boundary condition (?!)
  while (dt_rem > 1E-15)
  {
    // Returns facet which is intersected and the time it takes to do so
    std::tuple<std::size_t, double> intersect_info
        = time2intersect(cidx_recv, dt_rem, _P->x(cidx, pidx), up);
    // std::cout << "time2intersect: " << std::get<0>(intersect_info) << std::endl;
    // std::cout << "time2intersect: " << std::get<1>(intersect_info) << std::endl;
    const std::size_t target_facet = std::get<0>(intersect_info);
    const double dt_int = std::get<1>(intersect_info);

    if (target_facet == std::numeric_limits<unsigned int>::max())
    {
      // Then remain within cell, finish time step
      _P->push_particle(dt_rem, up, cidx, pidx);
      dt_rem = 0.0;

      if (bounded_domain_active)
        bounded_domain_violation(cidx, pidx);

      if (step == (num_steps - 1))
        // Copy current position to old position
        _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

      // If cidx_recv != ci->index(), particles crossed facet and hence need to
      // be relocated
      if (cidx_recv != cidx)
        reloc.push_back({cidx, pidx, cidx_recv});
    }
    else
    {
      Facet f(*mesh, target_facet);
      const unsigned int* fcells = f.entities(tdim);

      // Two options: if internal (==2) else if boundary
      if (f.num_entities(tdim) == 2)
      {
        // Then we cross facet which has a neighboring cell
        _P->push_particle(dt_int, up, cidx, pidx);

        // Update index of receiving cell
        cidx_recv = (fcells[0] == cidx_recv) ? fcells[1] : fcells[0];

        // Update remaining time
        dt_rem -= dt_int;
        if (dt_rem < 1E-15)
        {
          // Then terminate
          dt_rem *= 0.;
          // Copy current position to old position
          if (step == (num_steps - 1))
            _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          if (cidx_recv != cidx)
            reloc.push_back({cidx, pidx, cidx_recv});
        }
      }
      else if (f.num_entities(tdim) == 1)
      {
        const facet_t ftype = facets_info[target_facet].type;
        // Then we hit a boundary, but which type?
        if (f.num_global_entities(tdim) == 2)
        { // Internal boundary between processes
          assert(ftype == facet_t::internal);
          _P->push_particle(dt_rem, up, cidx, pidx);
          dt_rem = 0.0;

          // Updates particle position if pbc_limits is violated
          if (pbc_active)
            pbc_limits_violation(cidx, pidx);

          if (bounded_domain_active)
            bounded_domain_violation(cidx, pidx);

          // Copy current position to old position
          if (step == (num_steps - 1) || hit_cbc)
            _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          reloc.push_back(
              {cidx, pidx, std::numeric_limits<unsigned int>::max()});

          return; // Stop right here
        }
        else if (ftype == facet_t::open)
        {
          // Particle leaves the domain. Relocate to another process (particle
          // will be discarded)

          // Issue 12 work around: do full push to push particle outside
          // domain
          _P->push_particle(dt_rem, up, cidx, pidx);

          // Then push back to relocate
          reloc.push_back(
              {cidx, pidx, std::numeric_limits<unsigned int>::max()});
          dt_rem = 0.0;
        }
        else if (ftype == facet_t::closed)
        {
          apply_closed_bc(dt_int, up, cidx, pidx, target_facet);
          dt_rem -= dt_int;

          // // TODO: CHECK THIS
          // dt_rem
          //     += (1. - dti[step]) * (dt / dti[step]); // Make timestep complete
          // // If we hit a closed bc, modify following, probably is first order:
          // // dt_rem = 0.0;

          // TODO: UPDATE AS PARTICLE!
          std::vector<double> dummy_vel(gdim,
                                        std::numeric_limits<double>::max());
          _P->set_property(cidx, pidx, up0_idx, Point(gdim, dummy_vel.data()));
          // std::cout << "up0_idx: " << xp0_idx << std::endl;

          std::tuple<std::size_t, double> intersect_info
              = time2intersect(cidx_recv, dt_rem, _P->x(cidx, pidx), up);
          // std::cout << "time2intersect: " << std::get<0>(intersect_info) << std::endl;
          // std::cout << "time2intersect: " << std::get<1>(intersect_info) << std::endl;
          // const std::size_t target_facet = std::get<0>(intersect_info);
          const double dt_int1 = std::get<1>(intersect_info);

          // Set tiem remaining to the time to hit next cell using up(reflected)
          dt_rem = dt_int1;
          // Push particle to the next cell
          _P->push_particle(dt_int, up, cidx, pidx);
          // Set step to max iteration to stop iterative (RK methods)
          step = num_steps;

          // Store rebound vector replacing current particle velocity#
          //    Currently inside "apply_closed_bc"
          // Point up(gdim, u_p.data());
          //_P->set_property(cidx, pidx, 1, Point(gdim, dummy_vel.data()) );

          ///// Testing using the internal movement commands

          // // Then we cross facet which has a neighboring cell
          // _P->push_particle(dt_int, up, cidx, pidx);

          // // Update index of receiving cell
          // cidx_recv = (fcells[0] == cidx_recv) ? fcells[1] : fcells[0];


          // // Then terminate
          // dt_rem *= 0.;
          // // Copy current position to old position
          // if (step == (num_steps - 1))
          //   _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          // if (cidx_recv != cidx)
          //   reloc.push_back({cidx, pidx, cidx_recv});

          // // Update remaining time
          // dt_rem -= dt_int;
          // if (dt_rem < 1E-15)
          // {
          //   // Then terminate
          //   dt_rem *= 0.;
          //   // Copy current position to old position
          //   if (step == (num_steps - 1))
          //     _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          //   if (cidx_recv != cidx)
          //     reloc.push_back({cidx, pidx, cidx_recv});
          // }
          
          hit_cbc = true;
        }
        else if (ftype == facet_t::periodic)
        {
          // TODO: add support for periodic bcs
          apply_periodic_bc(dt_rem, up, cidx, pidx, target_facet);

          // Copy current position to old position
          if (step == (num_steps - 1))
            _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          // Behavior in parallel
          // Always do a global push
          if (mpi_size > 1)
          {
            reloc.push_back(
                {cidx, pidx, std::numeric_limits<unsigned int>::max()});
          }
          else
          {
            // Behavior in serial
            // TODO: call particle locate
            std::size_t cell_id
                = mesh->bounding_box_tree()->compute_first_entity_collision(
                    _P->x(cidx, pidx));

            reloc.push_back({cidx, pidx, cell_id});
          }

          dt_rem = 0.0;
        }
        else if (ftype == facet_t::bounded)
        {
          // Then bounded bc
          apply_bounded_domain_bc(dt_rem, up, cidx, pidx, target_facet);

          // Copy current position to old position
          if (step == (num_steps - 1))
            _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          if (mpi_size > 1) // Behavior in parallel
            reloc.push_back(
                {cidx, pidx, std::numeric_limits<unsigned int>::max()});
          else
          {
            // Behavior in serial
            std::size_t cell_id = _P->mesh()
                                      ->bounding_box_tree()
                                      ->compute_first_entity_collision(
                                          _P->x(cidx, pidx));
            reloc.push_back({cidx, pidx, cell_id});
          }
          dt_rem = 0.0;
        }
        else
        {
          dolfin_error("advect_particles.cpp::do_step",
                       "encountered unknown boundary",
                       "Only internal boundaries implemented yet");
        }
      }
      else
      {
        dolfin_error("advect_particles.cpp::do_step",
                     "found incorrect number of facets (<1 or > 2)", "Unknown");
      }
    }
  } // end_while
}
//-----------------------------------------------------------------------------
void advect_particles::do_substepLPT(
    double dt, Point& up, const std::size_t cidx, std::size_t pidx,
    std::size_t& step, const std::size_t num_steps,
    const std::size_t xp0_idx, const std::size_t up0_idx,
    std::vector<std::array<std::size_t, 3>>& reloc, Point& ForceBalance,
    Point& Acceleration, Point& up_1)

    // void advect_particles::do_substep(
    // double dt, Point& up, const std::size_t cidx, std::size_t pidx,
    // std::size_t& step, const std::size_t num_steps,
    // const std::size_t xp0_idx, const std::size_t up0_idx,
    // std::vector<std::array<std::size_t, 3>>& reloc)
{
  double dt_rem = dt;

  const Mesh* mesh = _P->mesh();
  const std::size_t mpi_size = MPI::size(mesh->mpi_comm());
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology().dim();

  std::size_t cidx_recv = std::numeric_limits<unsigned int>::max();

  if (step == 0)
    cidx_recv = cidx;
  else
  {
    // The reason for doing this step is:
    // for the multistep (RK) schemes, the carried old position may not be the
    // same as the cell where the particle lives newest position is always
    // carried
    // TODO: Can we think of smarter implementation?
    cidx_recv = mesh->bounding_box_tree()->compute_first_entity_collision(
        _P->x(cidx, pidx));

    // One alternative might be:
    // Cell cell(*(_P->_mesh), cidx);
    // bool contain = cell.contains(_P->_cell2part[cidx][pidx][0])
    // If true  cidx_recv = cidx; and continue
    // if not: do entity collision

    // FIXME: this approach is robust for the internal points multistep schemes,
    // but what about multistage schemes and near closed/periodic bc's?
    if (cidx_recv == std::numeric_limits<unsigned int>::max())
    {
      _P->push_particleLPT(dt_rem, up, cidx, pidx, ForceBalance,
        Acceleration, up_1);
      if (pbc_active)
        pbc_limits_violation(cidx, pidx);

      if (bounded_domain_active)
        bounded_domain_violation(cidx, pidx);

      if (step == (num_steps - 1))
      {
        // Copy current position to old position
        // so something like
        _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));
      }
      // Apparently, this always lead to a communicate, but why?
      reloc.push_back({cidx, pidx, std::numeric_limits<unsigned int>::max()});
      return; // Stop right here
    }
  }

  // std::cout << "xp0_idx: " << xp0_idx << std::endl;

  bool hit_cbc = false; // Hit closed boundary condition (?!)
  while (dt_rem > 1E-15)
  {
    // Returns facet which is intersected and the time it takes to do so
    std::tuple<std::size_t, double> intersect_info
        = time2intersect(cidx_recv, dt_rem, _P->x(cidx, pidx), up);
    // std::cout << "time2intersect: " << std::get<0>(intersect_info) << std::endl;
    // std::cout << "time2intersect: " << std::get<1>(intersect_info) << std::endl;
    const std::size_t target_facet = std::get<0>(intersect_info);
    const double dt_int = std::get<1>(intersect_info);

    if (target_facet == std::numeric_limits<unsigned int>::max())
    {
      // Then remain within cell, finish time step
      _P->push_particleLPT(dt_rem, up, cidx, pidx, ForceBalance, Acceleration, up_1);
      dt_rem = 0.0;

      if (bounded_domain_active)
        bounded_domain_violation(cidx, pidx);

      if (step == (num_steps - 1))
        // Copy current position to old position
        _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

      // If cidx_recv != ci->index(), particles crossed facet and hence need to
      // be relocated
      if (cidx_recv != cidx)
        reloc.push_back({cidx, pidx, cidx_recv});
    }
    else
    {
      Facet f(*mesh, target_facet);
      const unsigned int* fcells = f.entities(tdim);

      // Two options: if internal (==2) else if boundary
      if (f.num_entities(tdim) == 2)
      {
        // Then we cross facet which has a neighboring cell
        _P->push_particleLPT(dt_rem, up, cidx, pidx, ForceBalance,
          Acceleration, up_1);

        // Update index of receiving cell
        cidx_recv = (fcells[0] == cidx_recv) ? fcells[1] : fcells[0];

        // Update remaining time
        dt_rem -= dt_int;
        if (dt_rem < 1E-15)
        {
          // Then terminate
          dt_rem *= 0.;
          // Copy current position to old position
          if (step == (num_steps - 1))
            _P->push_particleLPT(dt_rem, up, cidx, pidx, ForceBalance,
              Acceleration, up_1);

          if (cidx_recv != cidx)
            reloc.push_back({cidx, pidx, cidx_recv});
        }
      }
      else if (f.num_entities(tdim) == 1)
      {
        const facet_t ftype = facets_info[target_facet].type;
        // Then we hit a boundary, but which type?
        if (f.num_global_entities(tdim) == 2)
        { // Internal boundary between processes
          assert(ftype == facet_t::internal);
          _P->push_particleLPT(dt_rem, up, cidx, pidx, ForceBalance,
            Acceleration, up_1);
          dt_rem = 0.0;

          // Updates particle position if pbc_limits is violated
          if (pbc_active)
            pbc_limits_violation(cidx, pidx);

          if (bounded_domain_active)
            bounded_domain_violation(cidx, pidx);

          // Copy current position to old position
          if (step == (num_steps - 1) || hit_cbc)
            _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          reloc.push_back(
              {cidx, pidx, std::numeric_limits<unsigned int>::max()});

          return; // Stop right here
        }
        else if (ftype == facet_t::open)
        {
          // Particle leaves the domain. Relocate to another process (particle
          // will be discarded)

          // Issue 12 work around: do full push to push particle outside
          // domain
          _P->push_particleLPT(dt_rem, up, cidx, pidx, ForceBalance,
            Acceleration, up_1);

          // Then push back to relocate
          reloc.push_back(
              {cidx, pidx, std::numeric_limits<unsigned int>::max()});
          dt_rem = 0.0;
        }
        else if (ftype == facet_t::closed)
        {
          apply_closed_bc(dt_int, up, cidx, pidx, target_facet);
          dt_rem -= dt_int;

          // // TODO: CHECK THIS
          // dt_rem
          //     += (1. - dti[step]) * (dt / dti[step]); // Make timestep complete
          // // If we hit a closed bc, modify following, probably is first order:
          // // dt_rem = 0.0;

          // TODO: UPDATE AS PARTICLE!
          std::vector<double> dummy_vel(gdim,
                                        std::numeric_limits<double>::max());
          _P->set_property(cidx, pidx, up0_idx, Point(gdim, dummy_vel.data()));
          // std::cout << "up0_idx: " << xp0_idx << std::endl;

          std::tuple<std::size_t, double> intersect_info
              = time2intersect(cidx_recv, dt_rem, _P->x(cidx, pidx), up);
          // std::cout << "time2intersect: " << std::get<0>(intersect_info) << std::endl;
          // std::cout << "time2intersect: " << std::get<1>(intersect_info) << std::endl;
          // const std::size_t target_facet = std::get<0>(intersect_info);
          const double dt_int1 = std::get<1>(intersect_info);

          // Set tiem remaining to the time to hit next cell using up(reflected)
          dt_rem = dt_int1;
          // Push particle to the next cell
          _P->push_particleLPT(dt_rem, up, cidx, pidx, ForceBalance,
            Acceleration, up_1);
          // Set step to max iteration to stop iterative (RK methods)
          step = num_steps;

          // Store rebound vector replacing current particle velocity#
          //    Currently inside "apply_closed_bc"
          // Point up(gdim, u_p.data());
          //_P->set_property(cidx, pidx, 1, Point(gdim, dummy_vel.data()) );

          ///// Testing using the internal movement commands

          // // Then we cross facet which has a neighboring cell
          // _P->push_particle(dt_int, up, cidx, pidx);

          // // Update index of receiving cell
          // cidx_recv = (fcells[0] == cidx_recv) ? fcells[1] : fcells[0];


          // // Then terminate
          // dt_rem *= 0.;
          // // Copy current position to old position
          // if (step == (num_steps - 1))
          //   _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          // if (cidx_recv != cidx)
          //   reloc.push_back({cidx, pidx, cidx_recv});

          // // Update remaining time
          // dt_rem -= dt_int;
          // if (dt_rem < 1E-15)
          // {
          //   // Then terminate
          //   dt_rem *= 0.;
          //   // Copy current position to old position
          //   if (step == (num_steps - 1))
          //     _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          //   if (cidx_recv != cidx)
          //     reloc.push_back({cidx, pidx, cidx_recv});
          // }
          
          hit_cbc = true;
        }
        else if (ftype == facet_t::periodic)
        {
          // TODO: add support for periodic bcs
          apply_periodic_bc(dt_rem, up, cidx, pidx, target_facet);

          // Copy current position to old position
          if (step == (num_steps - 1))
            _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          // Behavior in parallel
          // Always do a global push
          if (mpi_size > 1)
          {
            reloc.push_back(
                {cidx, pidx, std::numeric_limits<unsigned int>::max()});
          }
          else
          {
            // Behavior in serial
            // TODO: call particle locate
            std::size_t cell_id
                = mesh->bounding_box_tree()->compute_first_entity_collision(
                    _P->x(cidx, pidx));

            reloc.push_back({cidx, pidx, cell_id});
          }

          dt_rem = 0.0;
        }
        else if (ftype == facet_t::bounded)
        {
          // Then bounded bc
          apply_bounded_domain_bc(dt_rem, up, cidx, pidx, target_facet);

          // Copy current position to old position
          if (step == (num_steps - 1))
            _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));

          if (mpi_size > 1) // Behavior in parallel
            reloc.push_back(
                {cidx, pidx, std::numeric_limits<unsigned int>::max()});
          else
          {
            // Behavior in serial
            std::size_t cell_id = _P->mesh()
                                      ->bounding_box_tree()
                                      ->compute_first_entity_collision(
                                          _P->x(cidx, pidx));
            reloc.push_back({cidx, pidx, cell_id});
          }
          dt_rem = 0.0;
        }
        else
        {
          dolfin_error("advect_particles.cpp::do_step",
                       "encountered unknown boundary",
                       "Only internal boundaries implemented yet");
        }
      }
      else
      {
        dolfin_error("advect_particles.cpp::do_step",
                     "found incorrect number of facets (<1 or > 2)", "Unknown");
      }
    }
  } // end_while
}

//-----------------------------------------------------------------------------
advect_particles::~advect_particles() {}
//
//-----------------------------------------------------------------------------
//
//      RUNGE KUTTA 2
//
//-----------------------------------------------------------------------------
void advect_rk2::do_step(double dt)
{
  if (dt <= 0.)
    dolfin_error("advect_particles.cpp::step", "set timestep.",
                 "Timestep should be > 0.");

  init_weights();

  const Mesh* mesh = _P->mesh();
  std::size_t gdim = mesh->geometry().dim();

  std::vector<std::vector<double>> coeffs_storage(mesh->num_cells());
  std::size_t num_substeps = 2;

  for (std::size_t step = 0; step < num_substeps; step++)
  {
    const Function& uh_step = uh(step, dt);

    // Needed for local reloc
    std::vector<std::array<std::size_t, 3>> reloc;

    for (CellIterator ci(*mesh); !ci.end(); ++ci)
    {
      for (std::size_t i = 0; i < _P->num_cell_particles(ci->index()); i++)
      {
        Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
        Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                   _element);

        // Compute value at point using expansion coeffs and basis matrix
        std::vector<double> coeffs;
        Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);
        Eigen::Map<Eigen::VectorXd> exp_coeffs(
            coeffs.data(), _space_dimension);
        Eigen::VectorXd u_p = basis_mat * exp_coeffs;

        Point up(gdim, u_p.data());
        if (step == 0)
          _P->set_property(ci->index(), i, up0_idx, up);
        else
        {
          // Goto next particle, this particle hitted closed bound
          if (_P->property(ci->index(), i, up0_idx)[0]
              == std::numeric_limits<double>::max())
            continue;
          up += _P->property(ci->index(), i, up0_idx);
          up *= 0.5;
        }

        // Reset position to old
        if (step == 1)
          _P->set_property(ci->index(), i, 0,
                           _P->property(ci->index(), i, xp0_idx));

        // Do substep
        do_substep(dt, up, ci->index(), i, step, num_substeps, xp0_idx, up0_idx,
                   reloc);
      }
    }

    // Relocate local and global
    _P->relocate(reloc);
  }
}
//-----------------------------------------------------------------------------
void advect_rk2::do_step(double dt,
  Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> LPTParameters)
{
  if (dt <= 0.)
    dolfin_error("advect_particles.cpp::step", "set timestep.",
                 "Timestep should be > 0.");

  init_weights();

  const Mesh* mesh = _P->mesh();
  std::size_t gdim = mesh->geometry().dim();

  std::vector<std::vector<double>> coeffs_storage(mesh->num_cells());
  std::size_t num_substeps = 2;

  for (std::size_t step = 0; step < num_substeps; step++)
  {
    const Function& uh_step = uh(step, dt);

    // Needed for local reloc
    std::vector<std::array<std::size_t, 3>> reloc;

    for (CellIterator ci(*mesh); !ci.end(); ++ci)
    {
      for (std::size_t i = 0; i < _P->num_cell_particles(ci->index()); i++)
      {
        Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
        Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                   _element);

        // Compute value at point using expansion coeffs and basis matrix
        std::vector<double> coeffs;
        Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);
        Eigen::Map<Eigen::VectorXd> exp_coeffs(
            coeffs.data(), _space_dimension);
        Eigen::VectorXd u_p = basis_mat * exp_coeffs;

        Point up(gdim, u_p.data());

        std::cout << "P Pos: " <<  _P->x(ci->index(), i) << std::endl;
        std::cout << "P Vel: " <<  up << std::endl;

        // y = -x
        Point pPos = _P->x(ci->index(), i);
        for (std::size_t iI = 0; (iI < gdim); iI++)
        {
          if (iI == 0) // based on y = -x
          {
            pPos[iI] = pPos[iI] + -1 * (up[iI] * dt);
          } else if (iI == 2)
          {
            // ((pPos[2] - 0.0110585) / 0.00016) * 100;
            // Set Z to midpoint of channel
            pPos[iI] = (0.00016 / 2) + 0.0110585;
          } else
          {
            pPos[iI] = pPos[iI] + (up[iI] * dt);
          }
        }
        std::cout << "P Pos: " <<  pPos << std::endl;

        Utils::return_basis_matrix(basis_mat.data(), pPos, *ci,
                                    _element);

        // Compute value at point using expansion coeffs and basis matrix, first
        // convert to Eigen matrix
        // std::vector<double> coeffs;
        // Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);
        // Eigen::Map<Eigen::VectorXd> exp_coeffs(
        //     coeffs.data(), _space_dimension);
        Eigen::VectorXd u_p1 = basis_mat * exp_coeffs;

        Point up1(gdim, u_p1.data());
        std::cout << "P Vel1: " <<  up1 << std::endl;
        // Create point to store up1 midpoint velocities
        Point up2 = up1;

        // Previous velocity point
        Point up_1 = (_P->property(ci->index(), i, 1));
        std::cout << "P Velocity B4: " << up_1 << std::endl;
        // Point up_0 = (_P->property(ci->index(), i, 0));
        Point pPos1 = _P->x(ci->index(), i);

        for (std::size_t iI = 0; (iI < gdim); iI++)
        {
          // Take new position away from old
          pPos[iI] = pPos[iI] - pPos1[iI];
          // on axis direction (+/-), is velocity inc / dec
          up2[iI] = up1[iI] - up[iI];

          // If velocity is moving in the positive direction
          //  relative to pPos+Vel - pPos on plane
          if (up2[iI] > 0) 
          {
            // up1[iI] = 1;
            up1[iI] = abs(up1[iI]);
          } else
          {
            // up1[iI] = -1;
            up1[iI] = abs(up1[iI]) * -1;
          }
          if (pPos[iI] < 0) // Invert flow where positive xy is +ve/-ve
          {
            up1[iI] *= -1;
          }
        }
        
        // up = do_stepLPT(dt, up, up_1, up1, LPTParameters);
        // dt = DEFINE_DPM_TIMESTEP(dt, up, up_1, LPTParameters, pPos, dPoint);
        
        // set dT to 1 second as up is 1 sec
        // up = do_stepLPT(dt, up, up_1, up1, pPos1, LPTParameters);

        // Calculate relax parameter
        double relax = this->cal_relax(
          LPTParameters[3], LPTParameters[0], LPTParameters[1]);

        Point Acceleration = up;
        Point ForceBalance = up;
        // Point pPos = _P->x(ci->index(), i);

        // Particle positive or negative along axis (ind. as vector)
        Point P4 = (_P->property(ci->index(), i, 4));
        // Particle distance from boundary, Boundary distance, 0
        Point dPoint = (_P->property(ci->index(), i, 3));

        // double Mag = sqrt((P4.x() * P4.x())
        //                 + (P4.y() * P4.y()));

        Point zParam = Acceleration;
        zParam[0] = LPTParameters[4]; // channel height
        zParam[1] = LPTParameters[5]; // minimum z

        // Point pPos1 = pPos;
        // pPos1[0] += P4.x() * (((dPoint[1]/2)-dPoint[0]) / Mag); // x axis
        // pPos1[1] += P4.y() * (((dPoint[1]/2)-dPoint[0]) / Mag); // y axis
        // pPos1[2] = zParam[1] + (zParam[0] / 2);

        // // Calculate uMax for lfit force
        // Utils::return_basis_matrix(basis_mat.data(), pPos1, *ci,
        //                             _element);
        // Eigen::VectorXd u_p1 = basis_mat * exp_coeffs;

        // double uMax = sqrt((u_p1.x() * u_p1.x())
        //                 + (u_p1.y() * u_p1.y())
        //                 + (u_p1.z() * u_p1.z()));

        // std::cout << "pPos1: " << pPos1 << std::endl;
        // std::cout << "u_p1: " << u_p1 << std::endl;
        // std::cout << "uMax: " << uMax << std::endl;
        std::cout << "dPoint: " << dPoint << std::endl;

        Acceleration *= 0;
        ForceBalance *= 0;
        for (std::size_t iI = 0; (iI < gdim); iI++)
        {
          double uMax = LPTParameters[6];
          if (dPoint[1] < 0.00025)
          {
            uMax *= 0.1;
          }
          
          // double uMax = 
          
          // if (LPTParameters[6] == 0)
          // {
            // pPos = cal_ParticleDistToBoundary(bmeshXY, Zmid)
            // Accelerations due to all other forces except drag force
            // Acceleration[iI] = this->cal_WallLiftSq(LPTParameters[3],
            //   LPTParameters[0], LPTParameters[2], reynolds, iI, up, up1,
            //   pPos, gdim, LPTParameters[4], LPTParameters[5], dt);

          // Calculate Wall Lift force
          // Acceleration[iI] = this->cal_WallLiftSq(LPTParameters[3],
          //   LPTParameters[0], LPTParameters[1], LPTParameters[2],
          //   zParam, uMax, iI, gdim, up, pPos, dPoint, P4);
          Acceleration[iI] = 0;

          // Cal Relative Reynolds (particle to flow)
          double reynolds = this->cal_reynolds(LPTParameters[3],
            LPTParameters[0], LPTParameters[2], up[iI], up_1[iI]);

          // Calculate Drag coefficient
          double drag = this->cal_drag(reynolds);

          // Velocity Response Time (ForceBalance) for LPT
          ForceBalance[iI] = ((drag * reynolds) / 24);
          ForceBalance[iI] *= (relax);
          ForceBalance[iI] = 1 / ForceBalance[iI];

          // Add Wall Correction
          // WallCor[iI] = this->cal_WallCorrection(particleDiameter,
          //   distance);

          // }
        }

        // if (LPTParameters[6] == 0)
        // {
        //   // Add s values from Wall Induced Lift Force
        //   dt = DEFINE_DPM_TIMESTEP(dt, up, up_1, LPTParameters, pPos, dPoint);
        // }

        std::cout << "P Acceleration: " << Acceleration << std::endl;
        std::cout << "P ForceBalance: " << ForceBalance << std::endl;
        // Store previous particle velocity in slot 2 
        //    Important to store here once rebound applied
        // std::cout << "up: " << (up) << std::endl;
        _P->set_property(ci->index(), i, 1, up);
        
        // Set current position to old slot before movement
        // _P->set_property(ci->index(), i, 1, (_P->property(ci->index(), i, 0)));
        std::cout << "up_1 Store: " << (_P->property(ci->index(), i, 1)) << std::endl;
    
        if (step == 0)
          _P->set_property(ci->index(), i, up0_idx, up);
        else
        {
          // Goto next particle, this particle hitted closed bound
          if (_P->property(ci->index(), i, up0_idx)[0]
              == std::numeric_limits<double>::max())
            continue;
          up += _P->property(ci->index(), i, up0_idx);
          up *= 0.5;
        }

        // Reset position to old
        if (step == 1)
          _P->set_property(ci->index(), i, 0,
                           _P->property(ci->index(), i, xp0_idx));

        // Do substep
        // do_substep(dt, up, ci->index(), i, step, num_substeps, xp0_idx, up0_idx,
        //            reloc);
        do_substepLPT(dt * dti[step], up, ci->index(), i, step, num_substeps,
                   xp0_idx, up0_idx, reloc, ForceBalance, Acceleration, up_1);
      }
    }

    // Relocate local and global
    _P->relocate(reloc);
  }
}
//-----------------------------------------------------------------------------
//
//      RUNGE KUTTA 3
//
//-----------------------------------------------------------------------------
void advect_rk3::do_step(double dt)
{
  if (dt < 0.)
    dolfin_error("advect_particles.cpp::step", "set timestep.",
                 "Timestep should be > 0.");

  init_weights();

  const Mesh* mesh = _P->mesh();
  const std::size_t gdim = mesh->geometry().dim();
  std::vector<std::vector<double>> coeffs_storage(mesh->num_cells());
  std::size_t num_substeps = 3;

  for (std::size_t step = 0; step < num_substeps; step++)
  {
    // Needed for local reloc
    std::vector<std::array<std::size_t, 3>> reloc;

    const Function& uh_step = uh(step, dt);

    for (CellIterator ci(*mesh); !ci.end(); ++ci)
    {
      // Loop over particles
      for (std::size_t i = 0; i < _P->num_cell_particles(ci->index()); i++)
      {
        Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
        Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                   _element);

        // Compute value at point using expansion coeffs and basis matrix, first
        // convert to Eigen matrix
        std::vector<double> coeffs;
        Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);
        Eigen::Map<Eigen::VectorXd> exp_coeffs(
            coeffs.data(), _space_dimension);
        Eigen::VectorXd u_p = basis_mat * exp_coeffs;

        Point up(gdim, u_p.data());

        // Then reset position to the old position
        _P->set_property(ci->index(), i, 0,
                         _P->property(ci->index(), i, xp0_idx));

        if (step == 0)
          _P->set_property(ci->index(), i, up0_idx, up * (weights[step]));
        else if (step == 1)
        {
          Point p = _P->property(ci->index(), i, up0_idx);
          if (p[0] == std::numeric_limits<double>::max())
            continue;
          _P->set_property(ci->index(), i, up0_idx, p + up * (weights[step]));
        }
        else if (step == 2)
        {
          Point p = _P->property(ci->index(), i, up0_idx);
          if (p[0] == std::numeric_limits<double>::max())
            continue;
          up *= weights[step];
          up += _P->property(ci->index(), i, up0_idx);
        }

        // Do substep
        do_substep(dt * dti[step], up, ci->index(), i, step, num_substeps,
                   xp0_idx, up0_idx, reloc);
      }
    }

    // Relocate local and global
    _P->relocate(reloc);
  }
}
//-----------------------------------------------------
// void advect_rk3::do_step(double dt,
//   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> LPTParameters)
// {
//   if (dt < 0.)
//     dolfin_error("advect_particles.cpp::step", "set timestep.",
//                  "Timestep should be > 0.");

//   init_weights();

//   const Mesh* mesh = _P->mesh();
//   const std::size_t gdim = mesh->geometry().dim();
//   std::vector<std::vector<double>> coeffs_storage(mesh->num_cells());
//   std::size_t num_substeps = 3;

//   for (std::size_t step = 0; step < num_substeps; step++)
//   {
//     // Needed for local reloc
//     std::vector<std::array<std::size_t, 3>> reloc;

//     const Function& uh_step = uh(step, dt);

//     for (CellIterator ci(*mesh); !ci.end(); ++ci)
//     {
//       // Loop over particles
//       for (std::size_t i = 0; i < _P->num_cell_particles(ci->index()); i++)
//       {
//         Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
//         Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
//                                    _element);

//         // Compute value at point using expansion coeffs and basis matrix, first
//         // convert to Eigen matrix
//         std::vector<double> coeffs;
//         Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);
//         Eigen::Map<Eigen::VectorXd> exp_coeffs(
//             coeffs.data(), _space_dimension);
//         Eigen::VectorXd u_p = basis_mat * exp_coeffs;
 
//         Point up(gdim, u_p.data());

//         std::cout << "P Pos: " <<  _P->x(ci->index(), i) << std::endl;
//         std::cout << "P Vel: " <<  up << std::endl;

//         // y = -x
//         Point pPos1 = _P->x(ci->index(), i);
//         for (std::size_t iI = 0; (iI < gdim); iI++)
//         {
//           if (iI == 0) // based on y = -x
//           {
//             pPos1[iI] = pPos1[iI] + (-1 * (up[iI] * dt));
//           } else if (iI == 2)
//           {
//             // ((pPos[2] - 0.0110585) / 0.00016) * 100;
//             // Set Z to midpoint of channel
//             pPos1[iI] = (0.00016 / 2) + 0.0110585;
//           } else
//           {
//             pPos1[iI] = pPos1[iI] + (up[iI] * dt);
//           }
//         }
//         std::cout << "P Pos1: " <<  pPos1 << std::endl;

//         Utils::return_basis_matrix(basis_mat.data(), pPos1, *ci,
//                                     _element);

//         // Compute value at point using expansion coeffs and basis matrix, first
//         // convert to Eigen matrix
//         // std::vector<double> coeffs;
//         // Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);
//         // Eigen::Map<Eigen::VectorXd> exp_coeffs(
//         //     coeffs.data(), _space_dimension);
//         Eigen::VectorXd u_p1 = basis_mat * exp_coeffs;

//         Point up1(gdim, u_p1.data());
//         std::cout << "P Vel1: " <<  up1 << std::endl;
//         // Create point to store up1 midpoint velocities
//         Point up2 = up1;

//         // Previous velocity point
//         Point up_1 = (_P->property(ci->index(), i, 1));
//         std::cout << "P Velocity B4: " << up_1 << std::endl;
//         // Point up_0 = (_P->property(ci->index(), i, 0));
//         Point pPos = _P->x(ci->index(), i);

//         for (std::size_t iI = 0; (iI < gdim); iI++)
//         {
//           // Take new position away from old
//           pPos[iI] = pPos1[iI] - pPos[iI];
//           // on axis direction (+/-), is velocity inc / dec
//           up2[iI] = abs(up1[iI]) - abs(up[iI]);

//           std::cout << "up2: " << up2[iI] << std::endl;
//           // If velocity is moving in the positive direction
//           //  relative to pPos+Vel - pPos on plane
//           if (up2[iI] > 0) 
//           {
//             // up1[iI] = 1;
//             up1[iI] = abs(up1[iI]);
//           } else
//           {
//             // up1[iI] = -1;
//             up1[iI] = abs(up1[iI]) * -1;
//           }
//           if (pPos[iI] < 0) // Invert flow where positive xy is +ve/-ve
//           {
//             std::cout << "Invert Flow" << std::endl;
//             up1[iI] *= -1;
//           }
//         }

//         std::cout << "up1: " << up1 << std::endl;
//         // up = do_stepLPT(dt, up, up_1, up1, LPTParameters);
        
//         // set dT to 1 second as up is 1 sec
//         // up = do_stepLPT(dt, up, up_1, up1, pPos1, LPTParameters);

//         double drag = this->cal_drag( 
//           LPTParameters[3], LPTParameters[0], LPTParameters[2], up, up_1);
//         double relax = this->cal_relax(
//           LPTParameters[3], LPTParameters[0], LPTParameters[1]);

//         // Cal Particle Reynolds
//         double reynolds = cal_reynolds(LPTParameters[3],
//           LPTParameters[0], LPTParameters[2], up, up_1);

//         // Drag force balance for LPT
//         double ForceBalance = ((drag * reynolds) / 24);
//         ForceBalance *= (relax);
//         ForceBalance = 1 / ForceBalance;

//         Point Acceleration = up;
//         pPos = _P->x(ci->index(), i);

//         Acceleration *= 0;
//         for (std::size_t iI = 0; (iI < gdim); iI++)
//         {
//           if (LPTParameters[6] == 0)
//           {
//             // Accelerations due to all other forces except drag force
//             Acceleration[iI] = this->cal_WallLiftSq(LPTParameters[3],
//               LPTParameters[0], LPTParameters[2], reynolds, iI, up, up1,
//               pPos, gdim, LPTParameters[4], LPTParameters[5], dt);
//           }
//         }
//         // if (LPTParameters[6] == 0)
//         // {
//         //   dt = DEFINE_DPM_TIMESTEP(dt, up, up_1, LPTParameters);
//         // }

//         std::cout << "P Acceleration: " << Acceleration << std::endl;
//         std::cout << "P ForceBalance: " << ForceBalance << std::endl;
    
//         // Then reset position to the old position
//         _P->set_property(ci->index(), i, 0,
//                          _P->property(ci->index(), i, xp0_idx));

//         if (step == 0)
//           _P->set_property(ci->index(), i, up0_idx, up * (weights[step]));
//         else if (step == 1)
//         {
//           Point p = _P->property(ci->index(), i, up0_idx);
//           if (p[0] == std::numeric_limits<double>::max())
//             continue;
//           _P->set_property(ci->index(), i, up0_idx, p + up * (weights[step]));
//         }
//         else if (step == 2)
//         {
//           Point p = _P->property(ci->index(), i, up0_idx);
//           if (p[0] == std::numeric_limits<double>::max())
//             continue;
//           up *= weights[step];
//           up += _P->property(ci->index(), i, up0_idx);
//         }
        
//         std::cout << "up: " << (up) << std::endl;
//         // _P->set_property(ci->index(), i, 1, up);
//         // Do substep
//         // do_substep(dt * dti[step], up, ci->index(), i, step, num_substeps,
//         //            xp0_idx, up0_idx, reloc);
                   
//         do_substepLPT(dt * dti[step], up, ci->index(), i, step, num_substeps,
//                    xp0_idx, up0_idx, reloc, ForceBalance, Acceleration, up_1);
//        //if rebound == true
//             // End this iteration and rebound
//         // dt_rem
//         std::cout << "steps: " << step << std::endl;
                   
//         // Store previous particle velocity in slot 2 
//         //    Important to store here once rebound applied
//         //      -Rebound applied in substep-
//         // std::cout << "up2 AFsubstep: " << (up) << std::endl;
        

//         // _P->set_property(ci->index(), i, 1, up);
//         // std::cout << "P Velocity Final: " << up << std::endl;
                   
//       } // End of particle loop
    
//     } // cycle mesh loop
    
//     // Relocate local and global
//     _P->relocate(reloc);
    
//   } // end of Runge-Kutta loop
// }
//-----------------------------------------------------
void advect_rk3::do_step(double dt,
  Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> LPTParameters)
{
  if (dt < 0.)
    dolfin_error("advect_particles.cpp::step", "set timestep.",
                 "Timestep should be > 0.");

  init_weights();

  const Mesh* mesh = _P->mesh();
  const std::size_t gdim = mesh->geometry().dim();
  std::vector<std::vector<double>> coeffs_storage(mesh->num_cells());
  std::size_t num_substeps = 3;

  for (std::size_t step = 0; step < num_substeps; step++)
  {
    // Needed for local reloc
    std::vector<std::array<std::size_t, 3>> reloc;

    const Function& uh_step = uh(step, dt);

    for (CellIterator ci(*mesh); !ci.end(); ++ci)
    {
      // Loop over particles
      for (std::size_t i = 0; i < _P->num_cell_particles(ci->index()); i++)
      {
        Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
        Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                   _element);

        // Compute value at point using expansion coeffs and basis matrix, first
        // convert to Eigen matrix
        std::vector<double> coeffs;
        Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);
        Eigen::Map<Eigen::VectorXd> exp_coeffs(
            coeffs.data(), _space_dimension);
        Eigen::VectorXd u_p = basis_mat * exp_coeffs;
 
        Point up(gdim, u_p.data());

        std::cout << "P Pos: " <<  _P->x(ci->index(), i) << std::endl;
        std::cout << "Flow Vel: " <<  up << std::endl;

        // Current Particle velocity
        Point up_1 = (_P->property(ci->index(), i, 1));
        std::cout << "P Vel Curr: " << up_1 << std::endl;
        double particleDiameter = LPTParameters[0];

        // Calculate relax parameter
        double relax = this->cal_relax(
          LPTParameters[3], particleDiameter, LPTParameters[1]);

        Point Acceleration = up;
        Point ForceBalance = up;
        Point WallCor = up;
        Point pPos = _P->x(ci->index(), i);

        // Particle positive or negative along axis (ind. as vector)
        Point P4 = (_P->property(ci->index(), i, 4));
        // Particle distance from boundary, Boundary distance, 0
        Point dPoint = (_P->property(ci->index(), i, 3));

        // double Mag = sqrt((P4.x() * P4.x())
        //                 + (P4.y() * P4.y()));

        Point zParam = Acceleration;
        zParam[0] = LPTParameters[4]; // channel height
        zParam[1] = LPTParameters[5]; // minimum z

        Point C_U_G, C_V_G, C_W_G;

        advect_particles::cal_ShearGradient(up, pPos, gdim, basis_mat,
                            *ci, exp_coeffs, _element);
        std::cout << "C_U_G.x() " << C_U_G.x() << std::endl;

        // Caluclate shear gradient C_U_G??

        // double Gxy = ( (sqrt(std::abs( ((u_p1.x() * u_p1.x())
        //         -(u_p.x() * u_p.x()))
        //         +((u_p1.y() * u_p1.y())
        //         -(u_p.y() * u_p.y()) ))))
        //         / particleDiameter);
        // double Gxz = ( (sqrt(std::abs( ((u_p1.x() * u_p1.x())
        //         -(u_p.x() * u_p.x()))
        //         +((u_p1.z() * u_p1.z())
        //         -(u_p.z() * u_p.z()) ))))
        //         / particleDiameter);
        // double Gyz = ( (sqrt(std::abs( ((u_p1.y() * u_p1.y())
        //         -(u_p.y() * u_p.y()))
        //         +((u_p1.z() * u_p1.z())
        //         -(u_p.z() * u_p.z()) ))))
        //         / particleDiameter);

        // double Gxy = ( ((u_p1.x() - u_p.x())
        //         + (u_p1.y() - u_p.y()))
        //         / particleDiameter);
        // double Gxz = ( ((u_p1.x() - u_p.x())
        //         + (u_p1.z() - u_p.z()))
        //         / particleDiameter);
        // double Gyz = ( ((u_p1.y() - u_p.y())
        //         + (u_p1.z() - u_p.z()))
        //         / particleDiameter);

        // double Gxy = ( (u_p.x()) / (u_p.y()) );
        // double Gxz = ( (u_p.x()) / (u_p.z()) );
        // double Gyz = ( (u_p.y()) / (u_p.z()) );

        // std::cout << "Gxy: " << Gxy << std::endl;
        // std::cout << "Gxz: " << Gxz << std::endl;
        // std::cout << "Gyz: " << Gyz << std::endl;
        std::cout << "P4: " << P4 << std::endl;
        // std::cout << "pPos1: " << pPos1 << std::endl;
        // std::cout << "u_p1: " << u_p1 << std::endl;
        // std::cout << "uMax: " << uMax << std::endl;
        std::cout << "dPoint: " << dPoint << std::endl;

        Acceleration *= 0;
        ForceBalance *= 0;
        WallCor *= 0;

        for (std::size_t iI = 0; (iI < gdim); iI++)
        {
          double uMax = LPTParameters[6];
          // if (dPoint[1] < 0.00025)
          // {
          //   uMax *= 0.1;
          // }
          
          // double uMax = 
          
          // if (LPTParameters[6] == 0)
          // {
            // pPos = cal_ParticleDistToBoundary(bmeshXY, Zmid)
            // Accelerations due to all other forces except drag force
            // Acceleration[iI] = this->cal_WallLiftSq(LPTParameters[3],
            //   LPTParameters[0], LPTParameters[2], reynolds, iI, up, up1,
            //   pPos, gdim, LPTParameters[4], LPTParameters[5], dt);

          ////// Calculate Wall Lift force
          //// Ho & Leal 1974 stated parabolic flow parameters
          Acceleration[iI] = this->cal_NetLiftHoLeal1974(LPTParameters[3],
            LPTParameters[0], LPTParameters[1], LPTParameters[2],
            zParam, uMax, iI, gdim, up, pPos, dPoint, P4);
          //// Ho & Leal 1974 using fluid devired shear gradient & rate
          // Acceleration[iI] = this->cal_NetLiftHoLeal1974(LPTParameters[3],
          //   LPTParameters[0], LPTParameters[1], LPTParameters[2],
          //   zParam, uMax, iI, gdim, up, pPos, dPoint, P4,
          //   C_U_G, C_V_G, C_W_G);
          // Acceleration[iI] = this->cal_WallLiftSq(LPTParameters[3],
          //   particleDiameter, LPTParameters[1], LPTParameters[2],
          //   zParam, uMax, iI, gdim, up, pPos, dPoint, P4,
          //   C_U_G, C_V_G, C_W_G);

          // Cal Relative Reynolds (particle to flow)
          double reynolds = this->cal_reynolds(LPTParameters[3],
            particleDiameter, LPTParameters[2], up[iI], up_1[iI]);
          
          std::cout << "Reynolds: " << reynolds << std::endl;

          // Calculate Drag coefficient
          double drag = this->cal_drag(reynolds);

          // Velocity Response Time (ForceBalance) for LPT
          ForceBalance[iI] = ((drag * reynolds) / 24.0);
          ForceBalance[iI] *= (relax);

          //// Add Wall Correction for drag
          // Wall correction is based on 
          // double distance = dPoint[0];
          // WallCor[iI] = ForceBalance[iI];
          
          // std::cout.precision(15);
          // std::cout << "P ForceBalance: " 
          //   << ForceBalance[iI] << std::endl;

          // if (iI == gdim-1)
          // {
          //   // Assumes channel height is constant using zMin
          //   double zMin = zParam[1];
          //   distance = (pPos[2] - zMin);
          //   double h = zParam[0];
          //   // P4[iI] = 1;
          //   if ( distance > (h / 2.0) )
          //   {
          //     distance = (h - distance);
          //     // P4[iI] = -1;
          //   }

          // }
          // WallCor[iI] *= this->cal_WallCorrection(LPTParameters[0],
          //   distance);

          // ForceBalance[iI] = WallCor[iI];

          // Convert drag into Particle Velocity response time
          ForceBalance[iI] = 1 / ForceBalance[iI];
          
          // if (P4[iI] < 0) // Negative along axis
          // {
          //   // std::cout << "Flnl: " << Flnl << std::endl;
          //   // up[ii] += ( lift * (std::abs(up[iP]) / ratioU) );// * 10000;
          //   // up[ii] += std::abs((lift * (std::abs(up[iP]) / ratioU) ) * ForceBal * (exp(-dt/ForceBal) - 1));
          //   WallCor *= -1;
          // }


          //// Virtual Mass
          // double VM = ( 0.5 * (LPTParameters[2] / LPTParameters[1])
          //             * ((up[iI] - up_1[iI]) / (dt * dti[step])) );
          // Acceleration[iI] += VM;
          // //// Non-bouyant particles
          // // Apply only to Z axis
          // if (iI == 2)
          // {
          //   const double G = 9.8; // Gravity
          //   double AddG = (LPTParameters[1] - LPTParameters[2]);
          //   AddG *= G;
          //   AddG /= LPTParameters[1];
          //   Acceleration[iI] -= AddG;
          // }

        }


        // if (LPTParameters[6] == 0)
        // {
        //   // Add s values from Wall Induced Lift Force
        //   dt = DEFINE_DPM_TIMESTEP(dt, up, up_1, LPTParameters, pPos, dPoint);
        // }

        std::cout << "P Acceleration: " << Acceleration << std::endl;
        std::cout << "P ForceBalance (1/x): " << ForceBalance << std::endl;
        std::cout << "P WallForce: " << WallCor << std::endl;

        // Then reset position to the old position
        _P->set_property(ci->index(), i, 0,
                         _P->property(ci->index(), i, xp0_idx));

        if (step == 0)
          _P->set_property(ci->index(), i, up0_idx, up_1 * (weights[step]));
        else if (step == 1)
        {
          Point p = _P->property(ci->index(), i, up0_idx);
          if (p[0] == std::numeric_limits<double>::max())
            continue;
          _P->set_property(ci->index(), i, up0_idx, p + up_1 * (weights[step]));
        }
        else if (step == 2)
        {
          Point p = _P->property(ci->index(), i, up0_idx);
          if (p[0] == std::numeric_limits<double>::max())
            continue;
          up *= weights[step];
          up += _P->property(ci->index(), i, up0_idx);
        }
        
        std::cout << "up: " << (up) << std::endl;
        // _P->set_property(ci->index(), i, 1, up);
        // Do substep
        // do_substep(dt * dti[step], up, ci->index(), i, step, num_substeps,
        //            xp0_idx, up0_idx, reloc);
                   
        do_substepLPT(dt * dti[step], up, ci->index(), i, step, num_substeps,
                   xp0_idx, up0_idx, reloc, ForceBalance, Acceleration, up_1);
       //if rebound == true
            // End this iteration and rebound
        // dt_rem
        std::cout << "steps: " << step << std::endl;

        std::cout << "up_1 Store: " << (_P->property(ci->index(), i, 1)) << std::endl;
        std::cout << "pPos2 Store: " << (_P->property(ci->index(), i, 2)) << std::endl;
        std::cout << "xp0_idx Store: " << xp0_idx << std::endl;
        std::cout << "up0_idx Store: " << up0_idx << std::endl;
                   
        // Store previous particle velocity in slot 2 
        //    Important to store here once rebound applied
        //      -Rebound applied in substep-
        // std::cout << "up2 AFsubstep: " << (up) << std::endl;
        

        // _P->set_property(ci->index(), i, 1, up);
        // std::cout << "P Velocity Final: " << up << std::endl;
                   
      } // End of particle loop
    
    } // cycle mesh loop
    
    // Relocate local and global
    _P->relocate(reloc);
    
  } // end of Runge-Kutta loop
}
//-----------------------------------------------------------------------------
//
//      RUNGE KUTTA 4
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void advect_rk4::do_step(double dt)
{
  if (dt < 0.)
    dolfin_error("advect_particles.cpp::step", "set timestep.",
                 "Timestep should be > 0.");

  init_weights();

  const Mesh* mesh = _P->mesh();
  const std::size_t gdim = mesh->geometry().dim();
//  std::vector<std::vector<double>> coeffs_storage(mesh->num_cells());
  std::size_t num_substeps = 4;

  for (std::size_t step = 0; step < num_substeps; step++)
  {
    // Needed for local reloc
    std::vector<std::array<std::size_t, 3>> reloc;

    const Function& uh_step = uh(step, dt);

    for (CellIterator ci(*mesh); !ci.end(); ++ci)
    {
      // Loop over particles
      for (std::size_t i = 0; i < _P->num_cell_particles(ci->index()); i++)
      {
        Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
        Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                   _element);

        // Compute value at point using expansion coeffs and basis matrix
        std::vector<double> coeffs;
        Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);
        Eigen::Map<Eigen::VectorXd> exp_coeffs(
            coeffs.data(), _space_dimension);
        Eigen::VectorXd u_p = basis_mat * exp_coeffs;

        Point up(gdim, u_p.data());

        // Then reset position to the old position
        _P->set_property(ci->index(), i, 0,
                         _P->property(ci->index(), i, xp0_idx));

        if (step == 0)
          _P->set_property(ci->index(), i, up0_idx, up * (weights[step]));
        else if (step == 1 or step == 2)
        {
          Point p = _P->property(ci->index(), i, up0_idx);
          if (p[0] == std::numeric_limits<double>::max())
            continue;
          _P->set_property(ci->index(), i, up0_idx, p + up * (weights[step]));
        }
        else if (step == 3)
        {
          Point p = _P->property(ci->index(), i, up0_idx);
          if (p[0] == std::numeric_limits<double>::max())
            continue;
          up *= weights[step];
          up += _P->property(ci->index(), i, up0_idx);
        }

        // Do substep
        do_substep(dt * dti[step], up, ci->index(), i, step, num_substeps,
                   xp0_idx, up0_idx, reloc);
      }
    }

    // Relocate local and global
    _P->relocate(reloc);
  }
}
//-----------------------------------------------------------------------
void advect_rk4::do_step(double dt,
  Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> LPTParameters)
{
  if (dt < 0.)
    dolfin_error("advect_particles.cpp::step", "set timestep.",
                 "Timestep should be > 0.");

  init_weights();

  const Mesh* mesh = _P->mesh();
  const std::size_t gdim = mesh->geometry().dim();
//  std::vector<std::vector<double>> coeffs_storage(mesh->num_cells());
  std::size_t num_substeps = 4;

  for (std::size_t step = 0; step < num_substeps; step++)
  {
    // Needed for local reloc
    std::vector<std::array<std::size_t, 3>> reloc;

    const Function& uh_step = uh(step, dt);

    for (CellIterator ci(*mesh); !ci.end(); ++ci)
    {
      // Loop over particles
      for (std::size_t i = 0; i < _P->num_cell_particles(ci->index()); i++)
      {
        Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
        Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
                                   _element);

        // Compute value at point using expansion coeffs and basis matrix
        std::vector<double> coeffs;
        Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);
        Eigen::Map<Eigen::VectorXd> exp_coeffs(
            coeffs.data(), _space_dimension);
        Eigen::VectorXd u_p = basis_mat * exp_coeffs;

        Point up(gdim, u_p.data());

        std::cout << "P Pos: " <<  _P->x(ci->index(), i) << std::endl;
        std::cout << "P Vel: " <<  up << std::endl;

        // y = -x
        Point pPos = _P->x(ci->index(), i);
        for (std::size_t iI = 0; (iI < gdim); iI++)
        {
          if (iI == 0) // based on y = -x
          {
            pPos[iI] = pPos[iI] + -1 * (up[iI] * dt);
          } else if (iI == 2)
          {
            // ((pPos[2] - 0.0110585) / 0.00016) * 100;
            // Set Z to midpoint of channel
            pPos[iI] = (0.00016 / 2) + 0.0110585;
          } else
          {
            pPos[iI] = pPos[iI] + (up[iI] * dt);
          }
        }
        std::cout << "P Pos: " <<  pPos << std::endl;

        Utils::return_basis_matrix(basis_mat.data(), pPos, *ci,
                                    _element);

        // Compute value at point using expansion coeffs and basis matrix, first
        // convert to Eigen matrix
        // std::vector<double> coeffs;
        // Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);
        // Eigen::Map<Eigen::VectorXd> exp_coeffs(
        //     coeffs.data(), _space_dimension);
        Eigen::VectorXd u_p1 = basis_mat * exp_coeffs;

        Point up1(gdim, u_p1.data());
        std::cout << "P Vel1: " <<  up1 << std::endl;
        // Create point to store up1 midpoint velocities
        Point up2 = up1;

        // Previous velocity point
        Point up_1 = (_P->property(ci->index(), i, 1));
        std::cout << "P Velocity B4: " << up_1 << std::endl;
        // Point up_0 = (_P->property(ci->index(), i, 0));
        Point pPos1 = _P->x(ci->index(), i);

        for (std::size_t iI = 0; (iI < gdim); iI++)
        {
          // Take new position away from old
          pPos[iI] = pPos[iI] - pPos1[iI];
          // on axis direction (+/-), is velocity inc / dec
          up2[iI] = up1[iI] - up[iI];

          // If velocity is moving in the positive direction
          //  relative to pPos+Vel - pPos on plane
          if (up2[iI] > 0) 
          {
            // up1[iI] = 1;
            up1[iI] = abs(up1[iI]);
          } else
          {
            // up1[iI] = -1;
            up1[iI] = abs(up1[iI]) * -1;
          }
          if (pPos[iI] < 0) // Invert flow where positive xy is +ve/-ve
          {
            up1[iI] *= -1;
          }
        }
        
        // Particle distance from boundary, Boundary distance, 0
        Point dPoint = (_P->property(ci->index(), i, 3));

        // up = do_stepLPT(dt, up, up_1, up1, LPTParameters);
        dt = DEFINE_DPM_TIMESTEP(dt, up, up_1, LPTParameters, pPos, dPoint);
        // set dT to 1 second as up is 1 sec
        up = do_stepLPT(dt, up, up_1, up1, pPos1, LPTParameters);

        // Store previous particle velocity in slot 2 
        //    Important to store here once rebound applied
        // std::cout << "up: " << (up) << std::endl;
        _P->set_property(ci->index(), i, 1, up);
        
        // Set current position to old slot before movement
        // _P->set_property(ci->index(), i, 1, (_P->property(ci->index(), i, 0)));
        std::cout << "up_1 Store: " << (_P->property(ci->index(), i, 1)) << std::endl;
        std::cout << "pPos2 Store: " << (_P->property(ci->index(), i, 2)) << std::endl;
        std::cout << "xp0_idx Store: " << xp0_idx << std::endl;
        std::cout << "up0_idx Store: " << up0_idx << std::endl;

        // Then reset position to the old position
        _P->set_property(ci->index(), i, 0,
                         _P->property(ci->index(), i, xp0_idx));

        if (step == 0)
          _P->set_property(ci->index(), i, up0_idx, up * (weights[step]));
        else if (step == 1 or step == 2)
        {
          Point p = _P->property(ci->index(), i, up0_idx);
          if (p[0] == std::numeric_limits<double>::max())
            continue;
          _P->set_property(ci->index(), i, up0_idx, p + up * (weights[step]));
        }
        else if (step == 3)
        {
          Point p = _P->property(ci->index(), i, up0_idx);
          if (p[0] == std::numeric_limits<double>::max())
            continue;
          up *= weights[step];
          up += _P->property(ci->index(), i, up0_idx);
        }

        // Do substep
        do_substep(dt * dti[step], up, ci->index(), i, step, num_substeps,
                   xp0_idx, up0_idx, reloc);
      }
    }

    // Relocate local and global
    _P->relocate(reloc);
  }
}
