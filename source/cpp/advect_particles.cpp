// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Vertex.h>
//#include "vtkMath.h"

#include "advect_particles.h"
#include "utils.h"

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

  // std::cout << "LPTParameters[6]: " << LPTParameters[6] << std::endl;

  // double smallestXY;
  // double smallestZ;
  // double distanceXY;
  // double distanceZ;
  // BoundariesLift is several arrays for checking the particle distance from boundary

  // distance = (LPTParameters[6][i][2]-pPos[2])**2
  //   + (LPTParameters[6][i][1]-pPos[1])**2 + (LPTParameters[6][i][0]-pPos[0])**2
  // idy,idx = numpy.where(distance==distance.min())
  // (unsigned int ii = 0; ii < gdim; ii++)
  // for (unsigned int ii = 0; ii < len(BoundariesLift); ii++)
  // {
  //   for (unsigned int i = 0; i < len(BoundariesLift[ii][1]); i++)
  //   {
  //     // XY axis
  //     distanceXY = (BoundariesLift[ii][1][i]-pPos[1])
  //       + (BoundariesLift[ii][0][i]-pPos[0]);
  //     if (distanceXY < smallestXY)
  //     {
  //       smallestXY = distanceXY;
  //     } else if (ii == 0)
  //     {
  //       smallestXY = distanceXY;
  //     }
  //     // Z axis
  //     distanceZ = (BoundariesLift[ii][2][i]-pPos[2]);
  //     if (distanceZ < smallestZ)
  //     {
  //       smallestZ = distanceZ;
  //     }
  //   }
  // }
  


  // double normFlow = up.norm();

  // cidx_recv = mesh->bounding_box_tree()->compute_first_entity_collision(
  //       _P->x(cidx, pidx));

  // double LPTLength = sizeof(LPTParameters) / sizeof(LPTParameters[0]);
  // std::cout << "LPTLength: " << LPTLength << std::endl;
  // double LPTVecT = LPTParameters[6];
  // std::cout << "LPTVecT: " << LPTVecT << std::endl;
  // std::cout << "LPT Not 1: " << (LPTVecT != 1) << std::endl;
  //(sizeof(LPTParameters)/sizeof(*LPTParameters));
  // if (LPTLength > 5){
  //   double PartVelAdj = LPTParameters[6];
  // }

  // Convert velocity to point
  //    gdim = 2 or 3 dimensions
  //    u_p.data() appears to be a complex array for vectors?
  
  // Imported from other functions
  // Point up(gdim, u_p.data());

  // Point up1(2, u_p.data());
  // std::cout << "gdim: " << gdim << std::endl;
  std::cout << "Flow Velocity B4: " << up << std::endl;
  // std::cout << "Flow up: " << up_v << std::endl;
  // std::cout << "Flow Velocity1: " << up1 << std::endl;

  // Lagrangian particle function based on Drag and bouyancy
  //
  // 1) Access prior particle velocity
  // 2) Get current mesh velocity
  // 3) Calculate lagrangian movement



  // Calculate drag, relax and Reynolds for LPT particles
  double drag = this->cal_drag( 
    flowDynamicViscosity, particleDiameter, flowDensity, up, up_1);
  double relax = this->cal_relax(
    flowDynamicViscosity, particleDiameter, particleDensity);
  double reynolds = this->cal_reynolds(
    flowDynamicViscosity, particleDiameter, flowDensity, up, up_1);

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
    int s = 0;
    // std::cout << "Distance: " << s << std::endl;

    // Calculate Wall lift for LPT particles - Dependent on axis
    double lift = this->cal_WallLiftSq(
      flowDynamicViscosity, particleDiameter, flowDensity, reynolds, ii, up,
      up1, pPos, gdim, MicroH, MicroW, s);

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
    up[ii] += (particleVelocity) * exp(-dt/ForceBal);

    // Calculating the reflection, 'r', to find if it is +ve/-ve movement
    // double d = (-1 * up[ii]); // Need to invert the particle direction
    // double r = d - ( 2 * (d * normFlow ) * normFlow ); // normFlow (-1 * up_1[ii])
    // r = r * 2;

    // Assume the particle is neutrally bouyant
    // up[ii] -= ((lift) * ForceBal * (exp(-dt/ForceBal) - 1));

    // Treat as neutrally bouyant
    // std::cout << "Lift Force: " << lift << std::endl;

    // std::cout << "Particle Velocity w/o lift: " << up[ii] << std::endl;
    std::cout << "Added lift: " << (lift * ForceBal * (exp(-dt/ForceBal) - 1)) << std::endl;
    
    // up[ii] -= lift * 100;

    // std::cout << "(exp(-dT/t) - 1): " << std::abs(exp(-dt/ForceBal) - 1) << std::endl;
    // std::cout << "Force Balance: " << ForceBal << std::endl;

    // Lift should be positive / negative defined in WallLiftSq function
    if (DisableLift == 1)
    {
      up[ii] += ((lift * ForceBal) * std::abs(exp(-dt/ForceBal) - 1));
    } else
    {
      std::cout << "Lift Disabled: " << std::endl;
    }
    
    // if (ii == gdim-1) // if Z dimension
    // {
    //   up[ii] += (((AddG + lift) * ForceBal) * std::abs(exp(-dt/ForceBal) - 1));
    // } else // X or Y axis
    // {
    //   up[ii] += ((lift * ForceBal) * std::abs(exp(-dt/ForceBal) - 1));
    // }
    

    // if (ii == gdim-1) // if Z dimension
    // { // Add gravity if the last slot
    //   // up[ii] -= lift * 100;
    //   // up[ii] -= std::abs((lift) * ForceBal * (exp(-dt/ForceBal) - 1));
    //   if (up1[ii] > 0) // R positive
    //   {
    //     // up[ii] += ( lift * (std::abs(up[iP]) / ratioU) );// * 10000;
    //     up[ii] += std::abs((lift) * ForceBal * (exp(-dt/ForceBal) - 1));
    //   } else
    //   {
    //     up[ii] -= std::abs((lift) * ForceBal * (exp(-dt/ForceBal) - 1));
    //   }

    // } else
    // if (ii != gdim-1) // Not Z axis as Z axis calculated in lfit
    // { // Add only lift
    //   // Add lift if above 1% uMax to opposite direction (XY)
    //   if ( (uMax * 0.01) < std::abs(uF) )
    //   {
    //     int iP;
    //     if (ii == 1) // y axis
    //     {
    //       iP = 0; // to change x axis
    //     }
    //     else // x axis
    //     {
    //       iP = 1; // to change y axis
    //     }
    //     // Add ratio of X:Y vectors
    //     double ratioU = 1;
    //     if (gdim == 3){
    //       // Where y / y would be 1, therefore a 1:1 ratio
    //       // ratio = x / y
    //       ratioU += std::abs(up[0]) / std::abs(up[1]);
    //     }
        
    //     std::cout << "up1 Add lift: " << up1 << std::endl;
    //     if (up1[ii] > 0) // R positive
    //     {
    //       // up[ii] += ( lift * (std::abs(up[iP]) / ratioU) );// * 10000;
    //       up[ii] += std::abs((lift * (std::abs(up[iP]) / ratioU) ) * ForceBal * (exp(-dt/ForceBal) - 1));
    //     } else // R negative.
    //     {
    //       // up[ii] -= ( lift * (std::abs(up[iP]) / ratioU) );// * 10000;
    //       up[ii] -= std::abs((lift * (std::abs(up[iP]) / ratioU) ) * ForceBal * (exp(-dt/ForceBal) - 1));
    //     }
    //   }
    // }


    // if (ii == gdim-1)//(ii == gdim-1)
    // { // Add gravity if the last slot
    //   std::cout << "G Force: " << AddG << std::endl;
    //   up[ii] += ((AddG + lift) * ForceBal * (1 - exp(-dt/ForceBal)));
    // } else
    // { // Add only lift
    //   up[ii] += ((lift) * ForceBal * (1 - exp(-dt/ForceBal)));
    // }
    // std::cout << "P Velocity1: " << up[ii] << std::endl;

    // If using Euler implicit discretization - Appearas to break
    // up[ii] = ( up_1[ii] + dt * ( (lift) + (up[ii] / ForceBal)  ) ) / ( 1 + (dt / ForceBal));
  }
  
  // if (LPTVecT != 1){
  //   double upNorm = up.norm();
  //   std::cout << "UP norm: " << upNorm << std::endl;
  //   for (unsigned int ii = 0; ii < gdim; ii++){
  //       // Needs to be the vector magnitude using Pythagorean theorem
  //       //  which is the point.norm() function
  //       // The PartVelAdj is pulled from top level using LPTParameter[6]
  //       //      - Often set to 2-10% channel distance velocity
  //       //          using the parabolic curve algorithm.
  //       //      Example == 10% channel distance is 36% velocity magnitude
  //       //                 5% channel distance is 19%
  //       //                 2.5% channel distance is 9.75%
  //       up[ii] = up[ii] * ( LPTVecT / upNorm);
  //   }
  // }


  // up[2-1] -= AddG;
  // up[gdim-1] -= AddG;
  // std::cout << "Gravity: " << AddG << std::endl;
  // up[gdim-1] -= AddG * Relax * (1 - exp(-dt/ForceBal))


  //up[gdim-1] -= G * ((particleDensity - flowDensity)/ particleDensity);

  // Feed adjusted particle velocity back into the "do_step" algorithm
  //    by applyiong the change to "up" vector

  
  std::cout << "Flow Velocity AF: " << up << std::endl;

  return up;
}

// //-----------------------------------------------------------------------------
// // float prob_prop[10] = {DynVisc, Density, P.Diameter}
// // - Add old particle velocity to particle slot 2
// void advect_particles::do_stepLPT(double dt,
//   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> LPTParameters) 
// {
//   init_weights();

//   // Initial setup
//   const Mesh* mesh = _P->mesh();
//   const MPI_Comm mpi_comm = mesh->mpi_comm();
//   const std::size_t gdim = mesh->geometry().dim();
//   const std::size_t tdim = mesh->topology().dim();

//   // std::cout << "LPTParameters: " << LPTParameters << std::endl;

//   // Set up LPT parametersLPTParameters = falsers[4]; // m Microfluidic Height
//   double MicroW = LPTParameters[5]; // m Microfluidic Width

//   // double particleDiameter = 1e-5; // m - 10 um
//   // double particleDensity = 1050; // Kg/m3 Rough polystyrene density
//   // double flowDensity = 998.2; // Kg/m3 Water density at 21C
//   // double flowDynamicViscosity= 1.003e-3; // Kg/m.s Water kinematic viscosity
//   // double MicroH = 0.00005; // m Microfluidic Height
//   // double MicroW = 0.0001; // m Microfluidic Width

//   // For MPI distributed processes
//   std::size_t num_processes = MPI::size(mpi_comm);

//   // Needed for local reloc
//   std::vector<std::array<std::size_t, 3>> reloc;

//   const Function& uh_step = uh(0, dt);

//   for (CellIterator ci(*mesh); !ci.end(); ++ci)
//   {
//     std::vector<double> coeffs;
//     // Restrict once per cell, once per timestep
//     Utils::return_expansion_coeffs(coeffs, *ci, &uh_step);

//     // Loop over particles in cell
//     for (unsigned int i = 0; i < _P->num_cell_particles(ci->index()); i++)
//     {
//       // FIXME: It might be better to use 'pointer iterator here' as we need to
//       // erase from cell2part vector now we decrement iterator int when needed

//       // basis_mat (
//       Eigen::MatrixXd basis_mat(_value_size_loc, _space_dimension);
//       Utils::return_basis_matrix(basis_mat.data(), _P->x(ci->index(), i), *ci,
//                                  _element);


//       // Create storage for vectors
//       Eigen::Map<Eigen::VectorXd> exp_coeffs(coeffs.data(), _space_dimension);
//       // std::cout << "exp_coeffs: " << exp_coeffs << std::endl;
//       // std::cout << "basis_mat: " << basis_mat << std::endl;
      
//       // Compute value at point using expansion coeffs and basis matrix, first
//       // convert to Eigen matrix
//       //    exp_coeffs = reference velocities from mesh / cell
//       //    basis_mat = particle equation
//       Eigen::VectorXd u_p = basis_mat * (exp_coeffs); 
//       // std::cout << "u_p: " << u_p << std::endl;
      
//       // Replace above with something like...
//       // f = Function(V)
//       // p = Point()
//       // Mesh Value at point = f(p)
      
//       // Point up_p = (_P->property(ci->index(), i, 0));
//       // double up_v = uh_step(up_p);
//        // std::cout << "u_p.data(): " << u_p.data() << std::endl;
      
//       // Convert velocity to point
//       //    gdim = 2 or 3 dimensions
//       //    u_p.data() appears to be a complex array for vectors?
//       Point up(gdim, u_p.data());

//       // Point up1(2, u_p.data());
//       // std::cout << "gdim: " << gdim << std::endl;
//       std::cout << "Flow Velocity B4: " << up << std::endl;
//       // std::cout << "Flow up: " << up_v << std::endl;
//       // std::cout << "Flow Velocity1: " << up1 << std::endl;

//       // Lagrangian particle function based on Drag and bouyancy
//       //
//       // 1) Access prior particle velocity
//       // 2) Get current mesh velocity
//       // 3) Calculate lagrangian movement

//       // Previous velocity point
//       Point up_1 = (_P->property(ci->index(), i, 1));
//       std::cout << "P Velocity: " << up_1 << std::endl;
//       // Point up_0 = (_P->property(ci->index(), i, 0));

//       // Calculate drag, relax and Reynolds for LPT particles
//       double drag = this->cal_drag( 
//         flowDynamicViscosity, particleDiameter, flowDensity, up, up_1);
//       double relax = this->cal_relax(
//         flowDynamicViscosity, particleDiameter, particleDensity);
//       double reynolds = this->cal_reynolds(
//         flowDynamicViscosity, particleDiameter, flowDensity, up, up_1);

//       // Loop particle values for LPT
//       for (unsigned int ii = 0; ii < gdim; ii++)
//       {
//         // Calculate Wall lift for LPT particles - Dependent on axis
//         double lift = this->cal_WallLiftSq(
//           flowDynamicViscosity, particleDiameter, flowDensity, ii, up,
//           up_1, MicroH, MicroW);

//         // Calculate particle velocity within flow in axis direction
//         // Use current and prior position to calculate P.velocity
//         double particleVelocity = (up[ii] - up_1[ii]);
//         // _P->set_property(ci->index(), i, 1, up);
//         std::cout << "particleAcceleration: " << particleVelocity << std::endl;
//         // Calculating Force Balance term (drag and relax)
//         double ForceBal = ((drag * reynolds) / 24);
//         std::cout << "ForceBal1: " << ForceBal << std::endl;
//         ForceBal *= (relax);
//         std::cout << "ForceBal2: " << ForceBal << std::endl;
//         ForceBal *= particleVelocity;
//         std::cout << "ForceBal3: " << ForceBal << std::endl;
//         // Calculate lift (Based upon microfluidic walls)
//         ForceBal += lift;
//         // Set particleVelcity to up[ii]
//         up[ii] = ForceBal;
        
//         // std::cout << "P Velocity1: " << up[ii] << std::endl;
//       }
      

//       // If final dimension (Y (2D) or Z (3D)), apply buoyancy term
//       const double G = 9.8; // Gravity
//       double AddG = (particleDensity - flowDensity);
//       AddG *= G;
//       AddG /= particleDensity;

//       // up[2-1] -= AddG;
//       up[gdim-1] -= AddG;
//       //up[gdim-1] -= G * ((particleDensity - flowDensity)/ particleDensity);

//       // Feed adjusted particle velocity back into the "do_step" algorithm
//       //    by applyiong the change to "up" vector

//       std::size_t cidx_recv = ci->index();
//       double dt_rem = dt;

//       std::cout << "Flow Velocity AF: " << up << std::endl;

//       // Store previous particle velocity in slot 2
//       //    Important to store here once rebound applied
//       // std::cout << "up: " << (up) << std::endl;
//       _P->set_property(ci->index(), i, 1, up);
      
//       // Set current position to old slot before movement
//       // _P->set_property(ci->index(), i, 1, (_P->property(ci->index(), i, 0)));
//       std::cout << "up_1 Store: " << (_P->property(ci->index(), i, 1)) << std::endl;

//       while (dt_rem > 1E-15)
//       {
//         // Returns facet which is intersected and the time it takes to do so
//         std::tuple<std::size_t, double> intersect_info
//             = time2intersect(cidx_recv, dt_rem, _P->x(ci->index(), i), up);
//         const std::size_t target_facet = std::get<0>(intersect_info);
//         const double dt_int = std::get<1>(intersect_info);

//         if (target_facet == std::numeric_limits<unsigned int>::max())
//         {
//           // Then remain within cell, finish time step
//           _P->push_particle(dt_rem, up, ci->index(), i);
//           dt_rem = 0.0;

//           if (bounded_domain_active)
//             bounded_domain_violation(ci->index(), i);

//           // TODO: if step == last tstep: update particle position old to most
//           // recent value If cidx_recv != ci->index(), particles crossed facet
//           // and hence need to be relocated
//           if (cidx_recv != ci->index())
//             reloc.push_back({ci->index(), i, cidx_recv});
//         }
//         else
//         {
//           const Facet f(*mesh, target_facet);
//           const unsigned int* facet_cells = f.entities(tdim);

//           // Two options: if internal (==2) else if boundary
//           if (f.num_entities(tdim) == 2)
//           {
//             // Then we cross facet which has a neighboring cell
//             _P->push_particle(dt_int, up, ci->index(), i);

//             cidx_recv = (facet_cells[0] == cidx_recv) ? facet_cells[1]
//                                                       : facet_cells[0];

//             // Update remaining time
//             dt_rem -= dt_int;
//             if (dt_rem < 1E-15)
//             {
//               // Then terminate
//               dt_rem = 0.0;
//               if (cidx_recv != ci->index())
//                 reloc.push_back({ci->index(), i, cidx_recv});
//             }
//           }
//           else if (f.num_entities(tdim) == 1)
//           {
//             const facet_t ftype = facets_info[target_facet].type;
//             // Then we hit a boundary, but which type?
//             if (f.num_global_entities(tdim) == 2)
//             {
//               assert(ftype == facet_t::internal);
//               // Then it is an internal boundary
//               // Do a full push
//               _P->push_particle(dt_rem, up, ci->index(), i);
//               dt_rem *= 0.;

//               if (pbc_active)
//                 pbc_limits_violation(ci->index(),
//                                      i); // Check on sequence crossing internal
//                                          // bc -> crossing periodic bc

//               if (bounded_domain_active)
//                 bounded_domain_violation(ci->index(), i);

//               // TODO: do same for closed bcs to handle (unlikely event):
//               // internal bc-> closed bc

//               // Go to the particle communicator
//               reloc.push_back(
//                   {ci->index(), i, std::numeric_limits<unsigned int>::max()});
//             }
//             else if (ftype == facet_t::open)
//             {
//               // Particle leaLPTParametersk around: do a full push to make sure that
//               // particle is pushed outside domain
//               _P->push_particle(dt_rem, up, ci->index(), i);

//               // Then push back to relocate
//               reloc.push_back(
//                   {ci->index(), i, std::numeric_limits<unsigned int>::max()});
//               dt_rem = 0.0;
//             }
//             else if (ftype == facet_t::closed)
//             {
//               // Closed BC
//               apply_closed_bc(dt_int, up, ci->index(), i, target_facet);
//               dt_rem -= dt_int;
//             }
//             else if (ftype == facet_t::periodic)
//             {
//               // Then periodic bc
//               apply_periodic_bc(dt_rem, up, ci->index(), i, target_facet);
//               if (num_processes > 1) // Behavior in parallel
//                 reloc.push_back(
//                     {ci->index(), i, std::numeric_limits<unsigned int>::max()});
//               else
//               {
//                 // Behavior in serial
//                 std::size_t cell_id = _P->mesh()
//                                           ->bounding_box_tree()
//                                           ->compute_first_entity_collision(
//                                               _P->x(ci->index(), i));
//                 reloc.push_back({ci->index(), i, cell_id});
//               }
//               dt_rem = 0.0;
//             }
//             else if (ftype == facet_t::bounded)
//             {
//               std::cout << "Hit bounded facet " << std::endl;
//               // Then bounded bc
//               apply_bounded_domain_bc(dt_rem, up, ci->index(), i, target_facet);

//               if (num_processes > 1) // Behavior in parallel
//                 reloc.push_back(
//                     {ci->index(), i, std::numeric_limits<unsigned int>::max()});
//               else
//               {
//                 // Behavior in serial
//                 std::size_t cell_id = _P->mesh()
//                                           ->bounding_box_tree()
//                                           ->compute_first_entity_collision(
//                                               _P->x(ci->index(), i));
//                 reloc.push_back({ci->index(), i, cell_id});
//               }
//               dt_rem = 0.0;
//             }
//             else
//             {
//               dolfin_error("advect_particles.cpp::do_step",
//                            "encountered unknown boundary",
//                            "Only internal boundaries implemented yet");
//             }
//           }
//           else
//           {
//             dolfin_error("advect_particles.cpp::do_step",
//                          "found incorrect number of facets (<1 or > 2)",
//                          "Unknown");
//           }

//         } // end else
//       }   // end while
//     }     // end for
//   }       // end for

//   // Relocate local and global
//   _P->relocate(reloc);
// }


//-----------------------------------------------------------------------------
double advect_particles::cal_drag(double dynVisc, 
  double particleDiameter, double flowDensity, Point& up, Point& up_1)
//  Drag coefficent determines the amount of drag acting upon the particle. 
//    Above a Reynolds number of 1000, flow is turbulent and drag is
//    approximately 0.44.
{

  if (dynVisc == 0)
  {
    return -1.0 * std::numeric_limits<double>::infinity();
  }

  double reynolds = cal_reynolds(dynVisc, particleDiameter, flowDensity, up, up_1);

  if (reynolds < 1000)
  {
    double rhs = pow(reynolds, 0.687);
    rhs *= 0.15;
    rhs += 1.0;
    double ans = (24 / reynolds);
    ans *= rhs;
    return ans;
    //return (24 / reynolds)*(1.0 + 0.15 * pow(reynolds, 0.687));
  }
  else
  {
    return 0.44;
  }

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
  double particleDiameter, double flowDensity, Point& up, Point& up_1)
// Reynolds number calculation requires average speed to calcualte
//    laminar flow (<1) or turbulent flow (>1).
//
{ 
  Point relativeVelocity;

  for (int i = 0; i < 3; i++)
  {
    // Particle Velocity - Flow Velocity
    relativeVelocity[i] = (up_1[i] - up[i]);
    // std::cout << relativeVelocity[i] << std::endl;
  }

  // Calculate relative speed of particle
  double relativeSpeed = relativeVelocity.norm();

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
  double h, double w, int s)
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

  // 4.0 * Umax * x[1] * (0.41 - x[1]) / pow(0.41, 2)
  // up[ii] = up[ii] * ( LPTVecT / upNorm);
  
  // s is the particle position from the walls important for lift
  //  This is generalised on a 2D parabolic and does not indicate
  //    closeness to a particular wall
  // x is uNorm, uMax is maximum flow rate
  // Function is divided by H to make percentage and times by 100
  //    to make it accessable "int" for GSpot
  // uNorm1 is the centre line of the channel for velocities, expected to be
  //    parabolic and avoid issues where s can be small on XY axis when 
  //    particle is centred.
  if (s != 0)
  {
    s = (((H/2 - H/2 * sqrt(1 - (1/uMax) * uNorm1)) / H) * 100);
  }

  // If Z axis, use Z axis height and P position
  //  Assumed is Z is a constant height
  if (i == 2)
  {
    // Point pPos = _P->x(ci->index(), 1);
    // s = ( Z particle position minus zMin ) / Zrange
    // std::cout << "pPos[2]: " << pPos[2] << std::endl;
    // std::cout << "Za axis S value: " << ((pPos[2] - 0.0110585) / 0.00016) << std::endl;
    // s needs to be out of 100, not decimal
    s = ((pPos[2] - 0.0110585) / 0.00016) * 100;
    // Set Z axis to something reasonable?
    // uNorm = std::abs(up[2]); // uMax * 0.01;//std::abs(up[2]);
    uMax = 0.1; // Bouyancy acting upon particle
    H = 0.00016; // Set Z to boundary height, not hydraulic
  }

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
  double Flnl = pow( ( (particleDiameter / 2) / H ) , 2);
  Flnl *= (flowDensity * uMax * H) / dynVisc; // Reynolds;
  Flnl *= CL;
  Flnl = std::abs(Flnl);

  // Method using Di Carlo
  // double Flnl = flowDensity;
  // Flnl = flowDensity;
  // Flnl *= pow( uMax, 2 );
  // Flnl *= pow( (particleDiameter / 2), 6 );
  // Flnl /= pow( H, 4 );

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
      ratioU += std::abs(up[0]) / std::abs(up[1]);
    }
    
    // std::cout << "up1 Add lift: " << up1 << std::endl;
    // up1 is the direction along the X or Y axis the particle is moving
    //  This assumes a parabolic where flow drops off towards the boundaries
    //  Therefore, particle movement should be in the opposite direction to movement
    Flnl *= (std::abs(up[iP]) / ratioU);
    if (up1[i] < 0) // Positive along axis
    {
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


  return Flnl;
  //return 1;
  //return (0.5 * flowDensity * pow(relativeSpeed,2) * pow(particleDiameter,4)) / pow(H, 2);
}

//-----------------------------------------------------------------------------
// double advect_particles::cal_ParticleDistFromBoundary(Point& pp, const Mesh* mesh)
// {
//   // Calculate particle's distance from boudnary on axis
//   // Mesh* mesh = _P->mesh();

//   Mesh mesh1 = mesh;
//   // BoundingBoxTree::build(mesh1); //bbtree //= BoundingBoxTree.build(mesh);
//   // bbtree -> init()
//   // bbtree->
//   double distance = 0;
//   int icup;



//   icup, distance = BoundingBoxTree::compute_closest_entity(pp);
//   std::cout << "P distance from boundary: " << distance << std::endl;

//   return distance;
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
      

      up = do_stepLPT(dt, up, up_1, up1, pPos1, LPTParameters);

      // Store previous particle velocity in slot 2 
      //    Important to store here once rebound applied
      // std::cout << "up: " << (up) << std::endl;
      _P->set_property(ci->index(), i, 1, up);
      
      // Set current position to old slot before movement
      // _P->set_property(ci->index(), i, 1, (_P->property(ci->index(), i, 0)));
      std::cout << "up_1 Store: " << (_P->property(ci->index(), i, 1)) << std::endl;
    

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
        up = do_stepLPT(dt, up, up_1, up1, pPos1, LPTParameters);

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
        do_substep(dt, up, ci->index(), i, step, num_substeps, xp0_idx, up0_idx,
                   reloc);
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
        up = do_stepLPT(dt, up, up_1, up1, pPos1, LPTParameters);

        
        // Set current position to old slot before movement
        // _P->set_property(ci->index(), i, 1, (_P->property(ci->index(), i, 0)));
        // std::cout << "up_1 Store: " << (_P->property(ci->index(), i, 1)) << std::endl;
        std::cout << "P Velocity AF: " << up << std::endl;
    


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
        
        std::cout << "up1: " << (up) << std::endl;
        // _P->set_property(ci->index(), i, 1, up);
        // Do substep
        do_substep(dt * dti[step], up, ci->index(), i, step, num_substeps,
                   xp0_idx, up0_idx, reloc);
                   
       //if rebound == true
            // End this iteration and rebound
        // dt_rem
        std::cout << "steps: " << step << std::endl;
                   
        // Store previous particle velocity in slot 2 
        //    Important to store here once rebound applied
        //      -Rebound applied in substep-
        std::cout << "up2 AFsubstep: " << (up) << std::endl;
        

        _P->set_property(ci->index(), i, 1, up);
        std::cout << "P Velocity Final: " << up << std::endl;
                   
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
        

        // up = do_stepLPT(dt, up, up_1, up1, LPTParameters);
        up = do_stepLPT(dt, up, up_1, up1, pPos1, LPTParameters);

        // Store previous particle velocity in slot 2 
        //    Important to store here once rebound applied
        // std::cout << "up: " << (up) << std::endl;
        _P->set_property(ci->index(), i, 1, up);
        
        // Set current position to old slot before movement
        // _P->set_property(ci->index(), i, 1, (_P->property(ci->index(), i, 0)));
        std::cout << "up_1 Store: " << (_P->property(ci->index(), i, 1)) << std::endl;
    

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
