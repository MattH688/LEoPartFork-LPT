# -*- coding: utf-8 -*-
# Copyright (C) 2018 Jakob Maljaars
# Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import math
import dolfin
import dolfin.cpp as cpp
from mpi4py import MPI as pyMPI
import pickle
import os

"""
    Wrapper for the CPP functionalities
"""

__all__ = [
    "particles",
    "advect_particles",
    "advect_rk2",
    "advect_rk3",
    "l2projection",
    "StokesStaticCondensation",
    "PDEStaticCondensation",
    "AddDelete",
]

from leopart.cpp import particle_wrapper as compiled_module

comm = pyMPI.COMM_WORLD


class particles(compiled_module.particles):
    """
    Python interface to cpp::particles.h
    """

    def __init__(self, xp, particle_properties, mesh):
        """
        Initialize particles.

        Parameters
        ----------
        xp: np.ndarray
            Particle coordinates
        particle_properties: list
            List of np.ndarrays with particle properties.
        mesh: dolfin.Mesh
            The mesh on which the particles will be generated.
        """

        gdim = mesh.geometry().dim()

        particle_template = [gdim]
        for p in particle_properties:
            if len(p.shape) == 1:
                particle_template.append(1)
            else:
                particle_template.append(p.shape[1])

        p_array = xp
        for p_property in particle_properties:
            # Assert if correct size
            assert p_property.shape[0] == xp.shape[0], "Incorrect particle property shape"
            if len(p_property.shape) == 1:
                p_array = np.append(p_array, np.array([p_property]).T, axis=1)
            else:
                p_array = np.append(p_array, p_property, axis=1)

        compiled_module.particles.__init__(self, p_array, particle_template, mesh)
        self.ptemplate = particle_template

    def interpolate(self, *args):
        """
        Interpolate field to particles. Example usage for updating the first property
        of particles. Note that first slot is always reserved for particle coordinates!

        .. code-block:: python

            p.interpolate(psi_h , 1)

        Parameters
        ----------
        psi_h: dolfin.Function
            Function which is used to interpolate
        idx: int
            Integer value indicating which particle property should be updated.

        """
        a = list(args)
        if not isinstance(a[0], cpp.function.Function):
            a[0] = a[0]._cpp_object
        super().interpolate(*tuple(a))

    def increment(self, *args):
        """
        Increment particle at particle slot by an incrementatl change
        in the field, much like the FLIP approach proposed by Brackbill

        The code to update a property psi_p at the first slot with a
        weighted increment from the current time step and an increment
        from the previous time step, can for example be implemented as:

        .. code-block:: python

            #  Particle
            p=particles(xp,[psi_p , dpsi_p_dt], msh)

            #  Incremental update with  theta =0.5, step=2
            p.increment(psih_new , psih_old ,[1, 2], theta , step

        Parameters
        ----------
        psih_new: dolfin.Function
            Function at new timestep
        psih_old: dolfin.Function
            Function at old time step
        slots: list
            Which particle slots to use? list[0] is always the quantity
            that will be updated
        theta: float, optional
            Use weighted update from current increment and previous increment/
            theta = 1: only use current increment
            theta = 0.5: average of previous increment and current increment
        step: int
            Which step are you at? The theta=0.5 increment only works from step >=2

        """

        a = list(args)
        if not isinstance(a[0], cpp.function.Function):
            a[0] = a[0]._cpp_object
        if not isinstance(a[1], cpp.function.Function):
            a[1] = a[1]._cpp_object
        super().increment(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)

    def return_property(self, mesh, index):
        """
        Return particle property by index.

        **FIXME**: mesh input argument seems redundant.

        Parameters
        ----------
        mesh: dolfin.Mesh
            Mesh
        index: int
            Integer index indicating which particle property should be returned.

        Returns
        -------
        np.array
            Numpy array which stores the particle property.
        """

        pproperty = np.asarray(self.get_property(index))
        if self.ptemplate[index] > 1:
            pproperty = pproperty.reshape((-1, self.ptemplate[index]))
        return pproperty

    def number_of_particles(self):
        """
        Get total number of particles

        Returns
        -------
        int:
            Global number of particles
        """
        xp_root = comm.gather(self.positions(), root=0)
        if comm.rank == 0:
            xp_root = np.float16(np.vstack(xp_root))
            num_particles = len(xp_root)
        else:
            num_particles = None
        num_particles = comm.bcast(num_particles, root=0)
        return num_particles

    def dump2file(self, mesh, fname_list, property_list, mode1, clean_old=False):
        """
        Export files in '.csv' format for Paraview visualisation of particle tracks

        Requires:
            mesh
            fname_list = file location + file name
                /ExampleData/
                
        Visualise in Paraview by...
        1) Import Particle .csv file
        2) Use filter "TableToPoints"
        2.5) Use "Representation" field to visualise particles
        """
        import csv
        import os
        
        CellPartVelAdj = False
        stop = False
        # Legacy code - Unsure on purpose but used for importing process
        if isinstance(fname_list, str) and isinstance(property_list, int):
            fname_list = [fname_list]
            property_list = [property_list]



        # ## Make folder to store particles in
        fname_list[0] = os.path.join(fname_list[0], "Particles")
        if (comm.Get_rank() == 0) and not os.path.exists(fname_list[0]):
            os.makedirs(fname_list[0])
        # print(fname_list)

        # assert isinstance(fname_list, list) and isinstance(property_list, list), (
        #     "Wrong dump2file" " request"
        # )
        # assert len(fname_list) == len(property_list), (
        #     "Property list and index list must " "have same length"
        # )

        # # Remove files if clean_old = True
        # if clean_old:
        #     for fname in fname_list:
        #         try:
        #             os.remove(fname)
        #         except OSError:
        #             pass

        for (property_idx, fname) in zip(property_list, fname_list):
            ParticleNum = comm.gather(self.return_property(mesh, 2).T, root=0)
            property_root = comm.gather(self.return_property(mesh, property_idx).T, root=0)

            if comm.Get_rank() == 0:
                if not ParticleNum[0]:
                    print("No particles detected")
                    CellPartVelAdj = False
                    stop = True
                else:
                    # with open('', mode='a+') as csvfile:
                    Ftext = "{}/Particle_{:.0f}.csv".format(fname_list[0], ParticleNum[0][0])
                    # Check if file exists to put row ID in
                    if not os.path.exists(Ftext):
                        RowID = True
                    else:
                        RowID = False
                    if (mode1=="a+"):
                        with open(Ftext, mode=mode1) as csvfile:
                            CSVwriter = csv.writer(csvfile, delimiter=',')

                            # spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|',
                            #     quoting=csv.QUOTE_MINIMAL)
                            # with open(fname, mode) as f:
                            property_root = np.float16(np.hstack(property_root).T)
                            # print(property_root)
                            # print(property_root[0])
                        #     pickle.dump(property_root, f))
                            # print("Length of Co-ords: ",len(property_root[0]))
                            if (RowID == True):
                                if (len(property_root[0]) == 3):
                                    CSVwriter.writerow(['x', 'y', 'z'])
                                else:
                                    CSVwriter.writerow(['x', 'y'])
                            CSVwriter.writerow(property_root[0])


                    if (mode1=="r"):
                        with open(Ftext) as f:
                            
                            N = 100
                            lastN = list(f)[-N:]
                            # print("is open: ", (lastN == True))

                            if (((not lastN) == False) & (all(lastN[0] == item for item in lastN))):
                                CellPartVelAdj = True
                                print("List all same")
                            # else:
                            #     CellPartVelAdj = False

                            #all(x==myList[0] for x in myList)
                            
        return CellPartVelAdj, stop
        
    def _Particle_Distance_To_Boundary(self, mesh, bmesh, particles, Zmid):

        """
        Similar to dump2file...
        
        1) Open each particle
        2) Extract co-ordinates
        3) Get distance from boundary selected
        
        """
        import math

        Zmid *= 100
        # P1 = df.Point(0.0007862,0.0115,Zmid) # Inlet Straight#0.011138502
        # P12 = df.Point(0.00937056, 0.0110739, (Zmid)) # Inlet Curve
        # P2 = df.Point(0.00137025, 0.00867547, Zmid) # Outlet Straight
        
        # # # Scale mesh Z axis
        # x = bmesh.coordinates()
        # x[:, 2] *= 100

        # # Creating bonding box tree of mesh
        # #     where Z axis is very large
        bbtree = mesh.bounding_box_tree()
        bbtree.build(bmesh)

        candidate_cells = [c.index() for c in dolfin.cells(mesh)]
        num_properties = particles.num_properties()
        print(num_properties)

        for c in candidate_cells:
            for pi in range(particles.num_cell_particles(c)):
                particle_props = list(particles.property(c, pi, prop_num)
                                    for prop_num in range(num_properties))

                pPos = particle_props[0]
                print("pPos: ", pPos.x(), pPos.y(), pPos.z())

                P2 = dolfin.Point(pPos[0], pPos[1], Zmid)
                # # Distance to closest XY exterior boundary 
                d1, distance = bbtree.compute_closest_entity(P2)


                # # Find vertices from mesh to find co-ordinate position of closet entity
                closest_cell = bmesh.cells()[d1]
                vertices_of_closest_cell = bmesh.coordinates()[closest_cell,:]
            
                # # Temporary distance D1
                # #     Make sure D1 is greater than XY width
                D1 = 100
                
                # # P3 Particle position without Z axis
                P3 = dolfin.Point(P2.x(), P2.y(), 0)

                ii = 0
                xy = 0
                P3x = [0,0,0]
                P3y = [0,0,0]

                
                # # Assumes triangle, cycle through 3 points
                for i in vertices_of_closest_cell:

                    # P1 verticies without Z axis
                    P1 = dolfin.Point(i[0], i[1], 0)
                    # Calculate nearest point
                    D = P3.distance(P1)
                    
                    # # Particle - Vertex
                    # P3x[ii] = P3.x() - P1.x()
                    # P3y[ii] = P3.y() - P1.y()

                    # If distance is smaller than previous distance
                    if D1 > D:
                        D1 = D
                        # Cal. vector from particle to vertex
                        # P4 = P3 - P1
                        # Save point to cal. norm off
                        P4a = P1



                # # Calculate XY line equation
                # y = mx + b
                xy = (vertices_of_closest_cell[1] 
                    - vertices_of_closest_cell[0])
                print("xy: ", xy)

                if ((xy[0] != 0)):
                    # Cal. m, y / x = m (assume b = 0)
                    mVal = xy[1] / xy[0]
                    # Cal. b, y - mx = b
                    bVal = (vertices_of_closest_cell[0][1] 
                        - (vertices_of_closest_cell[0][0] * mVal))
                    # # Using y = xc, b + (x * m) = y
                    # x is the particle pos.
                    ymVal = (P3[0] * mVal) + bVal
                    # y minus pPos y
                    print(ymVal)
                    ymVal -= P3[1]
                    print(ymVal)
                    # x = (y - b) / c
                    # y is the particle pos.
                    xmVal = (P3[1] - bVal) / mVal
                    # x minus pPos x
                    print(xmVal)
                    xmVal -= P3[0]
                    print(xmVal)
                    #
                    if (ymVal < 0):
                        yc = 1
                    else:
                        yc = -1

                    if (xmVal < 0):
                        xc = 1
                    else:
                        xc = -1
                    
                    # #  Calculate norm of the surface
                    ## B - A
                    BA = (dolfin.Point(vertices_of_closest_cell[1]
                        - vertices_of_closest_cell[0]))
                    ## C - A
                    CA = (dolfin.Point(vertices_of_closest_cell[2]
                        - vertices_of_closest_cell[0]))

                    print("BA: ", BA.x(), BA.y(), BA.z())
                    print("CA: ", CA.x(), CA.y(), CA.z())

                    ## Calculating without the Z axis
                    FaceNormVector = [0,0,0]
                    FaceNormVector[1] = abs(BA.x() * CA.x())
                    FaceNormVector[0] = abs(BA.y() * CA.y())
                else:
                    # Boundary Pos. minus Particle position
                    xVal = P4a[0] - P3[0]

                    if (xVal < 0):
                        xc = 1
                    else:
                        xc = -1

                    yc = 1

                    FaceNormVector = [0,0,0]
                    FaceNormVector[0] = abs(distance) * xc

                FaceNormVector = dolfin.Point(FaceNormVector)


                print("FaceNormVector: ", FaceNormVector.x(), FaceNormVector.y(),
                    FaceNormVector.z())

                # Escape if FaceNormVector is 0
                #   Uses previously calculated values (pos and H)
                print((FaceNormVector.x() != 0))
                print((FaceNormVector.y() != 0))
                if ((FaceNormVector.x() == 0) and (FaceNormVector.y() == 0)):
                # if ((FaceNormVector.x() != 0) and (FaceNormVector.y() != 0)):
                    break
                else:
                    ## Calculate the base magnitude of facenorm
                    Mag = math.sqrt((FaceNormVector.x() * FaceNormVector.x())
                        + (FaceNormVector.y() * FaceNormVector.y()))

                    print("Mag: ", Mag)

                    if (yc < 0):
                        FaceNormVector[1] *= -1

                    if (xc < 0):
                        FaceNormVector[0] *= -1

                    print("FaceNormVector: ", FaceNormVector.x(), FaceNormVector.y(),
                        FaceNormVector.z())

                    iI = 2
                    # D1 = distance
                    # Replace particle position with vertex position
                    P3 = P4a

                    # Add 1.5 distance vector to point moving particle
                    #   positive away from closest vertex to opposite vertex
                    # FaceNormVector1 = FaceNormVector
                    # FaceNormVector1[0] = FaceNormVector.x() * ((distance * 1.5) / Mag)
                    # FaceNormVector1[1] = FaceNormVector.y() * ((distance * 1.5) / Mag)
                    # FaceNormVector1[2] = 0



                    # Add normal distance to FaceNormVector
                    FaceNormVector[0] = FaceNormVector.x() * ((distance) / Mag)
                    FaceNormVector[1] = FaceNormVector.y() * ((distance) / Mag)
                    # FaceNormVector[2] = 0

                    # # Add initial distance
                    P3 += (FaceNormVector * 2)

                    while iI < 200:

                        # Reset Z axis to Zmid otherwise adds very large distance
                        P3 = dolfin.Point(P3.x(), P3.y(), Zmid)

                        print("P3: ", P3.x(), P3.y(), P3.z())

                        # Check if past midpoint of the channel so particle direction is
                        #   opposite in the X or Y axis (whichever is largest - see above)
                        d1, distance1 = bbtree.compute_closest_entity(P3)

                        print("Distance1: ", distance1)

                        # Check if distance prior is smaller than current distance
                        if (distance1 < (distance)):

                            closest_cell = bmesh.cells()[d1]
                            vertices_of_closest_cell = bmesh.coordinates()[closest_cell,:]

                            P1 = dolfin.Point(vertices_of_closest_cell[0][0],
                            vertices_of_closest_cell[0][1], Zmid)

                            # Produce vector from vertex + distance from newest vertex
                            # P5 = P3 - P1
                            P5 = P1 - P4a
                            print("P1: ", P1.x(), P1.y(), P1.z())
                            print("P4a: ", P4a.x(), P4a.y(), P4a.z())
                            print("P5 vec", P5.x(), P5.y())

                            DistanceT1 = math.sqrt((P5.x() * P5.x())
                                + (P5.y() * P5.y()))

                            break
                        else:
                            # Add distance to vertex every iteration. 
                            P3 += FaceNormVector

                        closest_cell = bmesh.cells()[d1]
                        vertices_of_closest_cell = bmesh.coordinates()[closest_cell,:]

                        P1 = dolfin.Point(vertices_of_closest_cell[0][0],
                            vertices_of_closest_cell[0][1], Zmid)

                        # Produce vector from vertex + distance from newest vertex
                        # P5 = P3 - P1
                        # P5 = P1 - P3
                        # print("P5 vec", P5.x(), P5.y())

                        # # Check vector to closest vertex is the opposite of FaceNormVector
                        # # if ((((FaceNormVector.x() > 0) & (0 > P5.x()))
                        # #     or ((FaceNormVector.x() < 0) & (0 < P5.x())))
                        # #     & (((FaceNormVector.y() > 0) & (0 > P5.y())) 
                        # #     or ((FaceNormVector.y() < 0) & (0 < P5.y())))):
                        # if ((((FaceNormVector.x() > 0) & (P5.x() > 0))
                        #     or ((FaceNormVector.x() < 0) & (P5.x() < 0)))
                        #     & (((FaceNormVector.y() > 0) & (P5.y() > 0)) 
                        #     or ((FaceNormVector.y() < 0) & (P5.y() < 0)))):
                            
                        #     print("break")
                        #     # Escape as found distance from old vertex to new vertex
                        #     break
                        # else:
                        #     # Add distance to vertex every iteration. 
                        #     P3 += FaceNormVector
                        # Ensure additional iteration count is added
                        iI += 1
                    # Escape if FaceNormVector is 0, set to maximum known length
                    else:
                        iI = 200

                    # To avoid excessive iteration, set default to 500 um
                    if (iI >= 200) or (DistanceT1 == 0):
                        # # Save distance to slot 6,1 of particle
                        # Use previous distanceT
                        distanceT = particle_props[3][1]
                        dPoint = dolfin.Point(distance, distanceT, 0)

                    else:
                        # Calculate distance from old to new vertex (total distance)
                        distanceT = (distance * iI) + distance1

                        # # Save distance, diameter and [0] to slot 6 of particle
                        dPoint = dolfin.Point(distance, DistanceT1, 0)
                        print("Total DistanceT1: ", (DistanceT1))

                    print("dPoint: ", dPoint.x(), dPoint.y(), dPoint.z())

                    particles.set_property(c, pi, 3, dPoint)
                    particles.set_property(c, pi, 4, FaceNormVector)

                    print("Distance: ", distance)
                    # print("Distance 1: ", (distance + distance1))
                    print("Total Distance: ", (distanceT))

                    # set.time(5)
        
        # Need to reset bounding box tree
        bbtree = mesh.bounding_box_tree()
        bbtree.build(mesh)
        # Need to write distance S XY and Z to particle

    def _build_exterior_mesh(self, mesh, XYZ):
        # Create an exterior mesh
        bmesh = dolfin.BoundaryMesh(mesh, "exterior")
        
        # # Scale mesh Z axis
        x = bmesh.coordinates()
        if XYZ < 2:
            x[:, 0:1] *= 100
        else:
            x[:, 2] *= 100

        # # Creating bonding box tree of mesh
        # #     where Z axis is very large
        # bbtree = mesh.bounding_box_tree()
        # bbtree.build(bmesh)

        return bmesh

            
def _parse_advect_particles_args(args):
    args = list(args)
    args[1] = args[1]._cpp_object
    if isinstance(args[2], dolfin.Function):
        uh_cpp = args[2]._cpp_object

        def _default_velocity_return(step, dt):
            return uh_cpp
        args[2] = _default_velocity_return
    return args


class advect_particles(compiled_module.advect_particles):
    """
    Particle advection with Euler method
    """

    def __init__(self, *args):
        """
        Initialize class

        Parameters
        ----------
        p: particles
            Particles instance
        V: dolfin.FunctionSpace
            FunctionSpace for the particle advection
            # TODO: can be derived from Function
        v: dolfin.Function
            Dolfin Function that will be used for the
            advection
        bc: string
            Boundary type. Any of "closed", "open" or "periodic"
        lims: np.array, optional
            Optional array for specifying the connected boundary parts
            in case of periodic bc's
        """
        a = _parse_advect_particles_args(args)
        super().__init__(*tuple(a))

    def do_step(self, *args):
        """
        Advect the particles over a timestep

        Parameters
        ----------
        dt: float
            Timestep
        """
        super().do_step(*args)
        
    def do_step_LP(self, *args):
        """
        Advect the particles over a timestep

        Parameters
        ----------
        dt: float
            Timestep
        """
        super().do_step(*args)

    def __call__(self, *args):
        return self.eval(*args)


class advect_rk2(compiled_module.advect_rk2):
    """
    Particle advection with RK2 method
    """

    def __init__(self, *args):
        """
        Initialize class

        Parameters
        ----------
        p: particles
            Particles instance
        V: dolfin.FunctionSpace
            FunctionSpace for the particle advection
            # TODO: can be derived from Function
        v: dolfin.Function
            Dolfin Function that will be used for the
            advection
        bc: string
            Boundary type. Any of "closed", "open" or "periodic"
        lims: np.array, optional
            Optional array for specifying the connected boundary parts
            in case of periodic bc's
        """
        a = _parse_advect_particles_args(args)
        super().__init__(*tuple(a))

    def do_step(self, *args):
        """
        Advect the particles over a timestep

        Parameters
        ----------
        dt: float
            Timestep
        """
        super().do_step(*args)

    def __call__(self, *args):
        return self.eval(*args)


class advect_rk3(compiled_module.advect_rk3):
    """
    RK3 advection
    """

    def __init__(self, *args):
        a = _parse_advect_particles_args(args)
        super().__init__(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)


class advect_rk4(compiled_module.advect_rk4):
    def __init__(self, *args):
        """
        Initialize class

        Parameters
        ----------
        p: particles
            Particles instance
        V: dolfin.FunctionSpace
            FunctionSpace for the particle advection
            # TODO: can be derived from Function
        v: dolfin.Function
            Dolfin Function that will be used for the
            advection
        bc: string
            Boundary type. Any of "closed", "open" or "periodic"
        lims: np.array, optional
            Optional array for specifying the connected boundary parts
            in case of periodic bc's
        """
        a = _parse_advect_particles_args(args)
        super().__init__(*tuple(a))

    def do_step(self, *args):
        """
        Advect the particles over a timestep

        Parameters
        ----------
        dt: float
            Timestep
        """
        super().do_step(*args)

    def __call__(self, *args):
        return self.eval(*args)


class l2projection(compiled_module.l2projection):
    """
    Class for handling the l2 projection from particle
    properties onto a FE function space.
    """

    def __init__(self, *args):
        """
        Initialize class

        Parameters
        ----------
        p: particles
            Particles object
        V: dolfin.FunctionSpace
            FunctionSpace that will be used for the
            projection
        property_idx: int
            Which particle property to project?
        """
        a = list(args)
        a[1] = a[1]._cpp_object
        super().__init__(*tuple(a))

    def project(self, *args):
        """
        Project particle property onto discontinuous
        FE function space

        Parameters
        ----------
        vh: dolfin.Function
            dolfin.Function into which particle properties
            are projected. Must match the specified
            FunctionSpace
        lb: float, optional
            Lowerbound which will activate a box-constrained
            projection. Should come in pairs with the upperbound
            ub.
        ub: float, optional
            Upperbound, for box-constrained projection.
            Should come in pairs with lowerbound lb
        """

        a = list(args)
        if not isinstance(a[0], cpp.function.Function):
            a[0] = a[0]._cpp_object
        super().project(*tuple(a))

    def project_cg(self, *args):
        """
        Project particle property onto continuous
        FE function space

        **NOTE**: this method is a bit a bonus and
        certainly could be improved

        Parameters
        ----------
        A: dolfin.Form
            bilinear form for the rhs
        f: dolfin.Form
            linear form for the rhs
        u: dolfin.Function
            dolfin.Function on which particle properties
            are projected.
        """
        super.project_cg(self, *args)

    def __call__(self, *args):
        return self.eval(*args)


class StokesStaticCondensation(compiled_module.StokesStaticCondensation):
    """
    Class for solving the HDG Stokes problem.
    Class interfaces the cpp StokesStaticCondensation class
    """

    def __init__(self, *args):
        """
        Parameters
        ----------
        args
        """
        super().__init__(*args)

    def solve_problem(self, *args):
        """
        Solve the Stokes problem

        Parameters
        ----------
        args
        """
        a = list(args)
        for i, arg in enumerate(a):
            # Check because number of functions is either 2 or 3
            if not isinstance(arg, str):
                if not isinstance(arg, cpp.function.Function):
                    a[i] = a[i]._cpp_object
        super().solve_problem(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)


class PDEStaticCondensation(compiled_module.PDEStaticCondensation):
    """
    Class for projecting the particle properties onto a discontinuous
    mesh function via a PDE-constrained projection in order to ensure
    conservation properties.

    Class interfaces PDEStaticCondensation
    """

    def solve_problem(self, *args):
        """
        Solve the PDE-constrained projection

        Parameters
        ----------
        args
        """
        a = list(args)
        for i, arg in enumerate(a):
            # Check because number of functions is either 2 or 3
            if not isinstance(arg, str):
                if not isinstance(arg, cpp.function.Function):
                    a[i] = a[i]._cpp_object
        super().solve_problem(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)


class AddDelete(compiled_module.AddDelete):
    """
    Class for adding/deleting particles
    """

    def __init__(self, *args):
        """
        Initialize class

        Parameters
        ----------
        args
        """
        a = list(args)
        for i, func in enumerate(a[3]):
            a[3][i] = func._cpp_object
        super().__init__(*tuple(a))

    def __call__(self, *args):
        return self.eval(*args)
