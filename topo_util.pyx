""" MOAB MeshTopoUtil """
# cython: boundscheck=False
from cython.operator cimport dereference as deref

cimport numpy as np
import numpy as np
from pymoab cimport eh
from pymoab import rng
from .tag cimport Tag, _tagArray
from .rng cimport Range
from .core cimport Core
from .types import check_error, _eh_array
from . import types
from libcpp.vector cimport vector
from libcpp.list cimport list as cpplist
import time

cdef class MeshTopoUtil(object):

    def __cinit__(self, Core c):
        self.interface  = <moab.Interface*> c.inst
        self.inst = new moab.MeshTopoUtil(self.interface)

    def __del__(self):
        del self.inst

    def get_bridge_adjacencies(self,
                               from_ent,
                               int bridge_dim,
                               int to_dim,
                               int num_layers = 1,
                               exceptions = ()):
        """
        Get "bridge" or "2nd order" adjacencies of dimension 'to_dim', going
        through dimension 'bridge_dim'.

        Example
        -------
        root_set = mb.get_root_set()
        all_volumes = mb.get_entities_by_dimension(root_set, 3)
        adj_volumes = mtu.get_bridge_adjacencies(all_volumes[0], 2, 3)

        Parameters
        ----------
        from_ent : Range of EntityHandles or EntityHandle
            Entity or entities to get adjacent entities from.
        bridge_dim : integer
            Dimension of the bridge entities.
        to_dim : integer
            Dimension of the target entities.
        num_layers : integer
            Depth of the adjacency search. If from_ent is a EntityHandle this
            parameter is ignored.
        exceptions : tuple
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
          Returns a Range containing the adjacent entities.

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs.
        """
        cdef moab.ErrorCode err
        cdef moab.EntityHandle ms_handle
        cdef Range r
        cdef Range adjs = Range()

        if isinstance(from_ent, Range):
            r = from_ent
            err = self.inst.get_bridge_adjacencies(deref(r.inst), bridge_dim, to_dim, deref(adjs.inst), num_layers)
        else:
            ms_handle = from_ent
            err = self.inst.get_bridge_adjacencies(ms_handle, bridge_dim, to_dim, deref(adjs.inst))

        check_error(err, exceptions)

        return adjs

    def get_ord_bridge_adjacencies(self,
                               from_ent,
                               int bridge_dim,
                               int to_dim,
                               all_ents = None,
                               int level = 0,
                               exceptions = ()):
        cdef moab.ErrorCode err
        cdef moab.EntityHandle ms_handle
        cdef int idx_count = 0
        cdef bint jagged = 0
        cdef int default_size = 0
        cdef vector[eh.EntityHandle] tempVector
        cdef np.ndarray[np.uint64_t] temp_array
        cdef np.ndarray[dtype = np.uint64_t, ndim = 1] inputArray
        cdef np.ndarray[dtype = np.int32_t, ndim = 1] idx_array
        cdef int siz
        if isinstance(from_ent, Range):
          inputArray = from_ent.get_array()
        elif isinstance(from_ent, np.ndarray):
          inputArray = from_ent
        else:
          inputArray = np.array(from_ent, dtype = np.uint64)
        siz = inputArray.size
        cdef Range adjs = Range()
        cdef int i
        cdef int j
        cdef int sizj
        idx_array = np.empty(siz, dtype = np.int32)
        for i in range(siz):
          err = self.inst.get_bridge_adjacencies(inputArray[i], bridge_dim, to_dim, deref(adjs.inst))
          check_error(err, exceptions)
          if level:
            adjs = rng.intersect(adjs, all_ents)
          temp_array = adjs.get_array()
          sizj = temp_array.size
          tempVector.insert(tempVector.end(), <eh.EntityHandle*> temp_array.data, <eh.EntityHandle*> temp_array.data+sizj)
          if not jagged:
            if default_size==0:
              default_size = sizj
            elif default_size != sizj:
              jagged = 1
          idx_count = idx_count + sizj
          idx_array[i] = idx_count
          adjs.clear()
        if jagged:
          return np.array(tempVector, dtype=np.uint64), idx_array, jagged
        return np.array(tempVector, dtype=np.uint64), default_size, jagged

    def get_average_position(self,
                             entity_handles,
                             exceptions = ()):
        """
        Returns the average position of the entities adjacent vertices average
        position.

        Example
        -------
        root_set = mb.get_root_set()
        all_volumes = mb.get_entities_by_dimension(root_set, 3)
        mtu.get_average_position(all_volumes)

        Parameters
        ----------
        entities : Range or iterable of EntityHandles
            Entities which adjacent vertices will be averaged.
        exceptions : tuple
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
          Returns average position as a 1-D Numpy array of xyz values.

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef moab.EntityHandle ms_handle
        cdef Range r
        cdef np.ndarray[np.uint64_t, ndim=1] arr
        cdef np.ndarray avg_position

        if isinstance(entity_handles, Range):
            r = entity_handles
            avg_position = np.empty((3,),dtype='float64')
            err = self.inst.get_average_position(deref(r.inst), <double*> avg_position.data)
            check_error(err, exceptions)
        else:
            arr = _eh_array(entity_handles)
            avg_position = np.empty((3,),dtype='float64')
            err = self.inst.get_average_position(<moab.EntityHandle*> arr.data, len(entity_handles), <double*> avg_position.data)
            check_error(err, exceptions)

        return avg_position

    def get_ord_average_position(self,
                             entity_handles,
                             exceptions = ()):
        cdef moab.ErrorCode err
        cdef moab.EntityHandle ms_handle
        cdef int i
        cdef np.ndarray[np.uint64_t, ndim=1] arr
        cdef np.ndarray avg_position
        cdef np.ndarray[np.double_t, ndim=2] avg_array
        if isinstance(entity_handles, Range):
          arr = entity_handles.get_array()
        elif isinstance(entity_handles, np.ndarray):
          arr = entity_handles
        else:
          arr = np.array(entity_handles, dtype = np.uint64)
        siz = arr.size
        avg_array=np.empty((siz, 3), dtype=np.double)
        for i in range(siz):
            avg_position = np.empty((3,),dtype='float64')
            err = self.inst.get_average_position(<moab.EntityHandle*> arr.data+i, 1, <double*> avg_position.data)
            check_error(err, exceptions)
            avg_array[i]=avg_position

        return avg_array

    def construct_aentities(self,
                            vertices,
                            exceptions = ()):
        """
        Generate all the AEntities bounding the vertices.

        Example
        -------
        root_set = mb.get_root_set()
        all_verts = mb.get_entities_by_dimension(root_set, 0)
        mtu.construct_aentities(all_verts)

        Parameters
        ----------
        vertices : Range or iterable of EntityHandles
            Vertices that will be used to generate the bounding AEntities
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
          None

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        """
        cdef moab.ErrorCode err
        cdef Range r

        r = vertices
        err = self.inst.construct_aentities(deref(r.inst))

        check_error(err, exceptions)
