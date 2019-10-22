"""Implements core functionality."""
from cython.operator cimport dereference as deref

cimport numpy as np
import numpy as np
import ctypes

from pymoab cimport moab
from pymoab cimport eh

from .tag cimport Tag, _tagArray
from .rng cimport Range
from .types import check_error, np_tag_type, validate_type, _convert_array, _eh_array, _eh_py_type
from . import types
from libcpp.vector cimport vector
from libcpp.string cimport string as std_string
from libc.stdlib cimport malloc

from collections import Iterable

cdef void* null = NULL
cdef class Core(object):

    def __cinit__(self):
        """ Constructor """
        self.inst = new moab.Core()

    def __del__(self):
        """ Destructor """
        del self.inst

    def impl_version(self):
        """MOAB implementation number as a float."""
        return self.inst.impl_version()

    def load_file(self, str fname, file_set = None, exceptions = ()):
        """
        Load or import a file.

        Load a MOAB-native file or import data from some other supported file format.

        Example
        -------
        mb = pymoab.core.Core()
        mb.load_file("/location/of/my_file.h5m")

        OR

        fs = mb.create_meshset()
        mb = pymoab.core.Core()
        mb.load_file("/location/of/my_file.h5m", fs)

        Parameters
        ----------
        fname : string
            The location of the file to read.
        file_set : EntityHandle (default None)
            If not None, this argument must be a valid
            entity set handle. All entities read from the file will be added to
            this set. File metadata will be added to tags on the set.
        exceptions : tuple
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
          None.

        Raises
        ------
          MOAB ErrorCode
              if a MOAB error occurs
          ValueError
              if a parameter is not of the correct type
        """
        cdef bytes cfname = fname.encode('UTF-8')
        cdef eh.EntityHandle fset
        cdef eh.EntityHandle* ptr
        if file_set != None:
            fset = file_set
            ptr = &fset
        else:
            ptr = NULL
        cdef const char * file_name = cfname
        cdef moab.ErrorCode err = self.inst.load_file(file_name,ptr)
        check_error(err, exceptions)

    def write_file(self, str fname, output_sets = None, output_tags = None, exceptions = ()):
        """
        Write or export a file.

        Write a MOAB-native file or export data to some other supported file format.

        Example
        -------
        mb.write_file("new_file.h5m")

        Parameters
        ----------
        fname : string
            the location of the file to write
        output_sets : EntityHandle or Range (default None)
            If not None, this argument must be a valid EntitySet handle or
            Range containing only EntitySets.
            When specified, the method will write any entities from the given
            meshsets.
        output_tags : List of PyMOAB Tags (default None)
            If not None, this argument must be a list of valid Tags.
            When specified, the write_file will not write any of the tag's data
            to the specified file. All tags handles will appear in the file, however.

        exceptions : tuple (default is empty tuple)
            tuple containing any error types that should
            be ignored (see pymoab.types module for more info)

        Returns
        -------
        None

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if a parameter is not of the correct type
        """
        cdef bytes cfname = fname.encode('UTF-8')
        cdef const char * file_name = cfname
        cdef moab.ErrorCode err
        cdef int num_tags = 0
        cdef _tagArray ta = _tagArray()

        if output_tags:
            assert isinstance(output_tags, Iterable), "Non-iterable output_tags argument."
            for tag in output_tags:
                assert isinstance(tag, Tag), "Non-tag type passed in output_tags."
            num_tags = len(output_tags)
            ta = _tagArray(output_tags)
        else:
            ta.ptr = NULL

        cdef Range r = Range()
        cdef np.ndarray[np.uint64_t, ndim=1] arr
        if output_sets:
          arr = _eh_array(output_sets)
          r = Range(arr)
          if not r.all_of_type(types.MBENTITYSET):
              raise IOError("Only EntitySets should be passed to write file.")
        if output_sets or output_tags:
            err = self.inst.write_file(
                file_name, <const char*> 0, <const char*> 0, deref(r.inst), ta.ptr, num_tags)
        else:
          err = self.inst.write_file(file_name)

        check_error(err, exceptions)

    def create_meshset(self, unsigned int options = 0x02, exceptions = ()):
        """
        Create a new mesh set.

        Create a new mesh set. Meshsets can store entities ordered or
        unordered. A set can include entities at most once (MESHSET_SET) or more
        than once (MESHSET_ORDERED). Meshsets can optionally track its members
        using adjacencies (MESHSET_TRACK_OWNER); if set, entities are deleted
        from tracking meshsets before being deleted. This adds data to mesh
        entities, which can be more memory intensive.

        Example
        -------
        # Create a standard meshset
        mb = pymoab.core.Core()
        new_meshset = mb.create_meshset()

        # Create a meshset which tracks entities
        mb = pymoab.core.Core()
        new_meshset = mb.create_meshset(pymoab.types.MESHSET_TRACK_OWNER)

        Parameters
        ----------
        options : MOAB EntitySet Property (default is MESHSET_SET)
            settings for the Meshset being created
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        MOAB Meshset (EntityHandle)

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        """
        cdef eh.EntityHandle ms_handle = 0
        cdef moab.EntitySetProperty es_property = <moab.EntitySetProperty> options
        cdef moab.ErrorCode err = self.inst.create_meshset(es_property, ms_handle)
        check_error(err, exceptions)
        return _eh_py_type(ms_handle)

    def add_entity(self, eh.EntityHandle ms_handle, entity, exceptions = ()):
        """
        Add an entity to the specified meshset.

        If meshset has MESHSET_TRACK_OWNER option set, the entity adjacencies
        are also added to the meshset.

        Example
        -------
        vertices # a iterable of MOAB vertex handles
        new_meshset = mb.create_meshset()
        mb.add_entity(new_meshset, entity)

        Parameters
        ----------
        ms_handle : EntityHandle
            EntityHandle of the Meshset the entities will be added to
        entity : EntityHandle
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
        cdef Range r = Range(entity)
        err = self.inst.add_entities(ms_handle, deref(r.inst))
        check_error(err, exceptions)

    def add_entities(self, eh.EntityHandle ms_handle, entities, exceptions = ()):
        """
        Add entities to the specified meshset. Entities can be provided either
        as a pymoab.rng.Range o bject or as an iterable of EntityHandles.

        If meshset has MESHSET_TRACK_OWNER option set, adjacencies are also
        added to the meshset.

        Example
        -------
        vertices # a iterable of MOAB vertex handles
        new_meshset = mb.create_meshset()
        mb.add_entities(new_meshset, entities)

        Parameters
        ----------
        ms_handle : EntityHandle
            EntityHandle of the Meshset the entities will be added to
        entities : Range or iterable of EntityHandles
            Entities that to add to the Meshset
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
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef Range r
        cdef np.ndarray[np.uint64_t, ndim=1] arr
        if isinstance(entities, Range):
           r = entities
           err = self.inst.add_entities(ms_handle, deref(r.inst))
        else:
           arr = _eh_array(entities)
           err = self.inst.add_entities(ms_handle, <eh.EntityHandle*> arr.data, len(entities))
        check_error(err, exceptions)

    def remove_entity(self, eh.EntityHandle ms_handle, entity, exceptions = ()):
        """
        Remove an entity from the specified meshset.

        If meshset has MESHSET_TRACK_OWNER option set, entity adjacencies
        are updated.

        Example
        -------
        new_meshset = mb.create_meshset()
        mb.remove_entity(new_meshset, entity)

        Parameters
        ----------
        ms_handle : EntityHandle
            EntityHandle of the Meshset the entities will be added to
        entity : EntityHandle
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
        cdef Range r = Range(entity)
        err = self.inst.remove_entities(ms_handle, deref(r.inst))
        check_error(err, exceptions)

    def remove_entities(self, eh.EntityHandle ms_handle, entities, exceptions = ()):
        """
        Remove entities from the specified meshset. Entities can be provided either
        as a pymoab.rng.Range object or as an iterable of EntityHandles.

        If meshset has MESHSET_TRACK_OWNER option set, adjacencies in entities
        in entities are updated.

        Example
        -------
        vertices # an iterable of MOAB vertex handles
        new_meshset = mb.create_meshset()
        mb.add_entities(new_meshset, entities)
        # keep only the first entity in this meshset
        mb.remove_entities(new_meshset, entities[1:])

        Parameters
        ----------
        ms_handle : EntityHandle
            EntityHandle of the Meshset the entities will be removed from
        entities : Range or iterable of EntityHandles
            Entities to remove from the Meshset
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
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef Range r
        cdef np.ndarray[np.uint64_t, ndim=1] arr
        if isinstance(entities, Range):
            r = entities
            err = self.inst.remove_entities(ms_handle, deref(r.inst))
        else:
            arr = _eh_array(entities)
            err = self.inst.remove_entities(ms_handle, <eh.EntityHandle*> arr.data, len(entities))
        check_error(err, exceptions)

    def delete_entity(self, entity, exceptions = ()):
        """
        Delete an entity from the database.

        If the entity is contained in any meshsets, it are removed
        from those meshsets which were created with MESHSET_TRACK_OWNER option.

        Example
        -------
        mb.delete_entity(entity)

        Parameters
        ----------
        entity : EntityHandle
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
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef Range r = Range(entity)
        err = self.inst.delete_entities(deref(r.inst))
        check_error(err, exceptions)

    def delete_entities(self, entities, exceptions = ()):
        """
        Delete entities from the database.

        If any of the entities are contained in any meshsets, they are removed
        from those meshsets which were created with MESHSET_TRACK_OWNER option.

        Example
        -------
        mb = pymoab.core.Core()
        # generate mesh using other Core functions #
        mb.delete_entities(entities)

        Parameters
        ----------
        entities : Range or iterable of EntityHandles
            Entities that to delete from the database
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
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef Range r
        cdef np.ndarray[np.uint64_t, ndim=1] arr
        if isinstance(entities, Range):
            r = entities
            err = self.inst.delete_entities(deref(r.inst))
        else:
            arr = _eh_array(entities)
            err = self.inst.delete_entities(<eh.EntityHandle*> arr.data, len(entities))
        check_error(err, exceptions)

    def create_vertices(self, coordinates, exceptions = ()):
        """
        Create vertices using the specified x,y,z coordinates.

        Example
        -------
        mb = pymoab.core.Core()
        #create two vertices
        coords = np.array((0, 0, 0, 1, 1, 1),dtype='float64')
        verts = mb.create_vertices(coords)

        OR

        mb = pymoab.core.Core()
        #create two vertices
        coords = [[0.0, 0.0, 0.0],[1.0, 1.0, 1.0]]
        verts = mb.create_vertices(coords)

        Parameters
        ----------
        coordinates : 1-D or 2-D iterable
            Coordinates can be provided as either a 1-D iterable of 3*n floats
            or as a 2-D array with dimensions (n,3) where n is the number of
            vertices being created.
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        MOAB Range of EntityHandles for the vertices

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef Range rng = Range()
        cdef np.ndarray coords_as_arr = np.asarray(coordinates)
        if coords_as_arr.ndim > 1:
            assert coords_as_arr.ndim == 2
            coords_as_arr = coords_as_arr.flatten()
        cdef np.ndarray[np.float64_t, ndim=1] coords_arr = _convert_array(coords_as_arr, (float, np.float64), np.float64)
        assert len(coords_arr)%3 == 0, "Incorrect number of coordinates provided."
        cdef moab.ErrorCode err = self.inst.create_vertices(<double *> coords_arr.data,
                                                            len(coords_arr)//3,
                                                            deref(rng.inst))
        check_error(err, exceptions)
        return rng

    def create_element(self, int entity_type, connectivity, exceptions = ()):
        """
        Create an elment of type, entity_type, using vertex EntityHandles in connectivity.

        Example
        -------
        mb = core.Core()
        # create some vertices
        coords = np.array((0,0,0,1,0,0,1,1,1),dtype='float64')
        verts = mb.create_vertices(coords)

        # create the triangle in the database
        tri_verts = np.array(((verts[0],verts[1],verts[2]),),dtype='uint64')
        tris = mb.create_element(types.MBTRI,tri_verts)

        OR

        tri_verts = [verts[0],verts[1],verts[2]]
        tris = mb.create_element(types.MBTRI,tri_verts)


        Parameters
        ----------
        entity_type : MOAB EntityType (see pymoab.types module)
            type of entity to create (MBTRI, MBQUAD, etc.)
        coordinates : iterable of EntityHandles
            1-D array-like iterable of vertex EntityHandles the element is to be
            created from
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        EntityHandle of element created

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef moab.EntityType typ = <moab.EntityType> entity_type
        cdef eh.EntityHandle handle = 0
        if isinstance(connectivity, Range):
            connectivity = list(connectivity)
        cdef np.ndarray[np.uint64_t, ndim=1] conn_arr = _eh_array(connectivity)
        cdef int nnodes = len(connectivity)
        cdef moab.ErrorCode err = self.inst.create_element(typ,
            <eh.EntityHandle*> conn_arr.data, nnodes, handle)
        check_error(err, exceptions)
        return _eh_py_type(handle)

    def create_elements(self, int entity_type, connectivity, exceptions = ()):
        """
        Create an elments of type, entity_type, using vertex EntityHandles in connectivity.

        Example
        -------
        mb = core.Core()
        # create some vertices for triangle 1
        coords1 = np.array((0,0,0,1,0,0,1,1,1),dtype='float64')
        verts1 = mb.create_vertices(coords)
        # create some more vertices for triangle 2
        coords2 = np.array((1,2,3,4,5,6,1,1,1),dtype='float64')
        verts2 = mb.create_vertices(coords)

        # create the triangles in the database
        tri_verts = [[verts1[0],verts1[1],verts1[2]],[verts2[0],verts2[1],verts2[2]]
        tris = mb.create_elements(types.MBTRI,tri_verts)

        OR

        tri_verts = [verts1,verts2]
        tris = mb.create_elements(types.MBTRI,tri_verts)

        Parameters
        ----------
        entity_type : MOAB EntityType (see pymoab.types module)
            type of entity to create (MBTRI, MBQUAD, MBHEX, etc.)
        coordinates : iterable of EntityHandles
            2-D array-like iterable of vertex EntityHandles the elements are to
            be created from. Each entry in the array should have an appropriate
            length for the element type being created (e.g. for MBTRI each entry
            has length 3).
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        EntityHandles of element created as a Range

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef int i
        cdef moab.ErrorCode err
        cdef moab.EntityType typ = <moab.EntityType> entity_type
        cdef np.ndarray connectivity_as_arr = np.asarray(connectivity)
        assert connectivity_as_arr.ndim == 2 #required for now
        cdef int nelems = connectivity_as_arr.shape[0]
        cdef int nnodes = connectivity_as_arr.shape[1]
        cdef np.ndarray[np.uint64_t, ndim=1] connectivity_i
        cdef np.ndarray[np.uint64_t, ndim=1] handles = np.empty(nelems, 'uint64')
        for i in range(nelems):
            connectivity_i = _eh_array(connectivity[i])
            err = self.inst.create_element(typ, <eh.EntityHandle*> connectivity_i.data,
                                           nnodes, deref((<eh.EntityHandle*> handles.data)+i))
            check_error(err, exceptions)
        return Range(handles)

    def tag_get_handle(self,
                       name,
                       size = None,
                       tag_type = None,
                       storage_type = None,
                       create_if_missing = False,
                       default_value = None,
                       exceptions = ()):
        """
        Retrieve and/or create tag handles for storing data on mesh elements, vertices,
        meshsets, etc.


        Example - creating a new tag handle
        -------
        # new MOAB core instance
        mb = core.Core()
        #
        tag_type = pymoab.types.MB_TYPE_INTEGER # define the tag's data type
        tag_size = 1 # the tag size (1 integer value)
        storage_type = pymoab.types.MB_TAG_DENSE # define the storage type
        tag_handle = mb.tag_get_handle("NewDataTag",
                                       tag_size,
                                       tag_type,
                                       storage_type,
                                       create_if_missing = True)

        Example - retrieving an existing tag handle
        -------
        # new MOAB core instance
        mb = core.Core()
        #
        tag_type = pymoab.types.MB_TYPE_INTEGER # define the tag's data type
        tag_size = 1 # the tag size (1 integer value)
        tag_handle = mb.tag_get_handle("NewDataTag",
                                       tag_size,
                                       tag_type)

        OR

        # find the tag by name rather than using the full signature (less robust)
        tag_handle = mb.tag_get_handle("NewDataTag")

        Parameters
        ----------
        name : string
            name of the tag
        size : int
            number of values the tag contains (or the number of characters in
            the case of an opaque tag)
        tag_type : MOAB DataType
            indicates the data type of the tag (MB_TYPE_DOUBLE, MB_TYPE_INTEGER,
            MB_TYPE_OPAQUE, etc.)
        storage_type : MOAB tag storage type (MB_TAG_DENSE, MB_TAG_SPARSE, etc.)
            in advanced use of the database, this flag controls how this tag's
            data is stored in memory. The two most common storage types are

            MB_TYPE_SPARSE -  sparse tags are stored as a list of (entity
                handle, tag value) tuples, one list per sparse tag, sorted by
                entity handle

            MB_TYPE_DENSE - Dense tag values are stored in arrays which match
                arrays of contiguous entity handles. Dense tags are more
                efficient in both storage and memory if large numbers of
                entities are assigned the same tag. Storage for a given dense
                tag is not allocated until a tag value is set on an entity;
                memory for a given dense tag is allocated for all entities in a
                given sequence at the same time.  Sparse: Sparse tags are stored
                as a list of (entity handle, tag value) tuples, one list per
                sparse tag, sorted by entity handle.

            MB_TYPE_BIT - Bit tags are stored similarly to dense tags, but with
                special handling to allow allocation in bit-size amounts per
                entity.
        create_if_missing : bool (default False)
            indicates to the database instance that the tag should be created if
            it cannot be found (requires that all arguments are provided to
            fully define the tag)
        default_value : default tag value (default None)
            valid_default value fo the tag based on the tag type and length
            specified in previous arguments. If no default value is specified,
            then the returned tag will have no default value in the MOAB
            instance.
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        MOAB TagHandle

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        """
        cdef bytes cname = str(name).encode('UTF-8')
        cdef const char* tag_name = cname
        cdef Tag tag = Tag()
        cdef moab.ErrorCode err
        cdef moab.DataType tt
        cdef int s
        cdef np.ndarray default_val_arr
        cdef const void* def_val_ptr = NULL
        incomplete_tag_specification = False

        # tag_type, size, and storage_type
        # should either be all None
        # or all non-None
        all_none = all(v == None for v in [size, tag_type, storage_type])
        all_non_none = all( v != None for v in [size, tag_type, storage_type])
        if not (all_none or all_non_none):
            raise ValueError("""
            Partial tag specification supplied. Please only provide tag name for
            a name-based tag retrieval or provide a tag name, size, type, and
            storage type for a more detailed specification of the tag to
            retrieve.
            """)

        # if a default value is provided, set ptr
        if default_value is not None:
            if tag_type is types.MB_TYPE_OPAQUE:
                default_val_arr = np.asarray((default_value,), dtype='S'+str(size))
            else:
                default_val_arr = np.asarray(default_value, dtype=np.dtype(np_tag_type(tag_type)))
            # validate default value data for tag type and length
            default_val_arr = validate_type(tag_type,size,default_val_arr)
            def_val_ptr = <const void*> default_val_arr.data

        if tag_type is None and size is None and storage_type is None:
            err = self.inst.tag_get_handle(tag_name, tag.inst)
        else:
            tt = tag_type
            s = size
            flags = storage_type|types.MB_TAG_CREAT if create_if_missing else storage_type
            err = self.inst.tag_get_handle(tag_name, s, tt, tag.inst, flags, def_val_ptr, NULL)

        check_error(err, exceptions)
        return tag

    def tag_set_data(self, Tag tag, entity_handles, data, exceptions = ()):
        """
        Set tag data for a set of MOAB entities. The data provided must be
        compatible with the tag's data type and be of an appropriate shape and
        size depending on the number of entity handles are provided.

        This function will ensure that the data is of the correct type and
        length based on the tag it is to be applied to. Data can be passed as a
        1-D iterable of appropriate length or as 2-D iterable with entries for
        each entity handle equal to the length of the tag.

        Example
        -------
        # create moab core instance
        mb = core.Core()
        # create some vertices
        coords = np.array((0,0,0,1,0,0,1,1,1),dtype='float64')
        verts = mb.create_vertices(coords)
        # create a new tag for data
        tag_length = 2
        tag_type = pymoab.types.MB_TYPE_DOUBLE
        data_tag = mb.tag_get_handle("Data",
                                     tag_length,
                                     tag_type,
                                     create_if_missing = True)
        # some sample data
        data = np.array([1.0,2.0,3.0,4.0,5.0,6.0], dtype = 'float')
        #use this function to tag vertices with the data
        mb.tag_set_data(data_tag, verts, data)

        OR

        # some sample data
        data = np.array([[1.0,2.0],[3.0,4.0],[5.0,6.0]], dtype = 'float')
        #use this function to tag vertices with the data
        mb.tag_set_data(data_tag, verts, data)

        OR

        ### pass data as a an iterable object ###
        # some sample data
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        #use this function to tag vertices with the data
        mb.tag_set_data(data_tag, verts, data)

        OR

        ### pass data as a nested iterable object ###
        ### (can be convenient for vector tags) ###
        # some sample data
        data = [[1.0, 2.0],[3.0, 4.0],[4.0, 5.0]]
        #use this function to tag vertices with the data
        mb.tag_set_data(data_tag, verts, data)

        OR

        ### tag a single vertex ###
        # get a single vertex
        vertex_to_tag = verts[0]
        data = [1.0, 2.0]
        #use this function to tag vertices with the data
        mb.tag_set_data(data_tag, vertex, data)

        Parameters
        ----------
        tag : MOAB TagHandle
            tag to which the data is applied
        entity_handles : iterable of MOAB EntityHandle's or single EntityHandle
            the EntityHandle(s) to tag the data on. This can be any iterable of
            EntityHandles or a single EntityHandle.
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)
        data : iterable of data to set for the tag
            This is a 1-D or 2-D iterable of data values to set for the tag and
            entities specified in the entity_handles parameter. If the data is
            1-D then it must be of size len(entity_handles)*tag_length. If the
            data is 2-D then it must be of size len(entity_handles) with each
            second dimension entry having a length equal to the tag's
            length. All entries in the data must be of the proper type for the
            tag or be able to be converted to that type.
        Returns
        -------
        None

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if a data entry is not of the correct type or cannot be converted to
            the correct type
        """
        cdef moab.ErrorCode err
        cdef np.ndarray ehs
        cdef moab.DataType tag_type = moab.MB_MAX_DATA_TYPE
        # create a numpy array for the entity handles to be tagged
        if isinstance(entity_handles, _eh_py_type):
            ehs = _eh_array([entity_handles,])
        else:
            ehs = _eh_array(entity_handles)
        err = self.inst.tag_get_data_type(tag.inst, tag_type);
        check_error(err, ())
        cdef int length = 0
        err = self.inst.tag_get_length(tag.inst,length);
        check_error(err,())
        cdef np.ndarray data_arr = np.asarray(data)
        # protect against shallow copy, delayed evaluation problem
        data_arr = np.copy(data_arr)
        #if the data array is not flat it must be dimension 2 and have
        #as many entries as entity handles provided
        if data_arr.ndim > 1:
            assert data_arr.ndim == 2
            assert data_arr.shape[0] == ehs.size
            #each entry must be equal to the tag length as well
            for entry in data_arr:
                len(entry) == length
            #if all of this is true, then flatten the array and continue
            data_arr = data_arr.flatten()
        error_str = "Incorrect data length"
        if types.MB_TYPE_OPAQUE == tag_type:
            assert data_arr.size == ehs.size, error_str
        else:
            assert data_arr.size == ehs.size*length, error_str
        data_arr = validate_type(tag_type,length,data_arr)
        err = self.inst.tag_set_data(tag.inst, <eh.EntityHandle*> ehs.data, ehs.size, <const void*> data_arr.data)
        check_error(err, exceptions)

    def tag_get_data(self, Tag tag, entity_handles, flat = False, exceptions = ()):
        """
        Retrieve tag data for a set of entities. Data will be returned as a 2-D
        array with number of entries equal to the number of EntityHandles passed
        to the function by default. Each entry in the array will be equal to the
        tag's length. A flat array of data can also be returned if specified by
        setting the 'flat' parameter to True.

        Example
        -------
        mb = core.Core()
        coords = np.array((0,0,0,1,0,0,1,1,1),dtype='float64')
        verts = mb.create_vertices(coords)
        # create a new tag for data
        tag_length = 2
        tag_type = pymoab.types.MB_TYPE_DOUBLE
        data_tag = mb.tag_get_handle("Data",
                                     tag_length,
                                     tag_type,
                                     create_if_missing = True)
        # some sample data
        data = np.array([1.0,2.0,3.0,4.0,5.0,6.0], dtype = 'float')
        # use this function to tag vertices with the data
        mb.tag_set_data(data_tag, verts, data)

        # now retrieve this tag's data for the vertices from the instance
        data = mb.tag_get_data(data_tag, verts) # returns 2-D array

        OR

        # now retrieve this tag's data for the vertices from the instance
        # returns 1-D array of len(verts)*tag_length
        data = mb.tag_get_data(data_tag, verts, flat=True)

        Parameters
        ----------
        tag : MOAB TagHandle
            the tag for which data is to be retrieved
        entity_handles : iterable of MOAB EntityHandles or a single EntityHandle
            the EntityHandle(s) to retrieve data for.
        flat : bool (default is False)
            Indicates the structure in which the data is returned. If False, the
            array is returned as a 2-D numpy array of len(entity_handles) with
            each second dimension entry equal to the length of the tag data. If
            True, the data is returned as a 1-D numpy array of length
            len(entity_handles)*tag_length.
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        Numpy array of data with dtype matching that of the tag's type

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef np.ndarray ehs
        cdef moab.DataType tag_type = moab.MB_MAX_DATA_TYPE
        # create a numpy array for the entity handles to be tagged
        if isinstance(entity_handles, _eh_py_type):
            ehs = _eh_array([entity_handles,])
        else:
            ehs = _eh_array(entity_handles)
        # get the tag type and length for validation
        err = self.inst.tag_get_data_type(tag.inst, tag_type);
        check_error(err,())
        cdef int length = 0
        err = self.inst.tag_get_length(tag.inst,length);
        check_error(err,())
        #create array to hold data
        cdef np.ndarray data
        if tag_type is types.MB_TYPE_OPAQUE:
            data = np.empty((ehs.size,),dtype='S'+str(length))
        else:
            data = np.empty((length*ehs.size,), dtype=np.dtype(np_tag_type(tag_type)))
        err = self.inst.tag_get_data(tag.inst, <eh.EntityHandle*> ehs.data, ehs.size, <void*> data.data)
        check_error(err, exceptions)
        # return data as user specifies
        if tag_type is types.MB_TYPE_OPAQUE:
            data = data.astype('str')
        if flat:
            return data
        else:
            entry_len = 1 if tag_type == types.MB_TYPE_OPAQUE else length
            return data.reshape((ehs.size,entry_len))

    def tag_delete_data(self, Tag tag, entity_handles, exceptions = ()):
        """
        Delete the data of a tag on a set of EntityHandle's. Only sparse tag
        data are deleted with this method; dense tags are deleted by deleting
        the tag itself using tag_delete.

        Example
        -------
        # delete data associated with the iterable entities from the tag data_tag
        data = mb.tag_delete_data(data_tag, entities)

        Parameters
        ----------
        tag : MOAB TagHandle
            the tag from which data is to be deleted
        entity_handles : iterable of MOAB EntityHandles or a single EntityHandle
            the EntityHandle(s) to delete data from. This can be any iterable of
            EntityHandles or a single EntityHandle.

        Returns
        -------
        None

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef np.ndarray ehs
        # create a numpy array for the entity handles to be tagged
        if isinstance(entity_handles, _eh_py_type):
            ehs = _eh_array([entity_handles,])
        else:
            ehs = _eh_array(entity_handles)
        # Delete the data
        err = self.inst.tag_delete_data(tag.inst, <eh.EntityHandle*> ehs.data, ehs.size)
        check_error(err, exceptions)

    def tag_delete(self, Tag tag, exceptions = ()):
        """
        Removes the tag from the database and deletes all of its associated data.

        Example
        -------
        # delete a tag that was previously created
        data = mb.tag_delete(data_tag)

        Parameters
        ----------
        tag : MOAB TagHandle
            the tag from which data is to be deleted

        Returns
        -------
        None

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        """
        cdef moab.ErrorCode err
        # Delete the tag
        err = self.inst.tag_delete(tag.inst)
        check_error(err, exceptions)

    def get_adjacencies(self, entity_handles, int to_dim, bint create_if_missing = False, int op_type = types.INTERSECT, exceptions = ()):
        """
        Get the adjacencies associated with a range of entities to entities of a specified dimension.

        Example
        -------
        mb = core.Core()
        coords = np.array((0,0,0,1,0,0,1,1,1),dtype='float64')
        verts = mb.create_vertices(coords)
        #create elements
        verts = np.array(((verts[0],verts[1],verts[2]),),dtype='uint64')
        tris = mb.create_elements(types.MBTRI,verts)
        #get the adjacencies of the triangle of dim 1 (should return the vertices)
        adjacent_verts = mb.get_adjacencies(tris, 0, False)
        #now get the edges and ask MOAB to create them for us
        adjacent_edges = mb.get_adjacencies(tris, 1, True)

        Parameters
        ----------
        entity_handles : iterable of MOAB EntityHandles or a single EntityHandle
            the EntityHandle(s) to get the adjacencies of. This can be any
            iterable of EntityHandles or a single EntityHandle.
        to_dim : integer
            value indicating the dimension of the entities to return
        create_if_missing : bool (default is false)
            this parameter indicates that any adjacencies that do not exist
            should be created. For instance, in the example above, the second
            call to get_adjacencies will create the edges of the triangle which
            did not previously exist in the mesh database.
        op_type : int (default is 0 or INTERSECT)
            this value indicates how overlapping adjacencies should be handled
            INTERSECT - will return the intersection of the adjacencies between mesh elements
            UNION - will return the union of all adjacencies for the elements provided to the function
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        Range of MOAB EntityHAndles

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if an EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef Range r
        cdef Range adj = Range()
        r = Range(entity_handles)
        err = self.inst.get_adjacencies(deref(r.inst), to_dim, create_if_missing, deref(adj.inst), op_type)
        check_error(err, exceptions)
        return adj

    def get_ord_adjacencies(self, from_ent, int to_dim, Tag tag_handle, bint create_if_missing = False, int op_type = types.INTERSECT, exceptions = ()):

        cdef moab.ErrorCode err
        cdef moab.EntityHandle ms_handle
        cdef Range r
        cdef Range adjs = Range()
        cdef Range inR = Range()
        cdef int i
        cdef int j
        cdef int sizj
        cdef int siz
        cdef int idx_count = 0
        cdef int npinput = 0
        cdef np.ndarray[dtype = np.uint64_t, ndim = 1] inputArray
        cdef vector[eh.EntityHandle] rangeList
        cdef np.ndarray[np.int32_t] tag_array
        if isinstance(from_ent, Range):
            r = from_ent
        elif isinstance(from_ent, np.ndarray):
            inputArray = from_ent
            npinput=1
            siz = inputArray.size
        else:
            r = Range(from_ent)
        if not npinput:
          siz = r.size()
        cdef np.ndarray[np.int32_t, ndim = 1] idx_array = np.empty(siz, dtype = np.int32)
        for i in range(siz):
          if npinput:
            inR.insert(inputArray[i])
          else:
            inR.insert(r[i])
          err = self.inst.get_adjacencies(deref(inR.inst), to_dim, create_if_missing, deref(adjs.inst), op_type)
          inR.pop_front()
          check_error(err, exceptions)
          sizj = adjs.size()
          tag_array = self.tag_get_data(tag_handle, adjs, flat=True)
          for j in range(sizj):
            rangeList.push_back(tag_array[j])
          idx_count = idx_count + sizj
          idx_array[i] = idx_count
          adjs.clear()
        return np.delete(np.array(np.split(np.array(rangeList, dtype = np.int64), idx_array)), -1)

    def type_from_handle(self, entity_handle):
        """
        Returns the type (MBVERTEX, MBQUAD, etc.) of an EntityHandle

        Example
        -------
        mb = core.Core()
        coords = np.array((0,0,0,1,0,0,1,1,1),dtype='float64')
        verts = mb.create_vertices(coords)
        #create elements
        verts = np.array(((verts[0],verts[1],verts[2]),),dtype='uint64')
        tris = mb.create_elements(types.MBTRI,verts)
        entity_type = mb.type_from_handle(tris[0])

        Parameters
        ----------
        entity_handle : a MOAB EntityHandle

        Returns
        -------
        An integer representing the EntityType

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        """
        cdef moab.EntityType t
        t = self.inst.type_from_handle(<unsigned long> entity_handle)
        return t

    def get_child_meshsets(self, meshset_handle, num_hops = 1, exceptions = ()):
        """
        Retrieves the child meshsets from a parent meshset.

        Example
        -------
        mb = core.Core()
        parent_set = mb.create_meshset()
        child_set = mb.create_meshset()
        mb.add_parent_meshset(child_set, parent_set)
        child_sets = mb.get_child_meshsets(parent_set)

        Parameters
        ----------
        meshset_handle : MOAB EntityHandle
            handle of the parent meshset
        num_hops : integer (default is 1)
            the number of generations to traverse when collecting child
            meshsets.  If this value is zero, then the maximum number of
            generations will be traversed.
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        Range of MOAB EntityHandles

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if the Meshset EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef Range r = Range()
        err = self.inst.get_child_meshsets(<unsigned long> meshset_handle, deref(r.inst), num_hops)
        check_error(err, exceptions)
        return r

    def get_parent_meshsets(self, meshset_handle, num_hops = 1, exceptions = ()):
        """
        Retrieves the parent meshsets from a child meshset.

        Example
        -------
        mb = core.Core()
        parent_set = mb.create_meshset()
        child_set = mb.create_meshset()
        mb.add_parent_meshset(child_set, parent_set)
        parent_sets = mb.get_parent_meshsets(child_set)

        Parameters
        ----------
        meshset_handle : MOAB EntityHandle
            handle of the child meshset
        num_hops : integer (default is 1)
            the number of generations to traverse when collecting
            meshsets.  If this value is zero, then the maximum number of
            generations will be traversed.
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        Range of MOAB EntityHandles

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if the Meshset EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef Range r = Range()
        err = self.inst.get_parent_meshsets(<unsigned long> meshset_handle, deref(r.inst), 0)
        check_error(err, exceptions)
        return r

    def add_parent_meshset(self, child_meshset, parent_meshset, exceptions = ()):
        """
        Add a parent meshset to a meshset.

        Example
        -------
        mb = core.Core()
        parent_set = mb.create_meshset()
        child_set = mb.create_meshset()
        mb.add_parent_meshset(child_set, parent_set)

        Parameters
        ----------
        child_meshset : MOAB EntityHandle
            handle of the child meshset
        parent_meshset : MOAB EntityHandle
            handle of the parent meshset
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
        ValueError
            if the Meshset EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        err = self.inst.add_parent_meshset(<unsigned long> child_meshset, <unsigned long> parent_meshset)
        check_error(err, exceptions)

    def add_child_meshset(self, parent_meshset, child_meshset, exceptions = ()):
        """
        Add a child meshset to a meshset.

        Example
        -------
        mb = core.Core()
        parent_set = mb.create_meshset()
        child_set = mb.create_meshset()
        mb.add_child_meshset(parent_set, child_set)

        Parameters
        ----------
        parent_meshset : MOAB EntityHandle
            handle of the parent meshset
        child_meshset : MOAB EntityHandle
            handle of the child meshset
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
        ValueError
            if the Meshset EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        err = self.inst.add_child_meshset(<unsigned long> parent_meshset, <unsigned long> child_meshset)
        check_error(err, exceptions)

    def add_parent_child(self, parent_meshset, child_meshset, exceptions = ()):
        """
        Add a parent-child link between two meshsets.

        Example
        -------
        mb = core.Core()
        parent_set = mb.create_meshset()
        child_set = mb.create_meshset()
        mb.add_parent_child(parent_set, child_set)

        Parameters
        ----------
        parent_meshset : MOAB EntityHandle
            handle of the parent meshset
        child_meshset : MOAB EntityHandle
            handle of the child meshset
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
        ValueError
            if the Meshset EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        err = self.inst.add_parent_child(<unsigned long> parent_meshset, <unsigned long> child_meshset)
        check_error(err, exceptions)

    def get_root_set(self):
        """
        Return the entity meshset representing the whole mesh.

        Returns
        -------
        MOAB EntityHandle
        """
        return _eh_py_type(0)

    def get_connectivity(self, entity_handles, exceptions = ()):
        """
        Returns the vertex handles which make up the mesh entities passed in via
        the entity_handles argument.

        Example
        -------
        # new PyMOAB instance
        mb = core.Core()
        #create some vertices
        coords = np.array((0,0,0,1,0,0,1,1,1),dtype='float64')
        verts = mb.create_vertices(coords)
        #create a triangle
        verts = np.array(((verts[0],verts[1],verts[2]),),dtype='uint64')
        tris = mb.create_elements(types.MBTRI,verts)
        # retrieve the vertex handles that make up the triangle
        conn = mb.get_connectivity(tris[0])


        Parameters
        ----------
        entity_handles : iterable of MOAB EntityHandles or a single EntityHandle
            to retrieve the vertex connectivity of. Note that these
            EntityHandles should represent mesh entities (MBEDGE,
            MBTRI, MBQUAD, etc.) rather than an EntitySet.
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        Numpy array of vertex EntityHandles

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if an EntityHandle is not of the correct datatype
        """
        cdef moab.ErrorCode err
        cdef np.ndarray[eh.EntityHandle] ehs
        if isinstance(entity_handles, _eh_py_type):
            ehs = _eh_array([entity_handles,])
        else:
            ehs = _eh_array(entity_handles)
        cdef vector[eh.EntityHandle] ehs_out
        cdef eh.EntityHandle* eh_ptr
        cdef int num_ents = 0
        err = self.inst.get_connectivity(<eh.EntityHandle*> ehs.data, ehs.size, ehs_out)
        check_error(err, exceptions)

        return np.asarray(ehs_out, dtype = np.uint64)

    def get_ord_connectivity(self, entity_handles, Tag tag_handle = None, bint tag_opt = True, exceptions = ()):
        """
        Returns the vertex handles which make up the mesh entities passed in via
        the entity_handles argument.

        Example
        -------
        # new PyMOAB instance
        mb = core.Core()
        #create some vertices
        coords = np.array((0,0,0,1,0,0,1,1,1),dtype='float64')
        verts = mb.create_vertices(coords)
        #create a triangle
        verts = np.array(((verts[0],verts[1],verts[2]),),dtype='uint64')
        tris = mb.create_elements(types.MBTRI,verts)
        # retrieve the vertex handles that make up the triangle
        conn = mb.get_connectivity(tris[0])


        Parameters
        ----------
        entity_handles : iterable of MOAB EntityHandles or a single EntityHandle
            to retrieve the vertex connectivity of. Note that these
            EntityHandles should represent mesh entities (MBEDGE,
            MBTRI, MBQUAD, etc.) rather than an EntitySet.
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        Numpy array of vertex EntityHandles

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if an EntityHandle is not of the correct datatype
        """
        cdef moab.ErrorCode err
        cdef np.ndarray[eh.EntityHandle] ehs
        cdef np.ndarray[np.int32_t] tag_array
        if isinstance(entity_handles, _eh_py_type):
            ehs = _eh_array([entity_handles,])
        elif isinstance(entity_handles, np.ndarray):
            ehs = entity_handles
        else:
            ehs = _eh_array(entity_handles)
        cdef vector[eh.EntityHandle] ehs_out
        cdef eh.EntityHandle* eh_ptr
        cdef int num_ents = 0
        cdef int idx_count = 0
        cdef int typej
        cdef int siz = ehs.size
        cdef np.ndarray[dtype = np.int32_t, ndim = 1] idx_array = np.empty(siz, dtype = np.int32)
        cdef np.ndarray[dtype = np.uint8_t] sizenum = np.array([1,2,3,4,0,4,5,0,0,8,0], dtype = np.uint8)
        err = self.inst.get_connectivity(<eh.EntityHandle*> ehs.data, ehs.size, ehs_out)
        check_error(err, exceptions)
        cdef int i
        for i in range(siz):
          typej = self.inst.type_from_handle(<unsigned long> ehs[i])
          idx_count = idx_count + sizenum[typej]
          idx_array[i] = idx_count
        if not tag_opt:
          return np.delete(np.array(np.split(np.array(ehs_out, dtype = np.uint64), idx_array)), -1)
        else:
          tag_array = self.tag_get_data(tag_handle, np.array(ehs_out, dtype = np.uint64), flat=True)
        return np.delete(np.array(np.split(tag_array.astype(np.int64), idx_array)), -1)

    def get_coords(self, entities, exceptions = ()):
        """
        Returns the xyz coordinate information for a set of vertices.

        Example
        -------
        mb = core.Core()
        verts # list of vertex EntityHandles
        ret_coords = mb.get_coords(verts)

        Parameters
        ----------
        entities : iterable of MOAB EntityHandles or a single EntityHandle
            the EntityHandle(s) to get the adjacencies of. This can be any
            iterable of EntityHandles or a single EntityHandle.
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Returns
        -------
        Returns coordinates as a 1-D Numpy array of xyz values

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if the Meshset EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef Range r
        cdef np.ndarray[np.uint64_t, ndim=1] arr
        cdef np.ndarray coords
        if isinstance(entities, _eh_py_type):
            entities = Range(entities)
        if isinstance(entities, Range):
            r = entities
            coords = np.empty((3*r.size(),),dtype='float64')
            err = self.inst.get_coords(deref(r.inst), <double*> coords.data)
        else:
            if isinstance(entities, np.ndarray):
              arr = entities
            else:
              arr = _eh_array(entities)
            coords = np.empty((3*len(arr),),dtype='float64')
            err = self.inst.get_coords(<eh.EntityHandle*> arr.data, len(entities), <double*> coords.data)
        check_error(err, exceptions)
        return coords


    def set_coords(self, entities, np.ndarray coords, exceptions = ()):
        """
        Sets the xyz coordinate information for a set of vertices.

        Example
        -------
        mb = core.Core()
        verts # list of vertex EntityHandles
        coords = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
        ret_coords = mb.set_coords(verts, coords)

        Parameters
        ----------
        entities : iterable of MOAB EntityHandles or a single EntityHandle
            the EntityHandle(s) to get the adjacencies of. This can be any
            iterable of EntityHandles or a single EntityHandle.
        coords : coordinate positions (float)
            vertex coordinates to be set in Core object
        exceptions : tuple (default is empty tuple)
            A tuple containing any error types that should
            be ignored. (see pymoab.types module for more info)

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if the Meshset EntityHandle is not of the correct type
        """
        cdef moab.ErrorCode err
        cdef Range r
        cdef np.ndarray[np.uint64_t, ndim=1] arr
        if isinstance(entities, _eh_py_type):
            entities = Range(entities)
        if isinstance(entities, Range):
            r = entities
            if 3*r.size() != len(coords):
              check_error(moab.MB_INVALID_SIZE, exceptions)
            err = self.inst.set_coords(deref(r.inst), <const double*> coords.data)
        else:
            arr = _eh_array(entities)
            if 3*len(arr) != len(coords):
              check_error(moab.MB_INVALID_SIZE, exceptions)
            err = self.inst.set_coords(<eh.EntityHandle*> arr.data, len(entities), <const double*> coords.data)
        check_error(err, exceptions)

    def get_entities_by_type(self, meshset, entity_type, bint recur = False, bint as_list = False, exceptions = ()):
        """
        Retrieves all entities of a given entity type in the database or meshset

        Parameters
        ----------
        meshset     : MOAB EntityHandle
                      meshset whose entities are being queried
        entity_type : MOAB EntityType
                      type of the entities desired (MBVERTEX, MBTRI, etc.)
        recur       : bool (default is False)
                      if True, meshsets containing meshsets are queried recusively. The
                      contenst of these meshsets are returned, but not the meshsets
                      themselves.
        as_list        : return entities in a list (preserves ordering if necessary, but
                      consumes more memory)
        Returns
        -------
        MOAB Range of EntityHandles

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if the Meshset EntityHandle is not of the correct type or
            if the EntityType provided is not valid
        """
        cdef moab.ErrorCode err
        cdef Range entities = Range()
        cdef moab.EntityType typ = entity_type
        cdef vector[eh.EntityHandle] hvec
        if as_list:
            err = self.inst.get_entities_by_type(<unsigned long> meshset,
                                                 typ,
                                                 hvec,
                                                 recur)
            check_error(err, exceptions)
            return hvec

        else:
            err = self.inst.get_entities_by_type(<unsigned long> meshset,
                                                 typ,
                                                 deref(entities.inst),
                                                 recur)
            check_error(err, exceptions)
            return entities

    def get_entities_by_type_and_tag(self,
                                     meshset,
                                     entity_type,
                                     tags,
                                     values,
                                     int condition = types.INTERSECT,
                                     bint recur = False,
                                     exceptions = ()):
        """
        Retrieve entities of a given EntityType in the database which also have
        the provided tag and (optionally) the value specified.

        Values are expected to be provided in one or two-dimensional arrays in
        which each entry in the highest array dimension represents the value (or
        set of values for a vector tag) to be matched for corresponding tag in
        the "tags" parameter. If an entry is filled with None values, then any
        entity with that tag will be returned.

        NOTE: This function currently only works for a single tag and set of values.

        Example
        -------
        mb = core.Core()
        rs = mb.get_root_set()
        entities = mb.get_entities_by_type(rs, types.MBVERTEX)

        Parameters
        ----------
        meshset : MOAB EntityHandle (defualt is the database root set)
            meshset whose entities are being queried.
        entity_type : MOAB EntityType
            type of the entities desired (MBVERTEX, MBTRI, etc.)
        tags : iterable of MOAB TagHandles or single TagHandle
            an iterable of MOAB TagHandles. Only entities with these tags will
            be returned.
        values : iterable of tag values
            an iterable data structure of the tag values to be matched for
            entities returned by the call (see general description for details
            on the composition of this data structure)
        recur : bool (default is False)
            if True, meshsets containing meshsets are queried recusively. The
            contenst of these meshsets are returned, but not the meshsets
            themselves.

        Returns
        -------
        MOAB Range of EntityHandles

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if the Meshset EntityHandle is not of the correct type or if the
            EntityType provided is not valid or if the tag data to be matched is
            not constructed properly
        """
        cdef np.ndarray vals = np.asarray(values, dtype='O')
        assert isinstance(tags, Tag) or len(tags) == 1 , "Only single-tag queries are currently supported."
        # overall dimension of the numpy array should be 2
        # one array for each tag passed to the function
        # assert vals.ndim == 2 (was for support of multiple tags)
        #ensure that the correct number of arrays exist
        cdef int num_tags
        cdef Tag single_tag
        if isinstance(tags, Tag):
            num_tags = 1
            single_tag = tags
        elif len(tags) == 1:
            num_tags = 1
            single_tag = tags[0]
        else:
            num_tags = len(tags)
        if 1 != num_tags:
            assert vals.ndim == 2
            assert num_tags == vals.shape[0]
        assert vals.ndim <= 2
        #allocate memory for an appropriately sized void** array
        cdef void** arr = <void**> malloc(num_tags*sizeof(void*))
        #some variables to help in setting up the void array
        cdef moab.ErrorCode err
        cdef moab.DataType this_tag_type = moab.MB_MAX_DATA_TYPE
        cdef int this_tag_length = 0
        cdef bytes val_str
        cdef np.ndarray this_data

        # if this is a single tag and the values array is 1-D
        # then validate and call function
        if (num_tags == 1) and (vals.ndim == 1):
            this_data = self._validate_data_for_tag(single_tag,vals)
            arr[0] = <void*> this_data.data if this_data is not None else NULL
        elif vals.ndim == 2:
            # assign values to void** array
            for i in range(num_tags):
                this_data = self._validate_data_for_tag(tags[i],vals[i])
                #set the array value
                arr[i] = <void*> this_data.data if this_data is not None else NULL

        #create tag array to pass to function
        cdef _tagArray ta = _tagArray(tags)
        #convert type to tag type
        cdef moab.EntityType typ = entity_type
        #a range to hold returned entities
        cdef Range ents = Range()
        #here goes nothing
        err = self.inst.get_entities_by_type_and_tag(<unsigned long> meshset,
                                                     typ,
                                                     ta.ptr,
                                                     <const void**> arr,
                                                     num_tags,
                                                     deref(ents.inst),
                                                     condition,
                                                     recur)
        check_error(err, exceptions)
        # return entities found in the MOAB function call
        return ents


    def get_entities_by_handle(self, meshset, bint recur = False, bint as_list = False, exceptions = ()):
        """
        Retrieves all entities in the given meshset.

        Parameters
        ----------
        meshset : MOAB EntityHandle
            meshset whose entities are being queried
        recur : bool (default is False)
            if True, meshsets containing meshsets are queried recusively. The
            contenst of these meshsets are returned, but not the meshsets
            themselves.
        as_list        : return entities in a list (preserves ordering if necessary, but
                      consumes more memory)
        Returns
        -------
        MOAB Range of EntityHandles

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if the Meshset EntityHandle is not of the correct type or
            if the EntityType provided is not valid
        """
        cdef moab.ErrorCode err
        cdef Range ents = Range()
        cdef vector[eh.EntityHandle] hvec
        if as_list:
            err = self.inst.get_entities_by_handle(<unsigned long> meshset, hvec, recur)
            check_error(err, exceptions)
            return hvec
        else:
            err = self.inst.get_entities_by_handle(<unsigned long> meshset, deref(ents.inst), recur)
            check_error(err, exceptions)
            return ents

    def get_entities_by_dimension(self, meshset, int dimension, bint recur = False, bint as_list = False, exceptions = ()):
        """
        Retrieves all entities of a given topological dimension in the database or meshset

        Parameters
        ----------
        meshset   : MOAB EntityHandle
                    meshset whose entities are being queried
        dimension : integer
                    topological dimension of the entities desired
        recur     : bool (default is False)
                    if True, meshsets containing meshsets are queried recusively. The
                    contenst of these meshsets are returned, but not the meshsets
                    themselves.
        as_list   : return entities in a list (preserves ordering if necessary, but
                    consumes more memory)
        Returns
        -------
        MOAB Range of EntityHandles

        Raises
        ------
        MOAB ErrorCode
            if a MOAB error occurs
        ValueError
            if the Meshset EntityHandle is not of the correct type or
            if the EntityType provided is not valid
        """
        cdef moab.ErrorCode err
        cdef Range ents = Range()
        cdef vector[eh.EntityHandle] hvec
        if as_list:
            err = self.inst.get_entities_by_dimension(<unsigned long> meshset, dimension, hvec, recur)
            check_error(err, exceptions)
            return hvec
        else:
            err = self.inst.get_entities_by_dimension(<unsigned long> meshset, dimension, deref(ents.inst), recur)
            check_error(err, exceptions)
            return ents

    def delete_mesh(self):
        """Deletes all mesh entities from the database"""
        self.inst.delete_mesh()

    def _validate_data_for_tag(self, Tag tag, data_arr):
        """Internal method for validating tag data."""

        cdef moab.ErrorCode err
        cdef moab.DataType tag_type = moab.MB_MAX_DATA_TYPE
        cdef int tag_length = 0
        # make sure we're dealing with a 1-D array at this point
        assert data_arr.ndim == 1
        # if None is passed, set pointer to NULL and continue
        if data_arr[0] == None:
            return None
        else:
            # otherwise get the tag type
            err = self.inst.tag_get_data_type(tag.inst, tag_type)
            check_error(err)
            #and length
            err = self.inst.tag_get_length(tag.inst,tag_length);
            check_error(err)
            #check that the data_arr for this tag is the correct length
            if tag_type == moab.MB_TYPE_OPAQUE:
                #if this is an opaque tag, there should only be one
                #string entry in the array
                assert data_arr.size == 1
            else:
                assert data_arr.size == tag_length
        #validate the array type and convert the dtype if necessary
        data_arr = validate_type(tag_type, tag_length, data_arr)
        return data_arr

    def tag_get_default_value(self, Tag tag, exceptions = () ):
        """
        Returns the default value for a tag as a 1-D numpy array

        Parameters
        ----------
        tag : MOAB tag

        Returns
        -------
        NumPy 1-D array

        Raises
        ------
        MOAB ErrorCode
            if an internal MOAB error occurs
        """
        # get the tag type and length for validation
        cdef moab.DataType tag_type = moab.MB_MAX_DATA_TYPE
        err = self.inst.tag_get_data_type(tag.inst, tag_type);
        check_error(err,())
        cdef int length = 0
        err = self.inst.tag_get_length(tag.inst,length);
        check_error(err,())
        #create array to hold data
        cdef np.ndarray data
        if tag_type is types.MB_TYPE_OPAQUE:
            data = np.empty((1,),dtype='S'+str(length))
        else:
            data = np.empty((length,),dtype=np.dtype(np_tag_type(tag_type)))
        err = self.inst.tag_get_default_value(tag.inst, <void*> data.data)
        check_error(err, exceptions)
        return data

    def tag_get_length(self, Tag tag, exceptions = () ):
        """
        Returns an integer representing the tag's length

        Parameters
        ----------
        tag : MOAB tag

        Returns
        -------
        int : the tag length

        Raises
        ------
        MOAB ErrorCode
            if an internal MOAB error occurs
        """
        cdef int length = 0
        err = self.inst.tag_get_length(tag.inst, length)
        check_error(err, exceptions)
        return length

    def tag_get_tags_on_entity(self, entity):
        """
        Retrieves all tags for a given EntityHandle

        Parameters
        ----------
        entity : MOAB EntityHandle

        Returns
        -------
        A list of tags on that entity

        Raises
        ------
        MOAB ErrorCode
            if an internal MOAB error occurs
        """
        cdef vector[moab.TagInfo*] tags
        err = self.inst.tag_get_tags_on_entity(entity, tags)
        tag_list = []
        for tag in tags:
            t = Tag()
            t.inst = tag
            tag_list.append(t)
        return tag_list
