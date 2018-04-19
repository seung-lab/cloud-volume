#ifndef __COMPRESSO_H__
#define __COMPRESSO_H__

#include <unordered_map>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <ctime>


namespace Compresso {
    // function definitions
    template<typename Type> unsigned char *Compress(Type *data, long res[3], long steps[3], long *nentries = NULL);
    template<typename Type> Type *Decompress(unsigned char *compressed_data, long *res = NULL);

    // dimension constants
    static const short RN_X = 2;
    static const short RN_Y = 1;
    static const short RN_Z = 0;

    // global variables
    static long row_size = -1;
    static long sheet_size = -1;
    static long grid_size = -1;

    // internal helper function
    static inline long IndicesToIndex(long ix, long iy, long iz) {
        return iz * sheet_size + iy * row_size + ix;
    };



    ///////////////////////////////////////////////////
    //// UNION-FIND CLASS FOR CONNECTED COMPONENTS ////
    ///////////////////////////////////////////////////

    class UnionFindElement {
    public:
        // constructor
        UnionFindElement(unsigned long label) :
        label(label),
        parent(this),
        rank(0)
        {}

    public:
        // instance variables
        unsigned long label;
        UnionFindElement *parent;
        int rank;
    };

    UnionFindElement *
    Find(UnionFindElement *x)
    {
        if (x->parent != x) x->parent = Find(x->parent);
        return x->parent;
    };

    void
    Union(UnionFindElement *x, UnionFindElement *y)
    {
        UnionFindElement *xroot = Find(x);
        UnionFindElement *yroot = Find(y);

        // root already the same
        if (xroot == yroot) return;

        // merge the two roots
        if (xroot->rank < yroot->rank) xroot->parent = yroot;
        else if (xroot->rank > yroot->rank) yroot->parent = xroot;
        else {
            yroot->parent = xroot;
            xroot->rank = xroot->rank + 1;
        }
    };



    ///////////////////////////////
    //// COMPRESSION ALGORITHM ////
    ///////////////////////////////

    template<typename Type> bool *
    ExtractBoundaries(Type *data, long res[3])
    {
        // create the boundaries array
        bool *boundaries = new bool[grid_size];
        if (!boundaries) { fprintf(stderr, "Failed to allocate memory for boundaries...\n"); return NULL; }

        // determine which pixels differ from east or south neighbors
        for (long iz = 0; iz < res[RN_Z]; ++iz) {
            for (long iy = 0; iy < res[RN_Y]; ++iy) {
                for (long ix = 0; ix < res[RN_X]; ++ix) {
                    long iv = IndicesToIndex(ix, iy, iz);

                    boundaries[iv] = false;

                    if (ix < res[RN_X] - 1) {
                        if (data[iv] != data[IndicesToIndex(ix + 1, iy, iz)]) boundaries[iv] = true;
                    }
                    if (iy < res[RN_Y] - 1) {
                        if (data[iv] != data[IndicesToIndex(ix, iy + 1, iz)]) boundaries[iv] = true;
                    }
                }
            }
        }

        return boundaries;
    };

    static unsigned long *
    ConnectedComponents(bool *boundaries, long res[3])
    {
        // create the connected components grid
        unsigned long *components = new unsigned long[grid_size];
        if (!components) { fprintf(stderr, "Failed to allocate memory for connected components...\n"); return NULL; }

        // initialize to zero
        for (long iv = 0; iv < grid_size; ++iv)
            components[iv] = 0;

        // run connected components for every slice
        for (long iz = 0; iz < res[RN_Z]; ++iz) {
            // create a vector of union find elements
            std::vector<UnionFindElement *> union_find = std::vector<UnionFindElement *>();

            // current label in connected component
            int curlab = 1;
            for (long iy = 0; iy < res[RN_Y]; ++iy) {
                for (long ix = 0; ix < res[RN_X]; ++ix) {
                    long iv = IndicesToIndex(ix, iy, iz);

                    // continue if boundary
                    if (boundaries[iv]) continue;

                    // only consider the pixel to the north and west
                    long north = IndicesToIndex(ix - 1, iy, iz);
                    long west = IndicesToIndex(ix, iy - 1, iz);

                    unsigned long neighbor_labels[2] = { 0, 0 };

                    // get the labels for the relevant neighbor
                    if (ix > 0) neighbor_labels[0] = components[north];
                    if (iy > 0) neighbor_labels[1] = components[west];

                    // if the neighbors are boundary, create new label
                    if (!neighbor_labels[0] && !neighbor_labels[1]) {
                        components[iv] = curlab;

                        // add to union find structure
                        union_find.push_back(new UnionFindElement(0));

                        // update the next label
                        curlab++;
                    }
                    // the two pixels have equal non-trivial values
                    else if (neighbor_labels[0] == neighbor_labels[1])
                        components[iv] = neighbor_labels[0];
                    else {
                        if (!neighbor_labels[0]) components[iv] = neighbor_labels[1];
                        else if (!neighbor_labels[1]) components[iv] = neighbor_labels[0];
                        else {
                            // take the minimum value
                            components[iv] = std::min(neighbor_labels[0], neighbor_labels[1]);

                            // set the equivalence relationship
                            Union(union_find[neighbor_labels[0] - 1], union_find[neighbor_labels[1] - 1]);
                        }
                    }
                }
            }

            // reset the current label to 1
            curlab = 1;

            // create the connected components in order
            for (long iy = 0; iy < res[RN_Y]; ++iy) {
                for (long ix = 0; ix < res[RN_X]; ++ix) {
                    long iv = IndicesToIndex(ix, iy, iz);

                    if (boundaries[iv]) continue;

                    // get the parent for this component
                    UnionFindElement *comp = Find(union_find[components[iv] - 1]);
                    if (!comp->label) {
                        comp->label = curlab;
                        curlab++;
                    }

                    components[iv] = comp->label;
                }
            }

            // free memory
            for (unsigned long iv = 0; iv < union_find.size(); ++iv)
                delete union_find[iv];
        }

        // return the connected components array
        return components;
    }

    template<typename Type> void
    IDMapping(unsigned long *components, Type *data, std::vector<unsigned long> &ids, long res[3])
    {
        // iterate over every slice
        for (int iz = 0; iz < res[RN_Z]; ++iz) {
            // create a set of components for this slice
            std::unordered_set<unsigned long> hash_map = std::unordered_set<unsigned long>();

            // iterate over the entire slice
            for (long iy = 0; iy < res[RN_Y]; ++iy) {
                for (long ix = 0; ix < res[RN_X]; ++ix) {
                    long iv = IndicesToIndex(ix, iy, iz);

                    // get the component label
                    unsigned long component_id = components[iv];

                    // if this component does not belong yet, add it
                    if (!hash_map.count(component_id)) {
                        hash_map.insert(component_id);

                        // add the segment id
                        unsigned long segment_id = (unsigned long)data[iv] + 1;
                        ids.push_back(segment_id);
                    }
                }
            }
        }
    }

    unsigned long *
    EncodeBoundaries(bool *boundaries, long res[3], long steps[3])
    {
        // determine the number of bloxks in each direction
        long nblocks[3];
        for (int dim = 0; dim <= 2; ++dim) {
            nblocks[dim] = (long) (ceil((double)res[dim] / steps[dim]) + 0.5);
        }
        long nwindows = nblocks[RN_Z] * nblocks[RN_Y] * nblocks[RN_X];

        // create an empty array for the encodings
        unsigned long *boundary_data = new unsigned long[nwindows];
        if (!boundary_data) { fprintf(stderr, "Failed to allocate memory for boundary windows...\n"); return NULL; }
        for (long iv = 0; iv < nwindows; ++iv)
            boundary_data[iv] = 0;

        for (long iz = 0; iz < res[RN_Z]; ++iz) {
            for (long iy = 0; iy < res[RN_Y]; ++iy) {
                for (long ix = 0; ix < res[RN_X]; ++ix) {
                    long iv = IndicesToIndex(ix, iy, iz);

                    // no encoding for non-boundaries
                    if (!boundaries[iv]) continue;

                    // find the block from the index
                    long zblock = iz / steps[RN_Z];
                    long yblock = iy / steps[RN_Y];
                    long xblock = ix / steps[RN_X];

                    // find the offset within the block
                    long zoffset = iz % steps[RN_Z];
                    long yoffset = iy % steps[RN_Y];
                    long xoffset = ix % steps[RN_X];

                    long block = zblock * (nblocks[RN_Y] * nblocks[RN_X]) + yblock * nblocks[RN_X] + xblock;
                    long offset = zoffset * (steps[RN_Y] * steps[RN_X]) + yoffset * steps[RN_X] + xoffset;

                    boundary_data[block] += (1LU << offset);
                }
            }
        }

        // return the encodings
        return boundary_data; 
    }

    void
    ValueMapping(unsigned long *boundary_data, std::vector<unsigned long> &values, long nwindows)
    {
        // keep a set of seen window values
        std::unordered_set<unsigned long> hash_map = std::unordered_set<unsigned long>();

        // go through all of the boundary data to create array of values
        for (long iv = 0; iv < nwindows; ++iv) {
            if (!hash_map.count(boundary_data[iv])) {
                hash_map.insert(boundary_data[iv]);
                values.push_back(boundary_data[iv]);
            }
        }

        // sort the values
        sort(values.begin(), values.end());

        // create mapping from values to indices
        std::unordered_map<unsigned long, unsigned long> mapping = std::unordered_map<unsigned long, unsigned long>();
        for (unsigned long iv = 0; iv < values.size(); ++iv) {
            mapping[values[iv]] = iv;
        }

        // update boundary data
        for (long iv = 0; iv < nwindows; ++iv) {
            boundary_data[iv] = mapping[boundary_data[iv]];
        }
    }

    template <typename Type> void
    EncodeIndeterminateLocations(bool *boundaries, Type *data, std::vector<unsigned long> &locations, long res[3])
    {
        for (long iz = 0; iz < res[RN_Z]; ++iz) {
            for (long iy = 0; iy < res[RN_Y]; ++iy) {
                for (long ix = 0; ix < res[RN_X]; ++ix) {
                    long iv = IndicesToIndex(ix, iy, iz);

                    if (!boundaries[iv]) continue;
                    else if (iy > 0 && !boundaries[IndicesToIndex(ix, iy - 1, iz)]) continue;
                    else if (ix > 0 && !boundaries[IndicesToIndex(ix - 1, iy, iz)]) continue;
                    else {
                        long north = IndicesToIndex(ix - 1, iy, iz);
                        long south = IndicesToIndex(ix + 1, iy, iz);
                        long east = IndicesToIndex(ix, iy - 1, iz);
                        long west = IndicesToIndex(ix, iy + 1, iz);
                        long up = IndicesToIndex(ix, iy, iz + 1);
                        long down = IndicesToIndex(ix, iy, iz - 1);

                        // see if any of the immediate neighbors are candidates
                        if (ix > 0 && !boundaries[north] && data[north] == data[iv]) locations.push_back(0);
                        else if (ix < res[RN_X] - 1 && !boundaries[south] && data[south] == data[iv]) locations.push_back(1);
                        else if (iy > 0 && !boundaries[east] && data[east] == data[iv]) locations.push_back(2);
                        else if (iy < res[RN_Y] - 1 && !boundaries[west] && data[west] == data[iv]) locations.push_back(3);
                        else if (iz > 0 && !boundaries[down] && data[down] == data[iv]) locations.push_back(4);
                        else if (iz < res[RN_Z] - 1 && !boundaries[up] && data[up] == data[iv]) locations.push_back(5);
                        else locations.push_back(((unsigned long)data[IndicesToIndex(ix, iy, iz)]) + 6);
                    }
                }
            }
        }
    }

    static unsigned char 
    BytesNeeded(unsigned long maximum_value)
    {
        if (maximum_value < 1L << 8) return 1;
        else if (maximum_value < 1L << 16) return 2;
        else if (maximum_value < 1L << 24) return 3;
        else if (maximum_value < 1L << 32) return 4;
        else if (maximum_value < 1L << 40) return 5;
        else if (maximum_value < 1L << 48) return 6;
        else if (maximum_value < 1L << 56) return 7;
        else return 8;
    }

    static void 
    AppendValue(std::vector<unsigned char> &data, unsigned long value, unsigned char nbytes)
    {
        for (unsigned char iv = 0; iv < nbytes; ++iv) {
            // get the 8 low order bits
            unsigned char low_order = value % 256;
            // add the one byte to the data
            data.push_back(low_order);
            // update the value by shifting one byte to the left
            value = value >> 8;
        }
    }

    template<typename Type> unsigned char * 
    Compress(Type *data, long res[3], long steps[3], long *nentries)
    {
        // set the global variables
        row_size = res[RN_X];
        sheet_size = res[RN_X] * res[RN_Y];
        grid_size = res[RN_X] * res[RN_Y] * res[RN_Z];

        // determine the number of blocks in each direction
        long nblocks[3];
        for (int dim = 0; dim <= 2; ++dim) {
            nblocks[dim] = (long) (ceil((double)res[dim] / steps[dim]) + 0.5);
        }
        long nwindows = nblocks[RN_Z] * nblocks[RN_Y] * nblocks[RN_X];

        // get the boundary voxels
        // std::clock_t start_time = std::clock();
        bool *boundaries = ExtractBoundaries(data, res);
        if (!boundaries) return NULL;
        // printf("Extract boundaries: %lf\n", (double)(std::clock() - start_time) / CLOCKS_PER_SEC);

        // get the connected components
        // use unsigned long since there could be more components than Type.MAX
        // start_time = std::clock();
        unsigned long *components = ConnectedComponents(boundaries, res);
        if (!components) return NULL;
        // printf("Connected components: %lf\n", (double)(std::clock() - start_time) / CLOCKS_PER_SEC);

        // get the ids
        // start_time = std::clock();
        std::vector<unsigned long> ids = std::vector<unsigned long>();
        IDMapping(components, data, ids, res);
        // printf("ID mapping: %lf\n", (double)(std::clock() - start_time) / CLOCKS_PER_SEC);

        // free memory
        delete[] components;

        // encode the boundary data
        // start_time = std::clock();
        unsigned long *boundary_data = EncodeBoundaries(boundaries, res, steps);
        if (!boundary_data) return NULL;
        // printf("Encode boundaries: %lf\n", (double)(std::clock() - start_time) / CLOCKS_PER_SEC);

        // map the window values
        // start_time = std::clock();
        std::vector<unsigned long> values = std::vector<unsigned long>();
        ValueMapping(boundary_data, values, nwindows);
        // printf("Map values: %lf\n", (double)(std::clock() - start_time) / CLOCKS_PER_SEC);

        // get the locations
        // start_time = std::clock();
        std::vector<unsigned long> locations = std::vector<unsigned long>();
        EncodeIndeterminateLocations(boundaries, data, locations, res);
        // printf("Encode locations: %lf\n", (double)(std::clock() - start_time) / CLOCKS_PER_SEC);

        // get the maximum id value
        unsigned long maximum_id = 0;
        for (unsigned long iv = 0; iv < ids.size(); ++iv)
            if (ids[iv] > maximum_id) maximum_id = ids[iv];

        // get the maximum value for the locations array
        unsigned long maximum_location = 0;
        for (unsigned long iv = 0; iv < locations.size(); ++iv)
            if (locations[iv] > maximum_location) maximum_location = locations[iv];

        // get the maximum boundary data window value
        unsigned long maximum_boundary_data = 2 * values.size() + 1;
        // find the maximum run of zeros
        unsigned long maximum_zero_run = 0;
        unsigned long current_run = 0;
        for (long iv = 0; iv < nwindows; ++iv) {
            if (!boundary_data[iv]) current_run++;
            else {
                if (current_run > maximum_zero_run) maximum_zero_run = current_run;
                current_run = 0;
            }
        }
        if (current_run > maximum_zero_run) maximum_zero_run = current_run;
        // multiply by two to pad for run length encoding
        maximum_zero_run *= 2;
        if (maximum_zero_run > maximum_boundary_data) maximum_boundary_data = maximum_zero_run;

        // get the number of bits per window as small as possible
        unsigned char bytes_per_window = steps[RN_X] * steps[RN_Y] * steps[RN_Z] / 8;
        unsigned char bytes_per_id = BytesNeeded(maximum_id);
        unsigned char bytes_per_location = BytesNeeded(maximum_location);
        unsigned char bytes_per_data = BytesNeeded(maximum_boundary_data);

        std::vector<unsigned char> compressed_data = std::vector<unsigned char>();

        // add the header to the decompressed data
        AppendValue(compressed_data, res[RN_Z], 8);
        AppendValue(compressed_data, res[RN_Y], 8);
        AppendValue(compressed_data, res[RN_X], 8);
        AppendValue(compressed_data, steps[RN_Z], 8);
        AppendValue(compressed_data, steps[RN_Y], 8);
        AppendValue(compressed_data, steps[RN_X], 8);
        AppendValue(compressed_data, values.size(), 8);
        AppendValue(compressed_data, ids.size(), 8);
        AppendValue(compressed_data, locations.size(), 8);
        // need one byte to say how large each chunk of data is
        AppendValue(compressed_data, bytes_per_window, 1);
        AppendValue(compressed_data, bytes_per_id, 1);
        AppendValue(compressed_data, bytes_per_location, 1);
        AppendValue(compressed_data, bytes_per_data, 1);
        // only final byte shows the original data Type
        AppendValue(compressed_data, sizeof(Type), 1);

        // add in all window values
        for (unsigned long iv = 0; iv < values.size(); ++iv)
            AppendValue(compressed_data, values[iv], bytes_per_window);
        // add in all ids
        for (unsigned long iv = 0; iv < ids.size(); ++iv)
            AppendValue(compressed_data, ids[iv], bytes_per_id);
        // add in all locations
        for (unsigned long iv = 0; iv < locations.size(); ++iv)
            AppendValue(compressed_data, locations[iv], bytes_per_location);
        
        // add in all boundary data - apply run length encoding
        unsigned long current_zero_run = 0;
        for (long iv = 0; iv < nwindows; ++iv) {
            if (!boundary_data[iv]) current_zero_run++;
            else {
                if (current_zero_run) AppendValue(compressed_data, 2 * current_zero_run, bytes_per_data);
                AppendValue(compressed_data, 2 * boundary_data[iv] + 1, bytes_per_data);
                current_zero_run = 0;
            }
        }

        // have to add in the last zero run
        if (current_zero_run) AppendValue(compressed_data, 2 * current_zero_run, bytes_per_data);

        unsigned char *compressed_pointer = new unsigned char[compressed_data.size()];
        for (unsigned long iv = 0; iv < compressed_data.size(); ++iv)
            compressed_pointer[iv] = compressed_data[iv];

        if (nentries) *nentries = compressed_data.size();

        // free memory
        delete[] boundaries;
        delete[] boundary_data;

        return compressed_pointer;
    };

    static unsigned long
    ExtractValue(unsigned char *data, unsigned long &offset, unsigned char nbytes)
    {
        // set the value to 0
        unsigned long value = 0;
        for (unsigned char iv = 0; iv < nbytes; ++iv) {
            // get the current bit values
            unsigned long byte = (unsigned long)data[offset];
            // shift over the proper amount
            byte = byte << (8 * iv);
            // update the value
            value += byte;
            // update the offset
            offset++;
        }

        return value;
    }

    static bool *
    DecodeBoundaries(unsigned long *boundary_data, std::vector<unsigned long> &values, long res[3], long steps[3])
    {
        // determine the number of bloxks in each direction
        long nblocks[3];
        for (int dim = 0; dim <= 2; ++dim) {
            nblocks[dim] = (long) (ceil((double)res[dim] / steps[dim]) + 0.5);
        }

        bool *boundaries = new bool[grid_size];
        for (long iv = 0; iv < grid_size; ++iv)
            boundaries[iv] = false;

        for (long iz = 0; iz < res[RN_Z]; ++iz) {
            for (long iy = 0; iy < res[RN_Y]; ++iy) {
                for (long ix = 0; ix < res[RN_X]; ++ix) {
                    long iv = IndicesToIndex(ix, iy, iz);

                    // find the block from the index
                    long zblock = iz / steps[RN_Z];
                    long yblock = iy / steps[RN_Y];
                    long xblock = ix / steps[RN_X];

                    // find the offset within the block
                    long zoffset = iz % steps[RN_Z];
                    long yoffset = iy % steps[RN_Y];
                    long xoffset = ix % steps[RN_X];

                    long block = zblock * (nblocks[RN_Y] * nblocks[RN_X]) + yblock * nblocks[RN_X] + xblock;
                    long offset = zoffset * (steps[RN_Y] * steps[RN_X]) + yoffset * steps[RN_X] + xoffset;

                    unsigned long value = values[boundary_data[block]];
                    if ((value >> offset) % 2) boundaries[iv] = true;
                }
            }
        }

        return boundaries;
    }

    template <typename Type> Type *
    IDReverseMapping(unsigned long *components, std::vector<unsigned long> ids, long res[3])
    {
        Type *decompressed_data = new Type[grid_size];
        for (long iv = 0; iv < grid_size; ++iv)
            decompressed_data[iv] = 0;

        int ids_index = 0;
        for (long iz = 0; iz < res[RN_Z]; ++iz) {
            // create mapping (not memory efficient but FAST!!)
            // number of components is guaranteed to be less than ids->size()
            unsigned long *mapping = new unsigned long[ids.size() + 1];
            for (unsigned long iv = 0; iv < ids.size() + 1; ++iv)
                mapping[iv] = 0;

            for (long iy = 0; iy < res[RN_Y]; ++iy) {
                for (long ix = 0; ix < res[RN_X]; ++ix) {
                    long iv = IndicesToIndex(ix, iy, iz);
                    if (!mapping[components[iv]]) {
                        mapping[components[iv]] = ids[ids_index];
                        ids_index++;
                    }

                    decompressed_data[iv] = (Type)(mapping[components[iv]] - 1);
                }
            }
        }

        return decompressed_data;
    }

    template <typename Type> void
    DecodeIndeterminateLocations(bool *boundaries, Type *decompressed_data, std::vector<unsigned long> locations, long res[3])
    {
        long index = 0;

        // go through all voxels
        for (long iz = 0; iz < res[RN_Z]; ++iz) {
            for (long iy = 0; iy < res[RN_Y]; ++iy) {
                for (long ix = 0; ix < res[RN_X]; ++ix) {
                    long iv = IndicesToIndex(ix, iy, iz);
                    
                    // get the north and west neighbors
                    long north = IndicesToIndex(ix - 1, iy, iz);
                    long west = IndicesToIndex(ix, iy - 1, iz);

                    if (!boundaries[iv]) continue;
                    else if (ix > 0 && !boundaries[north]) {
                        decompressed_data[iv] = decompressed_data[north];
                    }
                    else if (iy > 0 && !boundaries[west]) {
                        decompressed_data[iv] = decompressed_data[west];
                    }
                    else {
                        unsigned long offset = locations[index];
                        if (offset == 0) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix - 1, iy, iz)];
                        else if (offset == 1) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix + 1, iy, iz)];
                        else if (offset == 2) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix, iy - 1, iz)];
                        else if (offset == 3) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix, iy + 1, iz)];
                        else if (offset == 4) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix, iy, iz - 1)];
                        else if (offset == 5) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix, iy, iz + 1)];
                        else decompressed_data[iv] = (Type)(offset - 6);
                        index++;
                    }
                }
            }
        }
    }

    template <typename Type> Type*
    Decompress(unsigned char *compressed_data, long *res)
    {
        // extract all of the header information
        if (!res) res = new long[3];
        long steps[3];
        
        // the offset for the compressed data
        unsigned long offset = 0;
        // extract the header
        res[RN_Z] = ExtractValue(compressed_data, offset, 8);
        res[RN_Y] = ExtractValue(compressed_data, offset, 8);
        res[RN_X] = ExtractValue(compressed_data, offset, 8);
        steps[RN_Z] = ExtractValue(compressed_data, offset, 8);
        steps[RN_Y] = ExtractValue(compressed_data, offset, 8);
        steps[RN_X] = ExtractValue(compressed_data, offset, 8);
        unsigned long nvalues = ExtractValue(compressed_data, offset, 8);
        unsigned long nids = ExtractValue(compressed_data, offset, 8);
        unsigned long nlocations = ExtractValue(compressed_data, offset, 8);
        unsigned char bytes_per_window = ExtractValue(compressed_data, offset, 1);
        unsigned char bytes_per_id = ExtractValue(compressed_data, offset, 1);
        unsigned char bytes_per_location = ExtractValue(compressed_data, offset, 1);
        unsigned char bytes_per_data = ExtractValue(compressed_data, offset, 1);
        unsigned char bytes_for_output = ExtractValue(compressed_data, offset, 1);

        // set the global variables
        row_size = res[RN_X];
        sheet_size = res[RN_X] * res[RN_Y];
        grid_size = res[RN_X] * res[RN_Y] * res[RN_Z];

        // determine the number of blocks in each direction
        long nblocks[3];
        for (int dim = 0; dim <= 2; ++dim) {
            nblocks[dim] = (long) (ceil((double)res[dim] / steps[dim]) + 0.5);
        }
        long nwindows = nblocks[RN_Z] * nblocks[RN_Y] * nblocks[RN_X];

        // allocate memory for all arrays
        std::vector<unsigned long> ids = std::vector<unsigned long>();
        std::vector<unsigned long> values = std::vector<unsigned long>();
        std::vector<unsigned long> locations = std::vector<unsigned long>();
        unsigned long *boundary_data = new unsigned long[nwindows];
        for (unsigned long iv = 0; iv < nvalues; ++iv)
            values.push_back(ExtractValue(compressed_data, offset, bytes_per_window));
        for (unsigned long iv = 0; iv < nids; ++iv)
            ids.push_back(ExtractValue(compressed_data, offset, bytes_per_id));
        for (unsigned long iv = 0; iv < nlocations; ++iv)
            locations.push_back(ExtractValue(compressed_data, offset, bytes_per_location));

        // get the boundary data (undo run length encoding)
        long iv = 0;
        while (iv < nwindows) {
            unsigned long window_value = ExtractValue(compressed_data, offset, bytes_per_data);
            if (window_value % 2) {
                window_value = window_value / 2;
                assert (iv < nwindows);
                boundary_data[iv] = window_value;
                iv++;
            }
            else {
                unsigned long nzeros = window_value / 2;
                for (unsigned long iz = 0; iz < nzeros; ++iz, ++iv) {
                    boundary_data[iv] = 0;
                }
            }
        }
        
        // get the boundaries from the data
        // std::clock_t start_time = std::clock();
        bool *boundaries = DecodeBoundaries(boundary_data, values, res, steps);
        if (!boundaries) return NULL;
        // printf("Decode boundaries: %lf\n", (double)(std::clock() - start_time) / CLOCKS_PER_SEC);

        // free memory 
        delete[] boundary_data;

        // get the connected components
        // start_time = std::clock();
        unsigned long *components = ConnectedComponents(boundaries, res);
        if (!components) return NULL;
        // printf("Connected components: %lf\n", (double)(std::clock() - start_time) / CLOCKS_PER_SEC);

        // decompress the data
        // start_time = std::clock();
        Type *decompressed_data = IDReverseMapping<Type>(components, ids, res);
        if (!decompressed_data) return NULL;
        // printf("Reverse mapping: %lf\n", (double)(std::clock() - start_time) / CLOCKS_PER_SEC);

        // free memory
        delete[] components;

        // decode the final indeterminate locations
        // start_time = std::clock();
        DecodeIndeterminateLocations(boundaries, decompressed_data, locations, res);
        // printf("Decode locations: %lf\n", (double)(std::clock() - start_time) / CLOCKS_PER_SEC);

        // return the decompressed data
        return decompressed_data;
    }
};

#endif