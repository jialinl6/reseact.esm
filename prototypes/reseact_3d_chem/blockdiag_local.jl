# Load EarthSciMLBase's BlockDiagonal machinery (the block-diagonal Jacobian
# prototype the user asked for) WITHOUT dragging in the whole EarthSciMLBase
# dependency tree (ModelingToolkit/Catalyst). `map_algorithm.jl` supplies the
# MapAlgorithm dispatch `blockdiagonal.jl` needs; it references AcceleratedKernels
# (`AK`) — imported here — for the MapKernel/mapreduce_range paths (BlockDiagonal's
# `ldiv!` hardcodes `MapKernel()`).
module BlockDiag

import AcceleratedKernels as AK          # map_algorithm.jl's `AK` (ldiv!/lufact paths)
import Polyester                         # MapPolyester's methods (EarthSciMLBase ext/polyester_map.jl)

const _MB_SRC = get(ENV, "EARTHSCIMLBASE_SRC",
    normpath(joinpath(@__DIR__, "..", "..", "..", "EarthSciMLBase", "src")))
include(joinpath(_MB_SRC, "map_algorithm.jl"))     # MapAlgorithm, MapBroadcast, map_closure_to_range, mapreduce_range
include(joinpath(_MB_SRC, "blockdiagonal.jl"))     # BlockDiagonal + LU/ldiv!/mul!/+I, LinearSolve glue
# MapPolyester: the block LU and solve on Polyester's worker pool, the one EarthSciAST's kernels run on (the package
# loads this file through its EarthSciMLBasePolyesterExt extension; here it is included like the files above).
const _MB_POLYESTER = normpath(joinpath(_MB_SRC, "..", "ext", "polyester_map.jl"))
isfile(_MB_POLYESTER) && include(_MB_POLYESTER)

export BlockDiagonal, MapBroadcast, MapPolyester, block, nblocks

end # module
