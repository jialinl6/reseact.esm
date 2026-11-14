# ===========================================================================
# rx_band_program.jl -- the analytic Jacobian's BAND PROGRAM, from
# EarthSciASTDiff's evaluation document.
# ===========================================================================
#
#   band_program(part; source_layout, build_kwargs, form)  ->  BandProgram
#
# EarthSciASTDiff 0.3 evaluates nothing: b4c9c1b ("evaluate nothing") removed
# `prepare_jacobian` / `JacobianEvaluator`, and every value now comes from
# EarthSciAST evaluating one of its documents. What it still hands over is
# structural, and three pieces of it are what this file assembles:
#
#   evaluation_document   the derived `JacobianEval` document (spec §8): one
#                         band-field state `J_k` per band entry whose RHS is
#                         that band's coefficient, so ONE right-hand-side call
#                         at (u, p, t) leaves every band value in `du`;
#   jacobian_pattern      the structural pattern over a GIVEN state layout --
#                         the runner's, via `source_layout`;
#   scatter_plan(Csr) +   (band k, field cell) -> storage position, checked
#   bind_plan             against the source's own bands. Executing it is the
#                         consumer's (spec §9.5); this is the consumer.
#
# The result carries exactly the fields `block_jac_plan` and `validate_plan`
# (tools/reactant_handoff/rx_sym_block_jac.jl) were written against --
# `rownames`, `umap`, `uj`, `scatter`, `prototype`, `structure` -- with the
# meanings `prepare_jacobian` gave them, so the gather plan and the check that
# stands in front of it are unchanged by the port.
#
# HOW THE DOCUMENT IS BUILT, AND WHY NOT AS `model_name = "JacobianEval"`. A
# flattened source's names are scoped (`SuperFast.O3`), and the evaluation
# document lays such a source out as its COMPONENT TREE (spec §8.1): on the
# ReSEACT chemistry half, 12 top-level models, the source components holding
# their zeroed state equations and `JacobianEval` holding the 69 band fields.
# Building `JacobianEval` alone fails at once (measured on EarthSciAST 0.8 /
# EarthSciASTDiff 0.3.0, 7x7x8: "parameter_overrides: unknown parameter
# 'GEOSFP.CLDTOT'"), because the parameters live in the other components.
# The tree is loaded and FLATTENED, as EarthSciASTDiff's own test/docvals.jl
# does: the source's state cells then come back under their flattened names,
# identical to the runner's (all 5,096 at 7x7x8), and band cells are
# `JacobianEval.J_k[...]`. A source with no scoped name gives a one-model
# document, built by name.
#
# THE MATERIALIZER IS NOT OPTIONAL WHERE THE RHS HAS ONE. Without
# `materialize_out`, EarthSciAST keeps derived data-only fields inline. At 4x5
# and 2x2.5, LandUseRemap's `fxy_<class>` (11 Wesely classes, each a sum over
# the 101 x 81 = 8,181 MODIS cells in one model cell) then becomes a
# whole-array contraction, and the direct StableHLO emitter refuses all 11
# ("E_DIRECT_EMIT_UNSUPPORTED ... cannot lower 11 whole-array contraction(s)").
# With one, the chemistry RHS has 0 and so does this program. So pass a
# `DiscreteMaterializer` in `build_kwargs` whenever the RHS it serves was built
# with one; this function runs its `materialize!()` once, before anything reads
# the buffers, and the caller runs it at every forcing refresh, before syncing
# `rx_bufs(band.fJ)` to the device.
#
# WHAT IS CHECKED, AND THROWS. Every runner state must be a state of the band
# model (that is `umap`); every band-field cell the bound plan names must exist
# in it; every (row, column) the plan targets must resolve in the runner's
# layout and lie on `jacobian_pattern`'s pattern; and every pattern entry must
# receive at least one band. The last two are a CROSS-CHECK of two independent
# EarthSciASTDiff routes -- the pattern enumerates band points over the
# runner's layout, the bound plan decodes them in its own canonical order --
# and it is what stands in for the assembly `prepare_jacobian` used to do
# itself, now that the scatter `validate_plan` compares against is built here.
#
# THE SETUP CACHE (`cache`, prototypes/reseact_3d_chem/setup_cache.jl; off unless
# RESEACT_CACHE_DIR is set) keeps everything here that does not depend on the
# live buffers: the evaluation document (as the JSON the build parses), its
# fields, the band model's var map, umap, the scatter, the pattern, the structure
# and the row names. On a hit the document is built in the requested form over
# the caller's buffers, and the build's var map and the runner's row names must
# equal the cached ones, or everything is recomputed and the entry overwritten.
# Neither the evaluation document, jacobian_pattern nor the scatter plan runs.
# All three have to be cached together: on the cluster, with only the document
# cached, jacobian_pattern took 12.5-14.7 s instead of 1.7 s, because computing
# the document had memoised what it needs.
# ===========================================================================

module RxBandProgram

using EarthSciAST
using EarthSciASTDiff
using SparseArrays
const EA = EarthSciAST
const ED = EarthSciASTDiff

export BandProgram, band_program

struct BandProgram
    fJ                                  # the band build: :oop (compile with rx_rhs), or the
                                        # :inplace f!(du, u, p, t) when form = :inplace
    pJ                                  # the band build's own parameters
    var_map::Dict{String,Int}           # band-model name -> slot in uj / duj
    dmJ                                 # its DiscreteMaterializer, or nothing
    fields::Vector{String}              # fields[k]: band entry k's field, `J_k`
    rownames::Vector{String}            # the runner's states, in its slot order
    umap::Vector{Int}                   # uj[umap[i]] = u[i]
    uj::Vector{Float64}                 # zeros(NJ): the band state a host lift starts from
    scatter::Vector{Tuple{Int,Int}}     # (slot in duj, position in prototype.nzval)
    prototype::SparseMatrixCSC{Float64,Int}
    structure::Symbol
    pmissing::Vector{Symbol}            # band parameters the runner's `p` lacks
    timings::NamedTuple
end

Base.show(io::IO, b::BandProgram) = print(io,
    "BandProgram(fields=$(length(b.fields)), NJ=$(length(b.uj)), ",
    "nnz=$(nnz(b.prototype)), scatter=$(length(b.scatter)), structure=$(b.structure)",
    isempty(b.pmissing) ? "" :
        ", $(length(b.pmissing)) band parameters not in the runner's p, e.g. " *
        "$(first(b.pmissing))", ")")

_sysview(input, model_name) =
    model_name === nothing ? ED.sysview(input) : ED.sysview(input, String(model_name))

# The band build: the evaluation document's JSON, built as a flattened component
# tree or as the one-model document, in the form `kw` asks for. Materializes the
# build's DiscreteMaterializer, if it has one, before anything reads the buffers.
function _band_build(json::String, tree::Bool, kw, dmJ)
    src = EA.load_string(json)
    fJ, u0J, pJ, _, vmJ0 = tree ?
        EA.build_evaluator(EA.flatten(src); kw...) :
        EA.build_evaluator(src; model_name = "JacobianEval", kw...)
    dmJ === nothing || dmJ.materialize!()
    return fJ, length(u0J), pJ, Dict{String,Int}(String(k) => Int(v) for (k, v) in vmJ0)
end

# Informational. The traced callers hand the band program the RUNNER's `p`
# (it is the differentiated payload), and the emitter reads a parameter by
# name (`getfield(p, sym)`), so a band parameter the runner lacks fails the
# trace by name if it is read and costs nothing if it is not.
_pmissing(pJ, source_layout) = sort!(setdiff(collect(keys(pJ)), collect(keys(source_layout.p))))

_secs(x) = string(round(x; digits = 1))

"""
    band_program(input; source_layout, model_name = nothing, build_kwargs = NamedTuple(),
                 form = :oop, cache = nothing) -> BandProgram

Build the band program of `input`'s state Jacobian (a `FlattenedSystem`, or an
`EsmFile` with `model_name`) and the index tables that place its output.

`source_layout = (u0 = …, p = …, var_map = …)` is the 2nd, 3rd and 5th return of
the consumer's OWN `build_evaluator` of `input`, with the SAME `build_kwargs`.
Its `var_map` fixes `rownames` and the matrix layout, so the plan built from the
result indexes the program the consumer compiles. `build_kwargs` goes to the
band build with `form` imposed; put the `materialize_out` the RHS uses in it.
`form = :oop` (the default) is the traceable output the Reactant callers compile;
`form = :inplace` makes `fJ` an ordinary `f!(du, u, p, t)` for the native runner.
Everything after the build -- `umap`, the pattern, the scatter -- is the same for
both.

`cache` is `nothing` (the default) or a `setup_cache(...)` handle
(prototypes/reseact_3d_chem/setup_cache.jl) keyed on what `input` was prepared
from; see this file's header. Prints one line, HIT or MISS, through `cache.say`.
"""
function band_program(input; source_layout, model_name = nothing,
                      build_kwargs = NamedTuple(), form::Symbol = :oop, cache = nothing)
    vm = Dict{String,Int}(String(k) => Int(v) for (k, v) in source_layout.var_map)
    rownames = first.(sort(collect(vm), by = last))
    n = length(rownames)
    n == length(source_layout.u0) || error(
        "band_program: source_layout has $(length(source_layout.u0)) states but a " *
        "$(length(vm))-entry var_map")
    kw = merge(build_kwargs, (; form = form))
    dmJ = get(build_kwargs, :materialize_out, nothing)

    # ---- a setup-cache hit: build the cached document, reuse the rest -------
    stale = ""
    if cache !== nothing
        tl = time(); st = cache.load(); tl = time() - tl
        if st !== nothing
            if st.rownames != rownames
                stale = "the runner's state layout is not the cached one"
            else
                t1 = time()
                fJ, NJ, pJ, vmJ = _band_build(st.json, st.tree, kw, dmJ)
                tbuild = time() - t1
                if vmJ == st.vmJ
                    cache.say("    setup cache: $(cache.what) HIT, loaded in $(_secs(tl)) s  $(cache.file)")
                    return BandProgram(fJ, pJ, vmJ, dmJ, st.fields, rownames, st.umap, zeros(NJ),
                                       st.scatter, st.prototype, st.structure,
                                       _pmissing(pJ, source_layout),
                                       (doc = 0.0, build = tbuild, pattern = 0.0, scatter = 0.0))
                end
                stale = "the band build's var map is not the cached one"
            end
        end
    end

    t0 = time()
    doc, fields0 = ED.evaluation_document(input; wrt = :states, model_name = model_name)
    fields = String.(fields0)
    tdoc = time() - t0

    t1 = time()
    tree = length(doc["models"]) > 1
    json = ED.stable_json(doc)
    fJ, NJ, pJ, vmJ = _band_build(json, tree, kw, dmJ)
    tbuild = time() - t1
    pfx = tree ? "JacobianEval." : ""

    # ---- the state lift: runner state i -> band-model slot ------------------
    umap = Vector{Int}(undef, n)
    for (i, nm) in enumerate(rownames)
        j = get(vmJ, nm, 0)
        j == 0 && error(
            "band_program: runner state `$nm` is not a state of the band model " *
            "($(tree ? "flattened component tree" : "one-model document"), $NJ " *
            "states). The band model and the runner disagree on names, and a lift " *
            "by position would put every band in the wrong cell.")
        umap[i] = j
    end
    allunique(umap) || error("band_program: two runner states map to one band-model slot")

    # ---- the pattern, over the runner's layout ------------------------------
    t2 = time()
    pat = ED.jacobian_pattern(input; wrt = :states, model_name = model_name,
                              source_layout = source_layout)
    P = pat.pattern
    size(P) == (n, n) || error("band_program: jacobian_pattern is $(size(P)), expected ($n, $n)")
    pat.rownames == rownames || error(
        "band_program: jacobian_pattern's row order is not the runner's")
    proto = SparseMatrixCSC(n, n, copy(P.colptr), copy(P.rowval),
                            zeros(length(P.rowval)))
    structure = ED.detect_structure(pat.entries, _sysview(input, model_name))
    tpat = time() - t2

    # ---- the scatter: bound Csr plan -> (duj slot, nzval position) ---------
    t3 = time()
    plan = ED.scatter_plan(input, ED.Csr(); model_name = model_name)
    bound = ED.bind_plan(input, plan; model_name = model_name)
    targets = ED.scatter_targets(plan)
    length(plan.bands) == length(fields) || error(
        "band_program: the scatter plan has $(length(plan.bands)) bands, the " *
        "evaluation document $(length(fields)) fields")
    cellnames = [let sh = plan.bands[k][3]
                     isempty(sh) ? [pfx * fields[k]] :
                         [pfx * fields[k] * "[" * join(Tuple(c), ",") * "]"
                          for c in CartesianIndices(Tuple(sh))]
                 end for k in eachindex(fields)]
    scatter = Vector{Tuple{Int,Int}}(undef, length(bound.pairs))
    covered = falses(nnz(proto))
    for (q, (k, cell, pos)) in enumerate(bound.pairs)
        fname = cellnames[k][cell]
        slot = get(vmJ, fname, 0)
        slot == 0 && error("band_program: band field cell `$fname` is not in the band model")
        tg = targets[pos]
        tg === nothing && error("band_program: plan position $pos holds no (row, column)")
        r = get(vm, tg[1], 0); c = get(vm, tg[2], 0)
        (r == 0 || c == 0) && error(
            "band_program: the plan puts band $k at ($(tg[1]), $(tg[2])), which the " *
            "runner's layout does not hold")
        rng = proto.colptr[c]:(proto.colptr[c + 1] - 1)
        i = searchsortedfirst(view(proto.rowval, rng), r)
        (i <= length(rng) && proto.rowval[rng[i]] == r) || error(
            "band_program: band $k lands on ($(tg[1]), $(tg[2])), which is OFF " *
            "jacobian_pattern's pattern -- the bound plan and the pattern disagree")
        covered[rng[i]] = true
        scatter[q] = (slot, rng[i])
    end
    nc = count(!, covered)
    nc == 0 || error(
        "band_program: $nc of $(nnz(proto)) jacobian_pattern entries receive no band " *
        "from the bound plan -- the two routes disagree on the pattern")
    tscat = time() - t3

    if cache !== nothing
        ts = cache.store((; json, tree, fields, vmJ, rownames, umap, scatter,
                            prototype = proto, structure = Symbol(structure)))
        cache.say("    setup cache: $(cache.what) " *
                  (isempty(stale) ? "MISS" : "STALE ($stale)") *
                  ", computed in $(_secs(time() - t0)) s, " *
                  (ts === nothing ? "not stored" : "stored in $(_secs(ts)) s") * "  $(cache.file)")
    end
    return BandProgram(fJ, pJ, vmJ, dmJ, fields, rownames, umap, zeros(NJ), scatter,
                       proto, Symbol(structure), _pmissing(pJ, source_layout),
                       (doc = tdoc, build = tbuild, pattern = tpat, scatter = tscat))
end

end # module

using .RxBandProgram
