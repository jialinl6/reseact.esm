# ===========================================================================
# setup_cache.jl -- the opt-in SETUP CACHE: model preparation and the band
# program's static parts, kept between runs.
# ===========================================================================
#
#   setup_cache(what, model_path, metaparameters; extra, say) -> handle | nothing
#   setup_cached(compute, handle)                             -> value
#
# OFF UNLESS RESEACT_CACHE_DIR names a directory. Two callers use it:
# `prepare_split_docs` (its `(docs, parts)`) and `band_program`
# (tools/rx_band_program.jl: the evaluation document, its fields, the band
# model's var map, umap, the scatter, the pattern, the structure and the row
# names -- everything it computes that does not depend on the live buffers).
#
# ONE FILE PER KEY. The key is SHA-256 over what the cached objects are a
# function of: the model file's bytes, the sorted metaparameters, this
# repository's git HEAD, and a digest of the .jl sources under src/ of
# EarthSciAST, EarthSciASTSplitter and EarthSciASTDiff as the active environment
# resolves them, so a dev edit to any of them invalidates it. Also in it: what is
# cached, the caller's `extra`, and the Julia version, because Serialization's
# format is not stable across Julia versions. NOT in it: uncommitted edits to this
# repository, which leave HEAD alone -- empty the directory after editing
# split_common.jl or tools/rx_band_program.jl without committing.
#
# A file is written to a temporary name in the same directory and renamed into
# place, so no reader sees half of one and two runs storing the same key do not
# interleave. Any failure to read one -- truncated, another format, another key --
# recomputes it and overwrites the file. A failure to write one is reported and
# the run carries on uncached. Each use prints one line: HIT or MISS, with timings.
#
# A LOADED DICT ITERATES IN THE ORDER IT WAS STORED IN. Serialization rebuilds a
# Dict by inserting its pairs into a fresh table, and the order a Dict iterates in
# depends on its table's size and history, not only on its keys: on reseact.esm
# (7x7x8) 3 of 137,646 Dicts in (docs, parts) came back reordered (the `bindings`
# of an apply_expression_template), so the documents the RHS halves are built from
# no longer serialized to the same JSON. `LayoutSerializer` stores a Dict keyed by
# String or Symbol as its table itself, so it comes back with the same layout and
# order. That is sound because those keys hash by content, the same in every
# process of one Julia version (which is in the key), and every restored Dict is
# checked by looking each of its keys up; a failure is a load failure. Measured on
# the same model: the warm documents' JSON is then byte-identical to the uncached
# documents'.
#
# Measured on the cluster (CONUS 4x5, 1 day, a probe caching the prepared model and
# the band program's document only): model preparation 35.7 -> 1.3 s, the document
# 22.2 -> 0.2 s. The document alone is not enough: computing it memoises what
# jacobian_pattern needs, and with it cached the pattern took 12.5-14.7 s instead
# of 1.7 s. So band_program caches the pattern and the scatter plan with it.
import SHA, Serialization

const SETUP_CACHE_FORMAT = 1

# Serialization's own Serializer, field for field, plus the two Dict methods below.
mutable struct LayoutSerializer{I<:IO} <: Serialization.AbstractSerializer
    io::I
    counter::Int
    table::IdDict{Any,Any}
    pending_refs::Vector{Int}
    known_object_data::Dict{UInt64,Any}
    version::Int
    LayoutSerializer(io::I) where {I<:IO} =
        new{I}(io, 0, IdDict(), Int[], Dict{UInt64,Any}(), Serialization.ser_version)
end
const _LayoutDict = Dict{<:Union{String,Symbol}}
function Serialization.serialize(s::LayoutSerializer, d::_LayoutDict)
    Serialization.serialize_cycle_header(s, d) && return
    for f in fieldnames(Dict)
        Serialization.serialize(s, getfield(d, f))
    end
end
function Serialization.deserialize(s::LayoutSerializer, T::Type{Dict{K,V}}) where {K<:Union{String,Symbol},V}
    t = T()
    Serialization.deserialize_cycle(s, t)
    for f in fieldnames(Dict)
        setfield!(t, f, Serialization.deserialize(s))
    end
    all(k -> haskey(t, k), keys(t)) || error("a restored $T does not find its own keys")
    return t
end

# The key's process-wide part: this repository's HEAD and the three packages'
# sources, read once. `nothing` when HEAD cannot be read (no git, not a checkout).
const _SETUP_CACHE_SOURCES = Ref{Union{Nothing,Vector{UInt8}}}(nothing)
function _setup_cache_sources()
    _SETUP_CACHE_SOURCES[] === nothing || return _SETUP_CACHE_SOURCES[]
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    head = try
        readchomp(pipeline(`git -C $repo rev-parse HEAD`; stderr = devnull))
    catch
        return nothing
    end
    ctx = SHA.SHA256_CTX()
    SHA.update!(ctx, codeunits("HEAD $head\0"))
    for pkg in ("EarthSciAST", "EarthSciASTSplitter", "EarthSciASTDiff")
        entry = Base.find_package(pkg)
        SHA.update!(ctx, codeunits("$pkg $(entry === nothing ? "absent" : "src")\0"))
        entry === nothing && continue
        src = dirname(entry)
        for f in sort!([joinpath(r, f) for (r, _, fs) in walkdir(src) for f in fs if endswith(f, ".jl")])
            SHA.update!(ctx, codeunits(relpath(f, src) * "\0"))
            SHA.update!(ctx, SHA.sha256(read(f)))
        end
    end
    return _SETUP_CACHE_SOURCES[] = SHA.digest!(ctx)
end

_cache_s(x) = string(round(x; digits = 1))

"""
    setup_cache(what, model_path, metaparameters; extra = (), say = println)

The handle of one setup-cache entry, or `nothing` when RESEACT_CACHE_DIR is unset
(or this checkout's git HEAD cannot be read). `what` names what is cached and is
part of the key with `extra`; the rest of the key is in this file's header.

The handle is a NamedTuple: `load()` returns the stored value, or `nothing` when
there is none or it cannot be read; `store(x)` writes `x` and returns the seconds
it took, or `nothing` when it could not; `say` prints; `what` and `file` label the
line the caller prints.
"""
function setup_cache(what::AbstractString, model_path::AbstractString,
                     metaparameters::AbstractDict; extra = (), say = println)
    dir = get(ENV, "RESEACT_CACHE_DIR", "")
    isempty(dir) && return nothing
    src = _setup_cache_sources()
    if src === nothing
        say("    setup cache: off -- this checkout's git HEAD, part of the key, cannot be read")
        return nothing
    end
    mp = sort!([string(k, "=", v) for (k, v) in metaparameters])
    ctx = SHA.SHA256_CTX()
    for s in ("reseact setup cache $SETUP_CACHE_FORMAT", what, "julia $VERSION",
              join(mp, ","), repr(extra))
        SHA.update!(ctx, codeunits(s * "\0"))
    end
    SHA.update!(ctx, src)
    SHA.update!(ctx, SHA.sha256(read(model_path)))
    key = bytes2hex(SHA.digest!(ctx))
    file = joinpath(dir, replace(what, r"[^A-Za-z0-9]+" => "_") * "-" * key * ".jls")
    load = function ()
        isfile(file) || return nothing
        try
            d = open(io -> Serialization.deserialize(LayoutSerializer(io)), file)
            (d isa NamedTuple && get(d, :format, 0) == SETUP_CACHE_FORMAT &&
             get(d, :key, "") == key) || error("not a format-$SETUP_CACHE_FORMAT entry of this key")
            return d.value
        catch e
            say("    setup cache: cannot read $file (" * first(split(sprint(showerror, e), '\n')) *
                "); recomputing it")
            return nothing
        end
    end
    store = function (x)
        t = time(); tmp = ""
        try
            mkpath(dir)
            tmp = tempname(dir; cleanup = false)
            open(tmp, "w") do io
                s = LayoutSerializer(io)
                Serialization.writeheader(s)
                Serialization.serialize(s, (format = SETUP_CACHE_FORMAT, key = key, value = x))
            end
            mv(tmp, file; force = true)
            return time() - t
        catch e
            isempty(tmp) || rm(tmp; force = true)
            say("    setup cache: cannot write $file (" * first(split(sprint(showerror, e), '\n')) * ")")
            return nothing
        end
    end
    return (; what = String(what), file, load, store, say)
end

"""
    setup_cached(compute, cache)

`compute()` through the setup cache: the stored value on a hit, else `compute()`'s,
stored for the next run. With `cache === nothing` (the cache off) it is just
`compute()`. Prints one line saying which, with the timings.
"""
function setup_cached(compute, cache)
    cache === nothing && return compute()
    tl = time(); v = cache.load(); tl = time() - tl
    if v !== nothing
        cache.say("    setup cache: $(cache.what) HIT, loaded in $(_cache_s(tl)) s  $(cache.file)")
        return v
    end
    tc = time(); v = compute(); tc = time() - tc
    ts = cache.store(v)
    cache.say("    setup cache: $(cache.what) MISS, computed in $(_cache_s(tc)) s, " *
              (ts === nothing ? "not stored" : "stored in $(_cache_s(ts)) s") * "  $(cache.file)")
    return v
end
