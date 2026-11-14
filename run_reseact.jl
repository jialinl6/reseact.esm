#!/usr/bin/env julia
# ===========================================================================
# run_reseact.jl -- run the full ReSEACT model with the NATIVE in-place runner.
# ===========================================================================
# THE DEFAULT RUN IS A ONE-WEEK CONUS SIMULATION: 13x7x72 cells (lon -125..-65,
# lat 26..50; 6,552 cells, 85,176 states) over 168 h of GEOS-FP meteorology,
# macro-stepped at 300 s. Measured on this machine at `julia -t 8`: 683 s build
# + 15,227 s solve (4.2 h wall, ~40x realtime). Override any of it through the
# environment (below) -- nothing about the driver is CONUS-specific.
#
# It builds the operator-split model as two in-place `f!(du,u,p,t)` closures and
# drives a Lie-Trotter operator split entirely on the CPU:
#
#   transport (non-stiff, stencil terms)    -> SSPRK43        (explicit)
#   chemistry (stiff, cell-local pointwise) -> Rosenbrock23   (implicit, with a
#                                              block-diagonal Jacobian: one
#                                              NS x NS block per cell)
#
# The two halves share ONE state vector and ONE set of live GEOS-FP forcing
# buffers, so f_full(u) = f_transport(u) + f_chemistry(u) exactly. This is the
# REFERENCE runner: `run_reseact_reactant.jl` is validated against it.
#
# The Lie-Trotter coupling is done by the REAL SciML operator-splitting solver,
# OrdinaryDiffEqOperatorSplitting.LieTrotterGodunov, via the thin
# `lie_trotter_solve` driver in tools/reactant_handoff/op_split.jl. That package
# USED to be unusable here: it wraps each sub-operator in a plain ODEProblem,
# whose init() densified our BlockDiagonal jac_prototype into an N x N dense
# matrix (O(N^3) LU, the exact cost the split exists to avoid). The root cause
# was a single missing `similar(::BlockDiagonal, ::Type)` method; blockdiag_similar.jl
# supplies it. See SPLIT_SOLVER.md and op_split.jl for the full story.
#
# Two things a week-long run needs that a validation window does not, and that
# are therefore built in rather than optional:
#
#   * BISECTION ON FAILURE. The pre-dawn NO minimum (first hit at sim-hour 11.33)
#     makes Rosenbrock23's TRIAL sub-steps go negative and convergence collapse.
#     `clamp_nonneg` is INERT against it -- it inspects only ACCEPTED states, and
#     those never go negative. `lie_trotter_solve_bisect` retries the failed macro
#     step over two halves instead, which clears it in 2 bisections and costs LESS
#     chemistry work than the failed 300 s attempt. The durable fix is upstream
#     (the splitting package silently drops callbacks, so `PositiveDomain` cannot
#     fire) -- see HELPERS.md §3.
#   * STREAMING OUTPUT. Full spatiotemporal output goes through EarthSciAST's
#     sink protocol (`derive_output_meta` -> `build_zarr_sink` ->
#     sink_open!/write!/flush!/close!), the same surface `simulate` drives via
#     `sinks=`, flushed every FLUSH_EVERY records. Over a multi-day run the
#     difference between "flushed at the end" and "flushed as you go" is the
#     difference between a crash at hour 40 costing an hour and costing the run.
#     The sink is state-only in its current wave, so the store holds SuperFast
#     species + Transport3D.m; j-rates are observeds and do not appear.
#
# The console table is a domain-reduced digest of the same records -- solar
# geometry, O3 min/mean/max, OH (the radical that makes the day/night contrast
# unambiguous), NO2, and the continuity regression check -- for reading progress
# at a glance.
#
# THREADS. Run this under `julia -t 8` (or however many cores you have). The
# threaded codegen tier is what took the CONUS week run from ~1:1 realtime to
# ~40x; it activates only when EarthSciAST's Polyester extension is loadable, so
# the driver reports below whether it is actually on rather than letting a
# 10x-slow run look normal.
#
# Env:
#   RESEACT_MODEL      .esm to run                    (default: repo-root reseact.esm)
#   RESEACT_LABEL      tag for the RESULT line        (default "reseact")
#   RESEACT_LON0/LAT0  native GEOS-FP 4x5 slice origin (default 11/29 = CONUS;
#                      14 with NLON=7 is the old central-US box)
#   RESEACT_NLON/NLAT/NLEV  grid dims                 (default 13/7/72 = CONUS.
#                      Native 4x5 bounds: NLON <= 57, NLAT <= 17, NLEV <= 72.)
#   RESEACT_T0         start, s since 2016-01-01T00:00:00Z (default 5400 = 01:30Z).
#                      Also the model's solar epoch, so t IS the UTC clock.
#   RESEACT_SOLVE_SECS window length, s               (default 604800 = 7 days)
#   RESEACT_MACRO_DT   Lie-Trotter interval, s        (default 300)
#   RESEACT_MIN_MACRO_DT  bisection floor, s          (default 9 = five halvings)
#   RESEACT_JAC        "sym" (default), "ad" or "fd"  chemistry block Jacobian: EarthSciASTDiff's
#                      symbolic band program, ForwardDiff, or finite difference
#   RESEACT_JAC_CHECK  1: check the sym Jacobian against ForwardDiff's at T0 and stop
#                      on a mismatch (default 0; compiles the ForwardDiff Jacobian)
#   RESEACT_FORCING_TIME  "midstep" (default) or "stage": when the chemistry
#                      evaluates its time-dependent inputs -- once at the macro step's
#                      midpoint, as GEOS-Chem over its chemistry step, or at each stage
#                      time, the exact reference (see where `fc` is built)
#   RESEACT_ZARR       gridded output store           (default: none)
#   RESEACT_OUT_EVERY  write one record every N macro steps  (default 12 = hourly)
#   RESEACT_FLUSH_EVERY  commit the sink every N written records (default 12)
#   RESEACT_CSV        also write the digest as CSV   (default: none)
#   RESEACT_STATEDUMP  raw state written iff the solve fails  (default: none)
#   RESEACT_RUN_ENV    Julia env to activate          (default: <repo>/run-model-jl)
#
# Helper code it pulls in (see HELPERS.md for the migration plan):
#   prototypes/reseact_3d_chem/split_common.jl     prepare_split_docs, build_split_run,
#                                                  reseact_forcing, native_slice,
#                                                  validate_reseact, hydrostatic_dp
#   prototypes/reseact_3d_chem/blockdiag_local.jl  BlockDiagonal (from EarthSciMLBase)
#   prototypes/reseact_3d_chem/block_jac.jl        cellmajor_perm/rhs, block_{fd,ad,sym}_jac
#   tools/rx_band_program.jl                       band_program (the default RESEACT_JAC=sym)
#   tools/reactant_handoff/op_split.jl             lie_trotter_solve{,_bisect}
#   tools/grid_resize.jl                           slice_hybrid_coefs (NLEV < 72)
# ===========================================================================
import Pkg
const REPO = @__DIR__
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io = devnull)
# The kernel-class merge stays ON. It used to be forced off here, inheriting a
# default measured on a 7x7x7 box where the merged codegen tier's first-call
# compile swamped a ~25 s interpreted solve -- a real result, for a grid small
# enough that the unmerged IR was affordable in the first place. At CONUS that
# trade reverses: the unmerged per-cell IR is the thing that does not fit, and
# paying a fixed compile to keep the kernel count grid-independent is the whole
# point of the merge. ESS_KERNEL_CLASS_MERGE_DISABLE=1 forces the old behaviour.
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf, Statistics
# EarthSciIO keeps its compressors as WEAKDEPS so a base install stays light; the
# Zarr sink's default :diagnostic profile is Blosc-zstd, and loading Blosc is what
# activates EarthSciIOBloscExt to supply the encode. Without it sink_flush! throws.
using Blosc
# Polyester activates EarthSciASTPolyesterExt, the THREADED codegen tier (on
# EarthSciAST before 2026-10-03; later versions always carry the tier). It is a
# weakdep, so `using` it is the whole activation -- and it is worth ~10x on a long
# run, which is too large a difference to leave to whether the caller remembered.
# Loaded before the build, because the tier has to be live when the kernels are
# generated, not merely by the time they are called. If it is not installed in the
# active environment this stays false and the banner below says so out loud.
const POLYESTER_OK = try; @eval using Polyester; true; catch; false; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(RXDIR, "op_split.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL      = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL      = get(ENV, "RESEACT_LABEL", "reseact")
# One week of GEOS-FP, starting 01:30Z on 2016-01-01. T0=5400 is not arbitrary:
# it is the first A3 record centre, so the very first macro step already has both
# bracketing records of every cadence. `reseact_forcing(...; ndays)` hands the
# providers a `t -> url` resolver and EarthSciIO locates each record inside its
# OWN day's file, so the span is bounded only by how much data is fetched --
# `forcing_days_for` sizes that from the window. A window that crosses a MONTH
# boundary throws by design (NEI is a frozen monthly inventory), which is why the
# week starts on the 1st.
const T0         = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "604800"))
const MACRO_DT   = parse(Float64, get(ENV, "RESEACT_MACRO_DT", "300"))
const CSV        = get(ENV, "RESEACT_CSV", "")
const ZARR       = get(ENV, "RESEACT_ZARR", "")
const OUT_EVERY  = parse(Int, get(ENV, "RESEACT_OUT_EVERY", "12"))     # hourly at macro_dt=300
const FLUSH_EVERY = parse(Int, get(ENV, "RESEACT_FLUSH_EVERY", "12"))  # sink writes per flush
const STATEDUMP  = get(ENV, "RESEACT_STATEDUMP", "")   # raw state written iff the solve fails
# Floor on the bisection retry: five halvings from 300 s. A run needing chemistry
# steps shorter than this has a different problem, and should fail loudly.
const MIN_MACRO_DT = parse(Float64, get(ENV, "RESEACT_MIN_MACRO_DT", "9.0"))
# The chemistry block Jacobian, checked here so a bad value fails before the build.
# The default is exact: the species here span thirty orders (NO titrates to ~1e-26
# through the night, OH lives at 1e-13) and no finite difference step is
# defensible across that range -- see block_jac.jl. RESEACT_JAC=fd forces the old
# finite-difference block Jacobian for an A/B.
# "sym" (the default since 2026-11-12, as in the compiled runner and the adjoint
# driver) evaluates EarthSciASTDiff's symbolic Jacobian of the chemistry part
# (`block_sym_jac`). "ad", the ForwardDiff Jacobian and the default before it,
# stays as the A/B. Measured on the cluster, CONUS, 1 day, each sym run against an
# ad run of the same settings: every digest bit-identical, RESEACT_JAC_CHECK worst
# difference 0; time loop 4x5 8 threads 110.2 -> 82.9 s, 4x5 32 threads 83.9 ->
# 56.3 s, 2x2.5 8 threads 437.3 -> 317.2 s, 2x2.5 32 threads 286.8 -> 209.3 s
# (-25 to -33 %); the first macro step 23-37 s shorter (no Dual compile), against
# 52-78 s to build the band program. The whole run is about even for 1 day at 4x5
# on 8 threads, and shorter for longer runs, more threads or finer grids (2x2.5
# on 8 threads 1,211 -> 1,035 s).
const JACMODE = get(ENV, "RESEACT_JAC", "sym")
JACMODE in ("ad", "fd", "sym") || error("RESEACT_JAC must be ad, fd or sym, got $JACMODE")
const JAC_CHECK = get(ENV, "RESEACT_JAC_CHECK", "0") == "1"
_env(k, d) = parse(Int, get(ENV, "RESEACT_$k", string(d)))
_envi(k)   = haskey(ENV, "RESEACT_$k") ? parse(Int, ENV["RESEACT_$k"]) : nothing
# One origin -> metaparameters + the degree-space parameters + the index base
# hydrostatic_dp and continuity_drift read. See `native_slice`: the .esm cannot
# tie the two currencies together, so nothing else may spell them out.
# RESEACT_RES picks the GEOS-FP grid row (GEOSFP_GRIDS in split_common.jl:
# 4x5, 2x2.5, 0.25x0.3125, 0.25x0.3125_CH). The row carries the URL, the cell
# spacings, the native extent the halo bounds are checked against AND the CONUS
# index box, so LON0/LAT0/NLON/NLAT only need setting to depart from that box --
# hence `nothing` rather than the old 4x5 literals, which would have meant a
# different domain at every other resolution.
const RES   = get(ENV, "RESEACT_RES", "4x5")
const SLICE = native_slice(res = RES, lon0 = _envi("LON0"), lat0 = _envi("LAT0"),
                           nlon = _envi("NLON"), nlat = _envi("NLAT"),
                           nlev = _env("NLEV", 72))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const T_END = T0 + SOLVE_SECS
const RTOL, ATOL, ATOL_T = 1e-4, 1e-9, 1e-6
# GENERIC LU, not LUFactorization: on the BlockDiagonal W, LUFactorization calls
# the non-mutating lu(W), which copies every cell's block into a fresh result one
# block at a time; GenericLUFactorization reaches EarthSciMLBase's fused
# generic_lufact!, which factors the blocks in place, threaded. Same step counts,
# digest equal to round-off, 16% off the CONUS time loop.
const LU = LinearSolve.GenericLUFactorization()

# FastJX's solar chain (NOAA Spencer, as GasChem's FastJX), mirrored here ONLY to
# label the output rows with the sun angle the run is seeing (FastJX's cosSZA is
# an internal observed, not a state, so it is not in `u`). Same constants and
# formulas, with the hour taken continuously where FastJX reads the clock in
# whole seconds; the label is the domain centre's.
const K_GAMMA, D2R = 0.01721420632103996, 0.017453292519943295
# The digest's cos_sza column labels ONE point -- the domain centre -- so a wider
# slice does not go on being described by the sun over the middle of the old one.
const LON_MID = sum(SLICE.lon_deg) / 2
const LAT_MID = sum(SLICE.lat_deg) / 2
function cos_sza(t, lat, lon)
    g = K_GAMMA * (-12 / 24 + t / 86400)                       # doy0=1, hour0=0
    dec = 0.006918 - 0.399912cos(g) + 0.070257sin(g) - 0.006758cos(2g) +
          0.000907sin(2g) - 0.002697cos(3g) + 0.00148sin(3g)
    eqt = 229.18 * (0.000075 + 0.001868cos(g) - 0.032077sin(g) -
                    0.014615cos(2g) - 0.040849sin(2g))
    w = D2R * ((t / 60 + eqt + 4lon) / 4 - 180)
    return clamp(sin(D2R * lat) * sin(dec) + cos(D2R * lat) * cos(dec) * cos(w), -1, 1)
end

const NDAYS = forcing_days_for(T0, T_END)
say("=== $LABEL : build ($(basename(MODEL))) grid=$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF " *
    "T0=$(round(Int,T0)) window=$(round(SOLVE_SECS/3600, digits=2)) h macro_dt=$(round(Int,MACRO_DT)) ===")
say(@sprintf("    slice: lon %.1f..%.1f, lat %.1f..%.1f (native origin %d,%d); forcing spans %d daily files",
    SLICE.lon_deg[1], SLICE.lon_deg[2], SLICE.lat_deg[1], SLICE.lat_deg[2],
    SLICE.lon0, SLICE.lat0, NDAYS))
# Threading is the difference between ~40x realtime and ~1:1, so say which one
# this process is going to be BEFORE spending hours in it.
# EarthSciAST up to 2026-10-03 ships the tier as the EarthSciASTPolyesterExt
# weakdep extension; from e278a0183 on Polyester is a hard dependency and the
# tier is always built in, so either form counts.
let nt = Threads.nthreads(),
    ext = POLYESTER_OK && (Base.get_extension(EA, :EarthSciASTPolyesterExt) !== nothing ||
                           isdefined(EA, :Polyester))
    say(@sprintf("    threads=%d  polyester_ext=%s", nt, ext))
    (nt > 1 && ext) ||
        say("    !! SLOW CONFIGURATION -- the threaded codegen tier is off. Relaunch " *
            "with `julia -t 8`, in an environment where Polyester is installed " *
            "(see HELPERS.md). A week run takes roughly 10x longer without it.")
end
validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
docs, parts = prepare_split_docs(MODEL; metaparameters = GRID_MP, return_parts = true)
ff = reseact_forcing(CHEMDIR; ndays = NDAYS, res = SLICE.res)
# When NLEV<72 the `lev` axis is shorter than the 72-entry hybrid table; slice the
# vertical coefs to match (a truncated column, k=1 = surface).
ff = merge(ff, (; const_arrays = GridResize.slice_hybrid_coefs(ff.const_arrays, NLEV_EFF)))
# Pull every file the cadence will need BEFORE the solve. A mid-solve fetch is
# not wrong, but it puts a multi-hour run at the mercy of the network at an
# arbitrary macro step; this fails fast instead, while nothing is invested.
let tp = time(), n = sum(length(EarthSciIO.prefetch(pr)) for pr in values(ff.providers))
    say(@sprintf("    prefetch: %d provider-files warm in %.1f s", n, time() - tp))
end
# EarthSciAST splits each generated function into pieces of at most this many
# emitted nodes (default 20,000). For this model 10,000 compiles the first step
# ~175 s faster (612-632 s -> 444-454 s on the CONUS 13x7x30 test), loop unchanged,
# results bit-identical; 5,000 makes the loop 2.7x slower. `get!` keeps a cap
# already set in the environment.
get!(ENV, "ESS_CODEGEN_FN_NODE_CAP", "10000")
tb = time()
run = build_split_run(docs, (T0, T_END);
    providers = ff.providers, const_arrays = ff.const_arrays,
    parameters = ff.parameters, slice = SLICE)
say(@sprintf("BUILD %.2f s  nstates=%d", time() - tb, length(run.u0)))

# RESEACT_JAC=sym: the band program of the chemistry part, built :inplace from the
# SAME split part, overrides and live buffers as part 2's RHS (run.build_kwargs),
# with its own DiscreteMaterializer. It joins run.dms, so the forcing refresh
# re-materializes it with both RHS halves.
JACMODE == "sym" && include(joinpath(REPO, "tools", "rx_band_program.jl"))
BAND = nothing
if JACMODE == "sym"
    tj = time()
    BAND = band_program(parts[2]; source_layout = (u0 = run.u0, p = run.p, var_map = run.var_map),
                        build_kwargs = (; run.build_kwargs..., materialize_out = EA.DiscreteMaterializer()),
                        form = :inplace)
    push!(run.dms, BAND.dmJ)
    say(@sprintf("    band program %.1f s (document %.1f, build %.1f, pattern %.1f, scatter %.1f)  %s",
                 time() - tj, BAND.timings.doc, BAND.timings.build, BAND.timings.pattern,
                 BAND.timings.scatter, BAND))
end

# --------------------------------------------------------------------------- #
# Cell-major reorder + block chemistry Jacobian (the block-diagonal structure is
# only contiguous once the state is permuted species-major -> cell-major).
# --------------------------------------------------------------------------- #
P = cellmajor_perm(run.var_map)
f_trans_cm! = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
f_chem_cm!  = cellmajor_rhs(run.funcs[2], P.sm_of_cm)
# The chemistry W's block LU and solve run on Polyester's worker pool (MapPolyester), the pool EarthSciAST's kernels
# use: handed to ordinary Julia tasks between those kernels they wait on its spinning workers and run nearly serially.
# Bit-identical digest, 20% off the CONUS time loop on top of the generic LU. Needs an EarthSciMLBase checkout with
# MapPolyester (src/map_algorithm.jl + ext/polyester_map.jl; see README). Under sym the Jacobian's per-cell block fill
# runs on the same map (block_sym_jac).
jac_cm!, mkjp = JACMODE == "fd"  ? block_fd_jac(f_chem_cm!, P.NS, P.NC; alg = MapPolyester()) :
                JACMODE == "sym" ? block_sym_jac(BAND, P.sm_of_cm, P.NS, P.NC; alg = MapPolyester()) :
                                   block_ad_jac(run.funcs[2], P.sm_of_cm, P.NS, P.NC; alg = MapPolyester())
say("    chem Jacobian: " * (JACMODE == "fd"  ? "finite difference" :
                             JACMODE == "sym" ? "symbolic (EarthSciASTDiff band program, exact)" :
                                                "ForwardDiff (exact)") * "; block LU/solve on MapPolyester")
u = run.u0[P.sm_of_cm]
foreach(d -> d.materialize!(), run.dms)
# RESEACT_JAC_CHECK=1: the symbolic Jacobian against ForwardDiff's, at T0 on the
# initial state. Each entry's difference is taken relative to the largest entry of
# its cell's ForwardDiff block. Off by default: it compiles the ForwardDiff
# Jacobian (~40 s at CONUS).
"Worst |a - b| / (largest |b| in b's cell block) over NS x NS x NC block data, and its (row, col, cell)."
function block_rel_diff(a::AbstractArray{<:Real,3}, b::AbstractArray{<:Real,3})
    worst = 0.0; at = (0, 0, 0)
    for c in axes(b, 3)
        sc = maximum(abs, view(b, :, :, c))
        for s in axes(b, 2), r in axes(b, 1)
            d = abs(a[r, s, c] - b[r, s, c])
            e = d == 0 ? 0.0 : (isfinite(d) && sc > 0 ? d / sc : Inf)
            e > worst && (worst = e; at = (r, s, c))
        end
    end
    return worst, at
end
JAC_CHECK && JACMODE != "sym" && say("    RESEACT_JAC_CHECK applies to RESEACT_JAC=sym only; ignored")
if JAC_CHECK && JACMODE == "sym"
    let tc = time(), uc = run.u0[P.sm_of_cm], Js = mkjp()
        jad!, mkad = block_ad_jac(run.funcs[2], P.sm_of_cm, P.NS, P.NC; alg = MapPolyester())
        Ja = mkad()
        jac_cm!(Js, uc, run.p, T0)
        jad!(Ja, uc, run.p, T0)
        worst, at = block_rel_diff(Js.data, Ja.data)
        loc = worst == 0 ? "" : " at d $(P.bases[at[1]]) / d $(P.bases[at[2]]), cell $(P.cells[at[3]])"
        say(@sprintf("    RESEACT_JAC_CHECK: sym vs ForwardDiff at T0: worst |difference| / block max %.3e%s; nonzeros %d vs %d  %s  (%.1f s)",
                     worst, loc, count(!iszero, Js.data), count(!iszero, Ja.data),
                     worst <= 1e-10 ? "PASS" : "FAIL", time() - tc))
        worst <= 1e-10 || error("RESEACT_JAC_CHECK: the symbolic chemistry Jacobian differs from ForwardDiff's by $worst$loc")
    end
end
# m(0) from the REAL GEOS-FP PS, not a constant -- see hydrostatic_dp.
let dp0 = hydrostatic_dp(run.merged_param, ff.const_arrays, T0; slice = run.slice),
    mb = P.base_pos["Transport3D.m"]
    for c in P.cells
        u[(P.cell_pos[c] - 1) * P.NS + mb] = dp0(c[1], c[2], c[3])
    end
end
o3rng = P.base_pos["SuperFast.O3"]:P.NS:P.N
ohrng = P.base_pos["SuperFast.OH"]:P.NS:P.N
no2rng = P.base_pos["SuperFast.NO2"]:P.NS:P.N

tgz(g, u, p, t) = (fill!(g, 0); nothing)   # chem tgrad is zero (autonomous over a step)
# WHEN THE CHEMISTRY SEES ITS TIME-DEPENDENT INPUTS. "midstep" (the default) evaluates the
# time-blended GEOS-FP fields, the sun angle and Fast-JX photolysis, the emissions and the
# rate constants they feed ONCE per macro step, at its midpoint, for every chemistry RHS
# and Jacobian call in it -- what GEOS-Chem does over its chemistry timestep (fullchem:
# Update_RCONST once per box before KPP's Integrate, ICNTRL(15) = -1 so no per-call
# updates, photolysis at SUNCOS_MID). The chemistry is then autonomous over the step, so
# tgz above is exact. "stage" evaluates them at every Rosenbrock stage time: the exact
# reference, and the default until 2026-11-09. Transport is unchanged in both. Measured at
# CONUS 13x7x30, 1 day, 8 threads: time loop 232.0 -> 125.9 s, chemistry steps 9,092 ->
# 5,140; surface O3 mean -8e-5 ppb (max 0.017 ppb), NO2 RMS relative 0.17 %. Over 5 days
# of CONUS (2016-01-17 + 120 h) the loop is -48 % on every day and the difference from
# stage does not accumulate: surface O3 mean -8e-5 ppb on day 1, -4.2e-4 ppb on day 5.
const FORCING_TIME = get(ENV, "RESEACT_FORCING_TIME", "midstep")
FORCING_TIME in ("stage", "midstep") ||
    error("RESEACT_FORCING_TIME must be stage or midstep, got $FORCING_TIME")
const TFORCE = Ref(NaN)          # the current macro step's midpoint, set in the loop below
f_chem_run!, jac_run! = FORCING_TIME == "midstep" ?
    let f = f_chem_cm!, j = jac_cm!
        ((du, u, p, t) -> f(du, u, p, TFORCE[]), (J, u, p, t) -> j(J, u, p, TFORCE[]))
    end : (f_chem_cm!, jac_cm!)
say("    chemistry time-dependent inputs at: " *
    (FORCING_TIME == "midstep" ?
        "the macro-step midpoint, as GEOS-Chem over its chemistry step (RESEACT_FORCING_TIME=midstep)" :
        "each stage time, the exact reference (RESEACT_FORCING_TIME=stage)"))
FORCING_TIME == "midstep" && MACRO_DT > 600 &&
    say("    !! RESEACT_MACRO_DT=$(round(Int, MACRO_DT)) s under midstep: the chemistry's " *
        "time-dependent inputs are held fixed over each whole macro step (GEOS-Chem-like at 300 s)")
# `lie_trotter_solve` refreshes the forcing at any GEOS-FP boundary INSIDE a macro
# step, and the rest of that step would then read the step's midpoint against a
# bracket that no longer contains it. The default lattice (T0 on the cadence,
# MACRO_DT dividing 1800) puts every boundary on a macro edge; refuse one that does not.
if FORCING_TIME == "midstep"
    let inside = [b for b in run.tstops if T0 + 1e-6 < b < T_END - 1e-6 &&
                                           abs(rem(b - T0, MACRO_DT, RoundNearest)) > 1e-6]
        isempty(inside) ||
            error("RESEACT_FORCING_TIME=midstep: $(length(inside)) GEOS-FP refresh times fall " *
                  "inside a macro step (first at t=$(first(inside))); pick RESEACT_T0 / " *
                  "RESEACT_MACRO_DT so every refresh lands on a macro-step edge, or run " *
                  "with RESEACT_FORCING_TIME=stage")
    end
end
fc = SciMLBase.ODEFunction(f_chem_run!; jac = jac_run!, jac_prototype = mkjp(), tgrad = tgz)
inner_algs = (OrdinaryDiffEqSSPRK.SSPRK43(),
              OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff = false, linsolve = LU))
# Forcing refresh at cadence boundaries: rewrite the live buffers in
# run.merged_param from the discrete providers at time t, then rebuild the derived
# discrete caches -- the functional equivalent of run.cb's affect!, which the
# split solver cannot fire.
disc = Dict(String(k) => prov for (k, prov) in ff.providers if !EA.provider_is_const(prov))
refresh_forcing = t -> begin
    for (k, prov) in disc
        run.merged_param[k] .= EA._provider_const_field(EA.provider_sample(prov, t), k)
    end
    foreach(d -> d.materialize!(), run.dms)
end

# --- gridded streaming output (EarthSciAST sink protocol; EarthSciIO Zarr v3) ---
# The sink writes the flat state in var_map (species-major) order, but the solve
# runs cell-major, so un-permute before every write.
to_sm(ucm) = (usm = similar(ucm); @inbounds for i in 1:P.N; usm[P.sm_of_cm[i]] = ucm[i]; end; usm)
sink = nothing
if !isempty(ZARR)
    out_times = collect(T0:(MACRO_DT * OUT_EVERY):T_END)
    sink = EA.build_zarr_sink(run.var_map, ZARR; output_times = out_times,
                              meta = EA.derive_output_meta(docs[1]))
    EA.sink_open!(sink)
    say("  gridded output -> $ZARR  ($(length(out_times)) records)")
end
sink_put!(t, ucm) = sink === nothing ? nothing :
    EA.sink_write!(sink, EA.StateSnapshot(Float64(t),
        [(to_sm(ucm), (1:P.N,))], Dict{String,Array}()))

# --------------------------------------------------------------------------- #
# CONTINUITY DIAGNOSTIC.
#
# The drift between the integrated air mass m and the hydrostatic thickness
# dp = dA[k] + dB[k]*PS(t) that m is supposed to equal. This is the quantity the
# pressure fixer exists to kill, and it is what long runs used to die of: the
# face fluxes are built from GEOS-FP's time-AVERAGED A3 winds while PS is
# INSTANTANEOUS I3, so the raw discrete identity d(dp)/dt + div(M) = 0 does not
# hold (Jockel et al. 2001) and m wandered away from dp at ~3e-6 1/s.
#
# With the column-local fixer in place -- Mz diagnosed from the CORRECTED
# horizontal divergence and the continuity equation reading the same divh_fix --
# this should now sit at ROUNDOFF (measured 1.3e-17 1/s rms on the RHS itself;
# see tools/continuity_residual.jl). So the numbers below are no longer a
# characterisation of a known defect, they are a REGRESSION CHECK: anything that
# climbs off 1e-16 means the fixer has been broken, most likely by the m equation
# and Mz falling out of step about which divergence they use.
#
# Interior and wall are reported separately. The open lateral walls prescribe
# inflow tracer mixing ratios (qbc_*) but take the AIR-MASS flux straight from the
# wind field, which is not constrained to close the interior column budget -- a
# different defect from an interior continuity bias, and one a pressure fixer
# would NOT correct. Which of the two moved is what decides where to look.
# --------------------------------------------------------------------------- #
const _PSk = "GEOSFP.PS"
const _dA = Float64.(ff.const_arrays["Transport3D.dA"])
const _dB = Float64.(ff.const_arrays["Transport3D.dB"])
const NLON_ = maximum(c[1] for c in P.cells); const NLAT_ = maximum(c[2] for c in P.cells)
"w_I3 for the I3 cadence: 3-hourly instantaneous anchored at 00:00Z."
w_I3(t) = (t - 10800.0 * floor(t / 10800.0)) / 10800.0
function continuity_drift(u, t)
    F = run.merged_param[_PSk]          # NATIVE [time, lat, lon], hPa
    w = w_I3(t)
    s2i = 0.0; ni = 0; s2w = 0.0; nw = 0; worsti = 0.0; wci = (0, 0, 0)
    isw(c) = c[1] == 1 || c[1] == NLON_ || c[2] == 1 || c[2] == NLAT_
    for c in P.cells
        i, j, k = c
        ps = 100.0 * ((1 - w) * F[1, SLICE.lat0 + j, SLICE.lon0 + i] +
                            w * F[2, SLICE.lat0 + j, SLICE.lon0 + i])
        dp = _dA[k] + _dB[k] * ps
        mv = u[(P.cell_pos[c] - 1) * P.NS + P.base_pos["Transport3D.m"]]
        r = (mv - dp) / dp
        if isw(c)
            s2w += r * r; nw += 1
        else
            s2i += r * r; ni += 1
            abs(r) > abs(worsti) && (worsti = r; wci = c)
        end
    end
    return (rms_int = sqrt(s2i / max(ni, 1)), rms_wall = sqrt(s2w / max(nw, 1)),
            worst_int = worsti, cell_int = wci)
end

rows = NamedTuple[]
push_row!(t, u, wall) = push!(rows, (t = t, hours = t / 3600,
    cos_sza = cos_sza(t, LAT_MID, LON_MID),
    o3_min = minimum(u[o3rng]), o3_mean = mean(u[o3rng]), o3_max = maximum(u[o3rng]),
    oh_max = maximum(u[ohrng]), no2_mean = mean(u[no2rng]),
    m_min = minimum(u[P.base_pos["Transport3D.m"]:P.NS:P.N]), wall = wall))
report(t, u, wall) = begin
    push_row!(t, u, wall); local r = rows[end]; local d = continuity_drift(u, t)
    say(@sprintf("   %6.2f  %7.4f  %9.5f  %9.5f  %9.5f  %10.3e  %9.3e  | %8.2e %9.2e %9.2e %-12s %6.1f",
        r.hours, r.cos_sza, r.o3_min, r.o3_mean, r.o3_max, r.oh_max, r.no2_mean,
        d.rms_int, d.rms_wall, d.worst_int, string(d.cell_int), r.wall))
end

# --------------------------------------------------------------------------- #
# The macro-step loop.
# --------------------------------------------------------------------------- #
say("  t_hours  cos_sza     O3_min     O3_mean     O3_max      OH_max     NO2_mean  | rms_int  rms_wall  worst_int @cell_int    wall_s")
report(T0, u, 0.0); sink_put!(T0, u)

ts = time(); nT = nC = 0; ok = true; nstep = 0; nbisect = 0
t = T0
while t < T_END - 1e-9
    global nstep += 1
    tnext = min(t + MACRO_DT, T_END)
    # Read only under RESEACT_FORCING_TIME=midstep. A bisected macro step keeps the
    # midpoint of the WHOLE step [t, tnext] for both halves: the retry exists to get
    # the same map through the solver, not to change the chemistry's inputs.
    TFORCE[] = t + (tnext - t) / 2
    res = lie_trotter_solve_bisect(f_trans_cm!, fc, u, (t, tnext), run.p, inner_algs;
        macro_dt = tnext - t, min_macro_dt = MIN_MACRO_DT,
        reltols = (RTOL, RTOL), abstols = (ATOL_T, ATOL),
        refresh = refresh_forcing, forcing_tstops = run.tstops, clamp_nonneg = true,
        on_bisect = (a, b) -> (global nbisect += 1;
                               say(@sprintf("    bisecting [%.0f, %.0f]", a, b))))
    # A macro step that did not succeed must NOT be papered over by advancing t:
    # the state comes back unchanged, so the run marches on producing a frozen
    # trajectory that still LOOKS like a completed simulation. Report which
    # species is responsible and stop.
    if !(res.retcode == SciMLBase.ReturnCode.Success ||
         res.retcode == SciMLBase.ReturnCode.Default)
        global ok = false
        say("  !! macro step [$t, $tnext] retcode=$(res.retcode) -- diagnosing state:")
        for (nm, b) in sort(collect(P.base_pos), by = last)
            v = res.u[b:P.NS:P.N]
            say(@sprintf("     %-22s min=%- .6e max=%- .6e  nneg=%d nnan=%d",
                nm, minimum(v), maximum(v), count(<(0), v), count(isnan, v)))
        end
        # Dump the raw CELL-MAJOR state that failed. A failure reached after hours
        # of simulation is otherwise reproducible only by re-simulating to it.
        # Dump the PRE-STEP state `u`, not `res.u`: `lie_trotter_solve` returns
        # `copy(integ.u)` taken AFTER the failed `step!` and the clamp, so `res.u`
        # is where the integrator gave up -- past the transition, and restarting
        # from it does NOT reproduce the failure. `u` is the last state a macro
        # step accepted, so restarting there re-runs exactly the step that failed.
        # Both are written: `u` to restart from, `res.u` because it is what the
        # diagnostic above prints.
        if !isempty(STATEDUMP)
            open(STATEDUMP, "w") do io
                write(io, Int64(P.NS), Int64(P.NC), Float64(t))
                write(io, u)          # pre-step, RESTARTABLE
                write(io, res.u)      # post-failure, matches the printout above
            end
            open(STATEDUMP * ".names", "w") do io
                println(io, "# NS=$(P.NS) NC=$(P.NC) t=$t  layout: cell-major, u[(c-1)*NS+s]")
                for (nm, b) in sort(collect(P.base_pos), by = last)
                    println(io, b, "\t", nm)
                end
            end
            say("  !! failing state dumped -> $STATEDUMP")
        end
        say("  !! stopping at t=$t (see above); partial series below")
        break
    end
    global u = res.u; global nT += res.naT; global nC += res.naC
    global t = tnext
    report(t, u, time() - ts)
    if nstep % OUT_EVERY == 0
        sink_put!(t, u)
        sink === nothing || (nstep % (OUT_EVERY * FLUSH_EVERY) == 0 && EA.sink_flush!(sink))
    end
end
solve_s = time() - ts
if sink !== nothing
    EA.sink_flush!(sink); EA.sink_close!(sink)
    say("  gridded output committed: $ZARR")
end

o3m = [r.o3_mean for r in rows]
say(@sprintf("RESULT label=%s cells=%d nstates=%d NS=%d window_h=%.2f macro_dt=%.0f nmacro=%d nT=%d nC=%d nbisect=%d solve_s=%.1f",
    LABEL, P.NC, P.N, P.NS, SOLVE_SECS / 3600, MACRO_DT, length(rows) - 1, nT, nC, nbisect, solve_s))
say(@sprintf("DIURNAL O3_mean: start=%.5f min=%.5f max=%.5f end=%.5f  peak-to-trough=%.5f ppb  ok=%s",
    o3m[1], minimum(o3m), maximum(o3m), o3m[end], maximum(o3m) - minimum(o3m),
    ok && all(isfinite, u)))

if !isempty(CSV)
    open(CSV, "w") do io
        println(io, "t,hours,cos_sza,o3_min,o3_mean,o3_max,oh_max,no2_mean,m_min,wall")
        for r in rows
            println(io, join((r.t, r.hours, r.cos_sza, r.o3_min, r.o3_mean, r.o3_max,
                              r.oh_max, r.no2_mean, r.m_min, r.wall), ","))
        end
    end
    say("wrote $CSV")
end
say("DONE $LABEL")
