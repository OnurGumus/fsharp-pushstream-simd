namespace PushStream6.SimdBenchmark

open System
open System.Numerics
open System.Threading.Tasks
open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Running

// ---------------------------------------------------------------------------
// A generic SIMD push stream over any primitive 'T that Vector<'T> supports.
//
// The scalar PushStream's receiver is ('T -> bool): one element at a time, so
// nothing downstream can vectorize. Here the *source owns the loop*, so it can
// push Vector<'T> BLOCKS to a vector-receiver and the leftover elements to a
// scalar-receiver. InlineIfLambda fuses the per-block lambda away exactly like
// the scalar version, so the whole pipeline collapses to a vector loop + tail.
//
// No `bool` early-exit: these are drain/reduce pipelines (sum/min/max/...),
// the natural fit for SIMD (early-exit per element defeats packing). `filter`
// is deliberately absent — variable output count breaks vector packing.
// ---------------------------------------------------------------------------
module SimdStream =

  type S<'T when 'T : struct> = (Vector<'T> -> unit) -> ('T -> unit) -> unit

  // Source: stream a 'T[] as full Vector<'T> blocks + a scalar remainder.
  let inline ofArray (xs : 'T[]) : S<'T> =
    fun ([<InlineIfLambda>] vr) ([<InlineIfLambda>] sr) ->
      let n = Vector<'T>.Count
      let mutable i = 0
      while i + n <= xs.Length do
        vr (Vector<'T>(xs, i))
        i <- i + n
      while i < xs.Length do
        sr xs.[i]
        i <- i + 1

  // map: a vector op for the blocks + the matching scalar op for the tail.
  let inline map
      ([<InlineIfLambda>] vf : Vector<'T> -> Vector<'U>)
      ([<InlineIfLambda>] sf : 'T -> 'U)
      ([<InlineIfLambda>] ps : S<'T>) : S<'U> =
    fun ([<InlineIfLambda>] vr) ([<InlineIfLambda>] sr) ->
      ps (fun v -> vr (vf v)) (fun x -> sr (sf x))

  // Generic reduction with explicit combiners — fully type-agnostic, no SRTP:
  //   vzero/vcomb : vector accumulator;  szero/scomb : scalar-tail accumulator;
  //   horiz       : fold the final vector lanes into one scalar.
  let inline reduce
      (vzero : Vector<'T>) ([<InlineIfLambda>] vcomb : Vector<'T> -> Vector<'T> -> Vector<'T>)
      (szero : 'T)         ([<InlineIfLambda>] scomb : 'T -> 'T -> 'T)
      ([<InlineIfLambda>] horiz : Vector<'T> -> 'T)
      ([<InlineIfLambda>] ps : S<'T>) : 'T =
    let mutable vacc = vzero
    let mutable sacc = szero
    ps (fun v -> vacc <- vcomb vacc v) (fun x -> sacc <- scomb sacc x)
    scomb (horiz vacc) sacc

  // Parallel SIMD reduce: partition the array, run the *same fused vector kernel*
  // on each partition across cores, then combine the per-partition results.
  //
  // Fusion holds INSIDE each partition (vf/sf are InlineIfLambda and this fn is
  // inline, so the kernel inlines into the Parallel.For body). Only the cross-
  // partition dispatch goes through a delegate — its cost is amortized over a
  // whole chunk. This is a MONOID-ONLY operation: (vzero,vcomb)/(szero,scomb)
  // must be associative so partitions can be combined in any order. Early-exit
  // and order-dependent ops (scan/pairwise/dot) are deliberately not expressible
  // here — that is the honest boundary of the push model under partitioning.
  let inline parReduce
      (xs : 'T[])
      (vzero : Vector<'T>) ([<InlineIfLambda>] vf : Vector<'T> -> Vector<'T>)
                           ([<InlineIfLambda>] vcomb : Vector<'T> -> Vector<'T> -> Vector<'T>)
      (szero : 'T)         ([<InlineIfLambda>] sf : 'T -> 'T)
                           ([<InlineIfLambda>] scomb : 'T -> 'T -> 'T)
      ([<InlineIfLambda>] horiz : Vector<'T> -> 'T) : 'T =
    let len = xs.Length
    let n = Vector<'T>.Count
    // ~64k elements/worker minimum: tiny inputs stay single-threaded (no Task cost).
    let workers = max 1 (min Environment.ProcessorCount ((len + 65535) / 65536))
    if workers = 1 then
      let mutable vacc = vzero
      let mutable i = 0
      while i + n <= len do vacc <- vcomb vacc (vf (Vector<'T>(xs, i))); i <- i + n
      let mutable sacc = szero
      while i < len do sacc <- scomb sacc (sf xs.[i]); i <- i + 1
      scomb (horiz vacc) sacc
    else
      let chunk = (len + workers - 1) / workers
      let partials = Array.create workers szero
      Parallel.For(0, workers, fun p ->
        let lo = p * chunk
        let hi = min len (lo + chunk)
        let mutable vacc = vzero
        let mutable i = lo
        while i + n <= hi do vacc <- vcomb vacc (vf (Vector<'T>(xs, i))); i <- i + n
        let mutable sacc = szero
        while i < hi do sacc <- scomb sacc (sf xs.[i]); i <- i + 1
        partials.[p] <- scomb (horiz vacc) sacc)
      |> ignore
      Array.fold scomb szero partials

  // Fusion-preserving pipe (same trick as the scalar library).
  let inline (|>>) ([<InlineIfLambda>] v : _ -> _) ([<InlineIfLambda>] f : _ -> _) = f v

  // Horizontal lane fold for min/max, given a "is a better than b" predicate.
  let inline private horizBest (v : Vector<'T>) (seed : 'T) ([<InlineIfLambda>] better) =
    let mutable m = seed
    for j in 0 .. Vector<'T>.Count - 1 do
      if better v.[j] m then m <- v.[j]
    m

  // ---- concrete sinks built on the generic core (inline => stay fused) ----

  let inline sumI ([<InlineIfLambda>] ps : S<int>) : int =
    reduce Vector<int>.Zero (fun a b -> a + b) 0 (+) (fun v -> Vector.Sum v) ps

  let inline maxI ([<InlineIfLambda>] ps : S<int>) : int =
    reduce (Vector<int>(Int32.MinValue)) (fun a b -> Vector.Max(a, b))
           Int32.MinValue max (fun v -> horizBest v Int32.MinValue (fun a b -> a > b)) ps

  let inline minI ([<InlineIfLambda>] ps : S<int>) : int =
    reduce (Vector<int>(Int32.MaxValue)) (fun a b -> Vector.Min(a, b))
           Int32.MaxValue min (fun v -> horizBest v Int32.MaxValue (fun a b -> a < b)) ps

  // Same generic core instantiated at float (double) — proves the generality.
  let inline sumF ([<InlineIfLambda>] ps : S<float>) : float =
    reduce Vector<float>.Zero (fun a b -> a + b) 0.0 (+) (fun v -> Vector.Sum v) ps

  // Dot product needs TWO sources in lockstep. A single-source push stream
  // cannot express zip (the "needs a PumpStream" case from the design notes),
  // so dot is a direct vector routine rather than a stream pipe.
  let dotI (xs : int[]) (ys : int[]) : int =
    let len = min xs.Length ys.Length
    let n = Vector<int>.Count
    let mutable acc = Vector<int>.Zero
    let mutable i = 0
    while i + n <= len do
      acc <- acc + Vector<int>(xs, i) * Vector<int>(ys, i)
      i <- i + n
    let mutable s = Vector.Sum acc
    while i < len do
      s <- s + xs.[i] * ys.[i]
      i <- i + 1
    s

// Sum-of-squares via the SIMD push stream (int and float).
module Simd =
  open SimdStream
  let sumSqI (xs : int[])   = ofArray xs |>> map (fun v -> v * v) (fun x -> x * x) |>> sumI
  let sumSqF (xs : float[]) = ofArray xs |>> map (fun v -> v * v) (fun x -> x * x) |>> sumF

// Sum-of-squares via the scalar fused PushStream (the |>> tier).
module Scalar =
  open PushStream6.PushStream
  let sumSqI (xs : int[]) = ofArray xs |>> map (fun x -> x * x) |>> fold (+) 0

// A compute-bound, fully-vectorizable per-element kernel. A short fixed-length
// affine recurrence (STEPS fused multiply-adds) — heavy enough that the work is
// dominated by arithmetic, not memory bandwidth, which is the regime where
// adding cores actually pays. Scalar and vector forms do identical math, so
// every lane/partition produces the same per-element value (only the SUM
// reorders, hence the float tolerance in the sanity gate).
module Kernel =
  [<Literal>]
  let STEPS = 128
  let inline scalar (x : float) =
    let mutable t = x
    for _ in 1 .. STEPS do t <- t * 0.999 + 0.001
    t
  let inline vector (x : Vector<float>) =
    let a = Vector<float>(0.999)
    let b = Vector<float>(0.001)
    let mutable t = x
    for _ in 1 .. STEPS do t <- t * a + b
    t

  // Same math, but EXPLICIT fused multiply-add. RyuJIT won't auto-contract
  // `t*a + b` into one FMA (IEEE: FMA rounds once, mul-then-add twice), so the
  // plain version above emits vmulpd + vaddpd. These force the single vfmadd /
  // scalar FMA the hardware (FMA3, present since Haswell) already supports.
  let inline scalarFma (x : float) =
    let mutable t = x
    for _ in 1 .. STEPS do t <- System.Math.FusedMultiplyAdd(t, 0.999, 0.001)
    t
  let inline vectorFma (x : Vector<float>) =
    let a = Vector<float>(0.999)
    let b = Vector<float>(0.001)
    let mutable t = x
    for _ in 1 .. STEPS do t <- Vector.FusedMultiplyAdd(t, a, b)
    t

// The 2x2: {sequential, parallel} x {scalar, SIMD}, all summing Kernel over a float[].
module Compute =
  open SimdStream

  // sequential + SIMD: the existing fused SIMD push stream, with the heavy kernel as map.
  let seqSimdSum (xs : float[]) =
    ofArray xs |>> map Kernel.vector Kernel.scalar |>> sumF

  // parallel + SIMD: same fused kernel, partitioned across cores.
  let parSimdSum (xs : float[]) =
    parReduce xs
      Vector<float>.Zero Kernel.vector (fun a b -> a + b)
      0.0                Kernel.scalar (+)
      (fun v -> Vector.Sum v)

  // ...and the same two, but with the kernel using explicit fused multiply-add.
  let seqSimdFmaSum (xs : float[]) =
    ofArray xs |>> map Kernel.vectorFma Kernel.scalarFma |>> sumF
  let parSimdFmaSum (xs : float[]) =
    parReduce xs
      Vector<float>.Zero Kernel.vectorFma (fun a b -> a + b)
      0.0                Kernel.scalarFma (+)
      (fun v -> Vector.Sum v)

  // parallel + scalar: threads but no vectorization (isolates the cores axis).
  let parScalarSum (xs : float[]) =
    let len = xs.Length
    let workers = max 1 (min Environment.ProcessorCount ((len + 65535) / 65536))
    let chunk = (len + workers - 1) / workers
    let partials = Array.zeroCreate workers
    Parallel.For(0, workers, fun p ->
      let lo = p * chunk
      let hi = min len (lo + chunk)
      let mutable s = 0.0
      for i in lo .. hi - 1 do s <- s + Kernel.scalar xs.[i]
      partials.[p] <- s)
    |> ignore
    Array.sum partials

// ---------------------------------------------------------------------------
// Matrix multiply C = A*B (row-major float[]) — the canonical compute-bound,
// embarrassingly-parallel kernel, exercising the SAME two axes as parReduce:
//   * SIMD : the inner j-loop is a broadcast-FMA over Vector<float> blocks.
//            ikj loop order keeps B's row and C's row contiguous, so the inner
//            loop vectorizes cleanly (a column-wise ijk order would not).
//   * cores: output rows are independent, so Parallel.For over i has zero
//            contention — each task writes a disjoint slice of C, no combine.
// ---------------------------------------------------------------------------
module MatMul =

  // One output row i, vectorized: broadcast A[i,l], stream B's row l, FMA into C's row i.
  let kernelRow (a:float[]) (b:float[]) (c:float[]) k n i =
    let w = Vector<float>.Count
    let cRow = i * n
    for l in 0 .. k - 1 do
      let ail  = a.[i * k + l]
      let va   = Vector<float>(ail)
      let bRow = l * n
      let mutable j = 0
      while j + w <= n do
        (Vector<float>(c, cRow + j) + va * Vector<float>(b, bRow + j)).CopyTo(c, cRow + j)
        j <- j + w
      while j < n do
        c.[cRow + j] <- c.[cRow + j] + ail * b.[bRow + j]
        j <- j + 1

  // Same row, scalar (no Vector) — the honest baseline.
  let scalarRow (a:float[]) (b:float[]) (c:float[]) k n i =
    let cRow = i * n
    for l in 0 .. k - 1 do
      let ail  = a.[i * k + l]
      let bRow = l * n
      for j in 0 .. n - 1 do
        c.[cRow + j] <- c.[cRow + j] + ail * b.[bRow + j]

  let seqScalar a b (c:float[]) m k n =
    System.Array.Clear(c, 0, c.Length)
    for i in 0 .. m - 1 do scalarRow a b c k n i
  let seqSimd a b (c:float[]) m k n =
    System.Array.Clear(c, 0, c.Length)
    for i in 0 .. m - 1 do kernelRow a b c k n i
  let parScalar a b (c:float[]) m k n =
    System.Array.Clear(c, 0, c.Length)
    Parallel.For(0, m, fun i -> scalarRow a b c k n i) |> ignore
  let parSimd a b (c:float[]) m k n =
    System.Array.Clear(c, 0, c.Length)
    Parallel.For(0, m, fun i -> kernelRow a b c k n i) |> ignore

// Compute-bound 2x2: sequential vs parallel, crossed with scalar vs SIMD.
// SIMD multiplies throughput per core; cores multiply across partitions; the
// two compose (ParSimd ~= cores x lanes over the scalar baseline).
[<MemoryDiagnoser>]
type ParBench() =

  [<Params(1000000, 8000000)>]
  member val Length = 0 with get, set

  member val Fs : float[] = [||] with get, set

  [<GlobalSetup>]
  member this.Setup() =
    let r = Random 42
    this.Fs <- Array.init this.Length (fun _ -> r.NextDouble())

  [<Benchmark(Baseline = true)>]
  member this.SeqScalar() =
    let xs = this.Fs
    let mutable s = 0.0
    for i in 0 .. xs.Length - 1 do s <- s + Kernel.scalar xs.[i]
    s

  [<Benchmark>]
  member this.SeqSimd() = Compute.seqSimdSum this.Fs

  [<Benchmark>]
  member this.SeqSimdFma() = Compute.seqSimdFmaSum this.Fs

  [<Benchmark>]
  member this.ParScalar() = Compute.parScalarSum this.Fs

  [<Benchmark>]
  member this.ParSimd() = Compute.parSimdSum this.Fs

  [<Benchmark>]
  member this.ParSimdFma() = Compute.parSimdFmaSum this.Fs

// Matrix multiply, same 2x2. ParSimd here is cores x lanes on a real algorithm.
[<MemoryDiagnoser>]
type MatMulBench() =

  [<Params(512, 1024)>]
  member val N = 0 with get, set

  member val A : float[] = [||] with get, set
  member val B : float[] = [||] with get, set
  member val C : float[] = [||] with get, set

  [<GlobalSetup>]
  member this.Setup() =
    let n = this.N
    let r = Random 42
    this.A <- Array.init (n * n) (fun _ -> r.NextDouble())
    this.B <- Array.init (n * n) (fun _ -> r.NextDouble())
    this.C <- Array.zeroCreate (n * n)

  [<Benchmark(Baseline = true)>]
  member this.SeqScalar() = MatMul.seqScalar this.A this.B this.C this.N this.N this.N
  [<Benchmark>]
  member this.SeqSimd()   = MatMul.seqSimd   this.A this.B this.C this.N this.N this.N
  [<Benchmark>]
  member this.ParScalar() = MatMul.parScalar this.A this.B this.C this.N this.N this.N
  [<Benchmark>]
  member this.ParSimd()   = MatMul.parSimd   this.A this.B this.C this.N this.N this.N

[<MemoryDiagnoser>]
type SimdBench() =

  [<Params(1000, 1000000)>]
  member val Length = 0 with get, set

  member val Xs : int[] = [||] with get, set
  member val Fs : float[] = [||] with get, set

  [<GlobalSetup>]
  member this.Setup() =
    let r = Random 42
    this.Xs <- Array.init this.Length (fun _ -> r.Next(0, 4))
    this.Fs <- Array.init this.Length (fun _ -> float (r.Next(0, 4)))

  // ---- int sum-of-squares: scalar baseline vs SIMD ceiling vs streams ----

  [<Benchmark(Baseline = true)>]
  member this.HandScalar() =
    let xs = this.Xs
    let mutable s = 0
    for i in 0 .. xs.Length - 1 do
      s <- s + xs.[i] * xs.[i]
    s

  [<Benchmark>]
  member this.HandSimd() =
    let xs = this.Xs
    let n = Vector<int>.Count
    let mutable acc = Vector<int>.Zero
    let mutable i = 0
    while i + n <= xs.Length do
      let v = Vector<int>(xs, i)
      acc <- acc + v * v
      i <- i + n
    let mutable s = Vector.Sum acc
    while i < xs.Length do
      s <- s + xs.[i] * xs.[i]
      i <- i + 1
    s

  [<Benchmark>]
  member this.ScalarFused() = Scalar.sumSqI this.Xs

  [<Benchmark>]
  member this.SimdStream() = Simd.sumSqI this.Xs

  // ---- float instantiation of the SAME generic stream core ----

  [<Benchmark>]
  member this.HandScalarF() =
    let xs = this.Fs
    let mutable s = 0.0
    for i in 0 .. xs.Length - 1 do
      s <- s + xs.[i] * xs.[i]
    s

  [<Benchmark>]
  member this.SimdStreamF() = Simd.sumSqF this.Fs

module Main =
  open SimdStream

  [<EntryPoint>]
  let main argv =
    // Correctness gate across every sink/type: a fast-but-wrong impl fails here.
    let r = Random 1
    let xs = Array.init 1234 (fun _ -> r.Next(0, 100))
    let ys = Array.init 1234 (fun _ -> r.Next(0, 100))
    let fs = xs |> Array.map float

    let refSumSq = (let mutable s = 0   in (for x in xs do s <- s + x * x); s)
    let refSumSqF= (let mutable s = 0.0 in (for x in fs do s <- s + x * x); s)
    let refMax   = Array.max xs
    let refMin   = Array.min xs
    let refDot   = (let mutable s = 0 in (for i in 0 .. xs.Length - 1 do s <- s + xs.[i] * ys.[i]); s)

    let okSumSq  = Simd.sumSqI xs = refSumSq && Scalar.sumSqI xs = refSumSq
    let okSumSqF = Simd.sumSqF fs = refSumSqF
    let okMax    = maxI (ofArray xs) = refMax
    let okMin    = minI (ofArray xs) = refMin
    let okDot    = dotI xs ys = refDot

    // Compute-kernel paths: float sum reassociates across lanes/partitions, so
    // compare within a relative tolerance rather than for bit-equality.
    let cs = Array.init 100000 (fun i -> float i / 100000.0)
    let refCompute = (let mutable s = 0.0 in (for x in cs do s <- s + Kernel.scalar x); s)
    let approx a = abs (a - refCompute) <= 1e-6 * (abs refCompute + 1.0)
    let okSeqSimd   = approx (Compute.seqSimdSum cs)
    let okParScalar = approx (Compute.parScalarSum cs)
    let okParSimd   = approx (Compute.parSimdSum cs)
    // FMA changes rounding (one rounding vs two), so it only matches within tolerance.
    let okSeqFma    = approx (Compute.seqSimdFmaSum cs)
    let okParFma    = approx (Compute.parSimdFmaSum cs)

    // Matrix multiply: SIMD/parallel reassociate the dot-product sums, so compare
    // every variant to the scalar reference within a float tolerance.
    let nn = 64
    let ma = Array.init (nn * nn) (fun i -> float (i % 7) - 3.0)
    let mb = Array.init (nn * nn) (fun i -> float (i % 5) - 2.0)
    let refMM = (let c = Array.zeroCreate (nn * nn) in MatMul.seqScalar ma mb c nn nn nn; c)
    let mmClose f =
      let c = Array.zeroCreate (nn * nn)
      f ma mb c nn nn nn
      Array.forall2 (fun x y -> abs (x - y) <= 1e-9 * (abs y + 1.0)) c refMM
    let okMatMul = mmClose MatMul.seqSimd && mmClose MatMul.parScalar && mmClose MatMul.parSimd

    if not (okSumSq && okSumSqF && okMax && okMin && okDot
            && okSeqSimd && okParScalar && okParSimd && okSeqFma && okParFma && okMatMul) then
      failwithf "Mismatch: sumSq=%b sumSqF=%b max=%b min=%b dot=%b seqSimd=%b parScalar=%b parSimd=%b seqFma=%b parFma=%b matmul=%b"
        okSumSq okSumSqF okMax okMin okDot okSeqSimd okParScalar okParSimd okSeqFma okParFma okMatMul
    printfn "Sanity OK — sumSq=%d sumSqF=%g max=%d min=%d dot=%d compute=%g; Vector<int>.Count=%d, Vector<float>.Count=%d, cores=%d"
      refSumSq refSumSqF refMax refMin refDot refCompute Vector<int>.Count Vector<float>.Count Environment.ProcessorCount

    BenchmarkSwitcher.FromTypes([| typeof<SimdBench>; typeof<ParBench>; typeof<MatMulBench> |]).Run(argv) |> ignore
    0
