namespace PushStream6.SimdBenchmark

open System
open System.Numerics
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
  let main _ =
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

    if not (okSumSq && okSumSqF && okMax && okMin && okDot) then
      failwithf "Mismatch: sumSq=%b sumSqF=%b max=%b min=%b dot=%b" okSumSq okSumSqF okMax okMin okDot
    printfn "Sanity OK — sumSq=%d sumSqF=%g max=%d min=%d dot=%d; Vector<int>.Count=%d, Vector<float>.Count=%d"
      refSumSq refSumSqF refMax refMin refDot Vector<int>.Count Vector<float>.Count

    BenchmarkRunner.Run<SimdBench>() |> ignore
    0
